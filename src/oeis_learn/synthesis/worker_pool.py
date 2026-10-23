"""One aggregate owning-role supervisor; FIFO waits retain original deadlines."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import threading
import time

from .client import frame, FrameReader, encoded
from .contracts import ResourceProfile
from .admission import reference_identity
from .service_policy import ServiceLimits, RoleCapacity, BoundedCache
from .state import require_hash


class _SerialWorker:
    def __init__(
        self,
        *,
        command=None,
        contract_sha256,
        resources=ResourceProfile(),
        limits=ServiceLimits(),
        rss_bytes=1 << 30,
    ):
        require_hash(contract_sha256)
        if limits.max_inflight_batches != 1:
            raise ValueError("native-fitting/v1 supports one active batch per role")
        self.command = command or [
            os.environ.get("OEIS_SYNTHESIS_WORKER", "/opt/oeis-build-info/synthesis-worker")
        ]
        self.contract = contract_sha256
        self.resources = resources
        self.limits = limits
        self.rss_bytes = rss_bytes
        self.capacity = RoleCapacity(limits.max_pending_batches, limits.max_pending_bytes, 1)
        self.condition = threading.Condition()
        self.process = None
        self.generation = 0
        self.cancelled = set()
        self.closed = False
        self.stderr_tail = b""
        self.stats = dict(
            queue_wait_seconds=0.0, batches=0, restarts=0, stderr_bytes=0, rss_peak_bytes=None
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def cancel(self, key):
        with self.condition:
            self.cancelled.add(key)
            self.condition.notify_all()

    def cancel_all(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _retire(self):
        process = self.process
        self.process = None
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
            self.generation += 1
            self.stats["restarts"] += 1

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self._retire()

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        if self.process is not None:
            self._retire()
        env = dict(
            os.environ,
            OEIS_SYNTHESIS_CONTRACT=self.contract,
            OEIS_REFERENCE_SHA256=reference_identity(),
            OEIS_SYNTHESIS_RESOURCES=json.dumps(asdict(self.resources)),
        )
        allocator = getattr(self, 'generation_allocator', None)
        if allocator is not None:
            self.generation = allocator()
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
            env=env,
        )
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            os.set_blocking(stream.fileno(), False)

    def _rss(self):
        try:
            for line in Path(f"/proc/{self.process.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except FileNotFoundError:
            pass
        return None

    def call(self, request, *, deadline):
        key = request["request_id"]
        payload = frame(request)
        started = time.monotonic()
        ids = [r["candidate_id"] for r in request["candidates"]]
        if (
            not ids
            or len(ids) > 32
            or any(not isinstance(cid, str) for cid in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("bounded unique candidates required")
        if request["contract_sha256"] != self.contract:
            raise ValueError("contract mismatch")
        queued = False
        active = False
        results = {}
        reason = "backpressure_timeout"
        outcome = "resource_exhausted"
        complete = False
        try:
            with self.condition:
                while time.monotonic() < deadline and not self.closed and key not in self.cancelled:
                    now = time.monotonic()
                    if not queued:
                        queued = self.capacity.enqueue(
                            key, len(payload), deadline=deadline, now=now
                        )
                    if (
                        queued
                        and not self.capacity.active
                        and next(iter(self.capacity.pending), None) == key
                    ):
                        active = self.capacity.dispatch(now=now) == key
                        break
                    self.condition.wait(timeout=min(0.02, max(0, deadline - now)))
                if not active:
                    return self._response(results, ids, False, outcome, reason)
            self.stats["queue_wait_seconds"] += time.monotonic() - started
            self.stats["batches"] += 1
            self._start()
            request = dict(
                request,
                worker_generation=self.generation,
                remaining_budget=dict(
                    request["remaining_budget"],
                    wall_time_ms=max(
                        1,
                        min(
                            request["remaining_budget"]["wall_time_ms"],
                            int((deadline - time.monotonic()) * 1000),
                        ),
                    ),
                ),
            )
            payload = frame(request)
            reader = FrameReader()
            offset = 0
            stderr = 0
            self.stderr_tail = b""
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE, "input")
                selector.register(self.process.stdout, selectors.EVENT_READ, "output")
                selector.register(self.process.stderr, selectors.EVENT_READ, "error")
                while time.monotonic() < deadline:
                    if self.closed or key in self.cancelled:
                        reason = "cancelled"
                        break
                    guard = getattr(self, 'resource_guard', None)
                    if guard is not None:
                        guard()
                    rss = self._rss()
                    if rss is not None:
                        self.stats["rss_peak_bytes"] = max(rss, self.stats["rss_peak_bytes"] or 0)
                        if rss > self.rss_bytes:
                            reason = "worker_rss"
                            break
                    for event, _ in selector.select(
                        timeout=min(0.02, max(0, deadline - time.monotonic()))
                    ):
                        stream = event.fileobj
                        if event.data == "input":
                            try:
                                offset += os.write(
                                    stream.fileno(), payload[offset : offset + 65536]
                                )
                            except BlockingIOError:
                                continue
                            if offset == len(payload):
                                selector.unregister(stream)
                        else:
                            try:
                                data = os.read(stream.fileno(), 65536)
                            except BlockingIOError:
                                continue
                            if not data:
                                selector.unregister(stream)
                                if event.data == "output":
                                    raise ValueError("worker EOF before completion")
                                continue
                            if event.data == "error":
                                self.stderr_tail = (self.stderr_tail + data)[-4096:]
                                stderr += len(data)
                                self.stats["stderr_bytes"] += len(data)
                                if stderr > 65536:
                                    reason = "stderr_limit"
                                    raise TimeoutError(reason)
                            else:
                                for response in reader.feed(data):
                                    if any(
                                        response.get(k) != request[k]
                                        for k in (
                                            "protocol",
                                            "request_id",
                                            "worker_generation",
                                            "contract_sha256",
                                        )
                                    ):
                                        raise ValueError("stale or mismatched response")
                                    if (
                                        set(response)
                                        != {
                                            "protocol",
                                            "request_id",
                                            "worker_generation",
                                            "contract_sha256",
                                            "results",
                                            "unstarted",
                                            "complete",
                                        }
                                        or type(response["complete"]) is not bool
                                        or not isinstance(response["results"], list)
                                    ):
                                        raise ValueError("malformed response shape")
                                    for row in response.get("results", []):
                                        if not isinstance(row, dict) or not isinstance(
                                            row.get("candidate_id"), str
                                        ):
                                            raise ValueError("malformed result ID")
                                        cid = row["candidate_id"]
                                        if cid not in ids or cid in results:
                                            raise ValueError("duplicate/unknown result ID")
                                        results[cid] = row
                                    unstarted = response.get("unstarted")
                                    if (
                                        not isinstance(unstarted, list)
                                        or any(not isinstance(cid, str) for cid in unstarted)
                                        or set(unstarted) != set(ids) - set(results)
                                        or len(unstarted) != len(set(unstarted))
                                    ):
                                        raise ValueError("invalid partial coverage")
                                    if response.get("complete") is True:
                                        complete = True
                                        return self._response(results, ids, True, "ok", None)
                    if self.process.poll() is not None and not selector.get_map():
                        raise ValueError("worker exited")
                reason = "deadline" if reason == "backpressure_timeout" else reason
        except ValueError as exc:
            outcome = "verification_failure"
            reason = str(exc)
        except (OSError, TimeoutError, subprocess.SubprocessError) as exc:
            reason = str(exc)
        finally:
            if active and not complete:
                self._retire()
            with self.condition:
                if active:
                    self.capacity.finish(key)
                elif queued:
                    self.capacity.cancel(key)
                self.cancelled.discard(key)
                self.condition.notify_all()
        return self._response(results, ids, False, outcome, reason)

    def _response(self, results, ids, complete, outcome, reason):
        return dict(
            results=[results[k] for k in ids if k in results],
            unstarted=[k for k in ids if k not in results],
            complete=complete,
            outcome=outcome,
            reason=reason,
            generation=self.generation,
            metrics=dict(
                self.stats,
                stderr_tail=self.stderr_tail.decode("utf-8", errors="replace"),
                rss_unavailable_reason=None
                if self.stats["rss_peak_bytes"] is not None
                else "process RSS unavailable on this host",
            ),
        )


class WorkerPool(_SerialWorker):
    """Retained serial implementation; explicit 2/4-slot policies opt into shared ownership."""
    def __new__(cls, *args, **kwargs):
        limits = kwargs.get('limits', ServiceLimits())
        if limits.max_inflight_batches != 1:
            return NativeSlots(*args, **kwargs)
        return super().__new__(cls)


class NativeSlots:
    def __init__(self, *, command=None, contract_sha256, resources=ResourceProfile(),
                 limits=ServiceLimits(), rss_bytes=1 << 30):
        from dataclasses import replace
        from .service_policy import PipelineCapacity
        self.command=command or [os.environ.get('OEIS_SYNTHESIS_WORKER','/opt/oeis-build-info/synthesis-worker')]
        self.contract=contract_sha256;self.limits=limits;self.resources=resources
        self.capacity=PipelineCapacity(workers=limits.max_inflight_batches,
            pending=limits.max_pending_batches,pending_bytes=limits.max_pending_bytes)
        self.condition=threading.Condition(threading.RLock())
        self.local=threading.local();self.generation=0;self.closed=False;self.cancelled=set()
        self.slots=[_SerialWorker(command=self.command,contract_sha256=contract_sha256,
            resources=resources,limits=replace(limits,max_inflight_batches=1),rss_bytes=rss_bytes)
            for _ in range(limits.max_inflight_batches)]
        for slot in self.slots:
            slot.generation_allocator=self._generation
            slot.resource_guard=self._watch

    def _generation(self):
        with self.condition:
            self.generation+=1
            return self.generation

    @property
    def stats(self):
        return dict(slots=[dict(s.stats) for s in self.slots],workers=len(self.slots),
            queue_wait_seconds=sum(s.stats['queue_wait_seconds'] for s in self.slots),
            batches=sum(s.stats['batches'] for s in self.slots),
            restarts=sum(s.stats['restarts'] for s in self.slots))

    def _watch(self):
        observations=[s._rss() for s in self.slots if s.process is not None]
        if sum(v for v in observations if v is not None)>4 << 30:
            self._signal_all()
            raise MemoryError('aggregate native RSS watchdog')

    def cancel_all(self):
        self._signal_all()

    def _signal_all(self):
        for slot in self.slots:
            process=slot.process
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def affinity(self, ordinal):
        from contextlib import contextmanager
        @contextmanager
        def bound():
            previous=getattr(self.local,'prompt',0)
            self.local.prompt=ordinal
            try:
                yield
            finally:
                self.local.prompt=previous
        return bound()

    def call(self,request,*,deadline):
        key=request['request_id'];size=len(frame(request));prompt=getattr(self.local,'prompt',0)
        slot_id=prompt%len(self.slots);queued=False;active=False
        ids=[c['candidate_id'] for c in request['candidates']]
        def incomplete(reason):
            return dict(results=[],unstarted=ids,complete=False,outcome='resource_exhausted',reason=reason,generation=self.generation,metrics=self.stats)
        try:
            with self.condition:
                while not self.closed and key not in self.cancelled and time.monotonic()<deadline:
                    now=time.monotonic()
                    if not queued:
                        queued=self.capacity.enqueue(key,size,4<<20,prompt=prompt,deadline=deadline,now=now)
                    if queued:
                        # Slot FIFO is explicit; never dispatch a different caller's request.
                        head=next((k for k,r in self.capacity.pending.items() if r['slot']==slot_id),None)
                        if head==key and self.capacity.dispatch(slot_id,now=now)==key:
                            active=True;break
                    self.condition.wait(timeout=min(.02,max(0,deadline-now)))
                if not active:
                    return incomplete('cancelled' if self.closed or key in self.cancelled else 'queue_deadline')
            result=self.slots[slot_id].call(request,deadline=deadline)
            if time.monotonic()>=deadline:
                return incomplete('owner_deadline')
            return result
        finally:
            with self.condition:
                if active:
                    self.capacity.finish(key,acknowledged=True)
                    # Response transferred synchronously to its one caller; staging no longer owns it.
                    self.capacity.commit(key)
                elif queued:
                    self.capacity.cancel_pending(key)
                self.cancelled.discard(key);self.condition.notify_all()

    def cancel(self,key):
        with self.condition:
            self.cancelled.add(key)
            for slot in self.slots:
                slot.cancel(key)
            self.condition.notify_all()

    def close(self):
        started=time.monotonic()
        with self.condition:
            self.closed=True
            for slot in self.slots:
                slot.closed=True
            self.condition.notify_all()
        self._signal_all()  # Signal concurrently before waiting for any one process.
        for slot in self.slots:
            slot._retire()
        if time.monotonic()-started>5:
            raise TimeoutError('aggregate cleanup exceeded five seconds; external role watchdog required')

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


class RoleCaches:
    def __init__(self, limits, root=None):
        self.mutex = threading.RLock()
        self.result = BoundedCache(limits.result_cache_max_entries, limits.result_cache_max_bytes)
        self.compiled = BoundedCache(
            limits.compiled_cache_max_entries, limits.compiled_cache_max_bytes
        )
        self.disk = BoundedCache(limits.disk_cache_max_entries, limits.disk_cache_max_bytes)
        self.root = Path(root) if root is not None else None
        self.lock = None
        if self.root is not None:
            import fcntl

            self.root.mkdir(parents=True, exist_ok=True)
            self.lock = (self.root / ".owner.lock").open("a+b")
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Cache entries are expendable; reconstruct byte/count accounting from actual files.
            for p in sorted(self.root.glob("*.json")):
                try:
                    if p.is_symlink():
                        p.unlink()
                        continue
                    if p.stat().st_size > self.disk.max_bytes:
                        p.unlink()
                        continue
                    raw = p.read_bytes()
                    row = json.loads(raw)
                    key = row["key"]
                    require_hash(key)
                    if (
                        p.is_symlink()
                        or p.name != key.removeprefix("sha256:") + ".json"
                        or set(row) != {"key", "value"}
                        or len(raw) > 4 << 20
                    ):
                        p.unlink()
                        continue
                    if self.disk.reserve(key, len(raw)):
                        self.disk.publish(key, p.name)
                    else:
                        p.unlink()
                except (ValueError, KeyError):
                    p.unlink()
            self._prune()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def _prune(self):
        if self.root is not None:
            retained = {x[2] for x in self.disk.entries.values()}
            for path in self.root.glob("*.json"):
                if path.name not in retained:
                    path.unlink(missing_ok=True)

    def get(self, key):
        with self.mutex:
            return self._get(key)

    def _get(self, key):
        require_hash(key)
        value = self.result.get(key)
        if value is not None:
            return json.loads(value)
        name = self.disk.get(key)
        if name is not None and self.root is not None:
            try:
                path = self.root / name
                if path.is_symlink() or path.stat().st_size > 4 << 20:
                    raise ValueError("invalid disk cache entry")
                row = json.loads(path.read_bytes())
                if row["key"] == key:
                    return row["value"]
            except (FileNotFoundError, ValueError, KeyError):
                self.disk.entries.pop(key, None)
        return None

    def put(self, key, value):
        with self.mutex:
            return self._put(key, value)

    def _put(self, key, value):
        require_hash(key)
        raw = encoded(value)
        charge = len(encoded(key)) + len(raw)
        if self.result.reserve(key, charge):
            self.result.publish(key, raw)
        if self.root is None:
            return
        from oeis_learn.fresh.common import write

        body = encoded(dict(key=key, value=value))
        name = key.removeprefix("sha256:") + ".json"
        if not self.disk.reserve(key, len(body)):
            return
        self._prune()
        (self.root / name).unlink(missing_ok=True)
        try:
            write(self.root / name, body)
            self.disk.publish(key, name)
        except OSError:
            self.disk.abort(key)

    def metrics(self):
        result = {
            name: dict(
                entries=len(cache.entries),
                bytes=cache.used_bytes,
                hits=cache.hits,
                misses=cache.misses,
                evictions=cache.evictions,
                bypasses=cache.bypasses,
            )
            for name, cache in [
                ("result", self.result),
                ("compiled", self.compiled),
                ("disk", self.disk),
            ]
        }
        result["compiled"]["reason"] = (
            "native runtime owns ephemeral compiled modules; no serialized artifacts published"
        )
        return result
