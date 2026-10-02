"""Bounded local worker service with external deadlines and durable idempotency.

Workers have no WASI/guest I/O and a 1 GiB address-space ceiling. Network/device/
mount/seccomp isolation is the separate T046 container boundary; this class does
not claim that spawning a process alone establishes container isolation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import multiprocessing as mp
import os
from pathlib import Path
from queue import Queue, Full
import resource
import socket
import struct
import threading
import time

from oeis_learn.experiments.artifacts import (
    ArtifactPath,
    atomic_write,
    canonical_bytes,
    compute_canonical_digest,
    load_json,
)
from oeis_learn.sandbox.pipeline import (
    ExecutionEvidence,
    Limits,
    Runtime,
    digest_bytes,
    runtime_identity,
)

MAX_MESSAGE = 1 << 20


def _send(channel, value):
    data = canonical_bytes(value)
    if len(data) > MAX_MESSAGE:
        raise ValueError("IPC message exceeds 1 MiB")
    channel.sendall(struct.pack("!I", len(data)) + data)


def _receive(channel, deadline=None):
    def read(size):
        chunks = bytearray()
        while len(chunks) < size:
            if deadline is not None:
                remaining = (deadline - time.monotonic_ns()) / 1e9
                if remaining <= 0:
                    raise TimeoutError("worker deadline")
                channel.settimeout(remaining)
            part = channel.recv(size - len(chunks))
            if not part:
                raise EOFError("worker closed channel")
            chunks.extend(part)
        return bytes(chunks)

    length = struct.unpack("!I", read(4))[0]
    if length > MAX_MESSAGE:
        raise ValueError("worker message exceeds 1 MiB")
    return load_json(read(length))


def _worker(channel, cache_bytes):
    # Native compiler/execution stays single-threaded. Set limits before engine
    # creation, so compilation, reference execution and cache all share the cap.
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    runtime = Runtime(cache_bytes=cache_bytes)
    _send(channel, {"type": "ready", "pid": os.getpid()})
    try:
        while True:
            request = _receive(channel)
            if request.get("type") == "close":
                return

            def progress(evidence):
                _send(channel, {"type": "progress", "evidence": evidence})

            try:
                result = runtime.evaluate(
                    request["source"],
                    request["indices"],
                    Limits(**request["limits"]),
                    deadline_ns=request["deadline_ns"],
                    progress=progress,
                    proposed_source=request.get("proposed_source"),
                )
                _send(channel, {"type": "result", "evidence": result.to_dict()})
            except MemoryError:
                _send(channel, {"type": "worker_error", "reason": "worker_address_space_limit"})
                return  # potentially compromised allocator state is never reused
            except (ValueError, RuntimeError) as exc:
                _send(
                    channel,
                    {
                        "type": "worker_error",
                        "reason": f"worker_failure:{type(exc).__name__}:{exc}",
                    },
                )
                return
    except (EOFError, BrokenPipeError):
        return
    finally:
        channel.close()


class _Slot:
    def __init__(self, cache_bytes, target):
        self.cache_bytes, self.target = cache_bytes, target
        self.process = self.channel = None
        self.start()

    def start(self):
        parent, child = socket.socketpair()
        process = mp.get_context("spawn").Process(
            target=self.target, args=(child, self.cache_bytes), daemon=True
        )
        process.start()
        child.close()
        self.process, self.channel = process, parent
        try:
            ready = _receive(parent, time.monotonic_ns() + 2_000_000_000)
            if ready.get("type") != "ready":
                raise RuntimeError("invalid worker startup acknowledgment")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        start = time.monotonic()
        if self.channel is not None:
            self.channel.close()
        if self.process is not None:
            if self.process.is_alive():
                self.process.kill()
            self.process.join(timeout=2)
            if self.process.is_alive():
                raise RuntimeError("worker reclamation exceeded two seconds")
            self.process.close()
        self.process = self.channel = None
        return time.monotonic() - start


class WorkerPool:
    """At most eight single-request workers, at most 32 outstanding requests.

    Result IDs identify attempts (not source programs); caller must use a new ID
    for a new generation attempt. Duplicate delivery reuses immutable evidence,
    including after controller restart, and conflicting reuse fails closed.
    """

    def __init__(self, result_dir: Path, *, workers=1, cache_bytes=1 << 30, _worker_target=_worker):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("worker count must be in 1..8")
        if type(cache_bytes) is not int or not 0 <= cache_bytes <= 1 << 30:
            raise ValueError("aggregate cache must not exceed 1 GiB")
        self.root = Path(result_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime_sha256 = compute_canonical_digest(runtime_identity())
        self.limit = threading.BoundedSemaphore(32)
        self.lock = threading.Lock()
        self.inflight = {}
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.slots = Queue(maxsize=workers)
        self.all_slots = []
        self.closed = False
        try:
            for _ in range(workers):
                slot = _Slot(cache_bytes // workers, _worker_target)
                self.all_slots.append(slot)
                self.slots.put(slot)
        except BaseException:
            self.close()
            raise

    def submit(self, source, indices, *, request_id, limits=Limits(), proposed_source=None):
        if not isinstance(source, str) or (
            proposed_source is not None and not isinstance(proposed_source, str)
        ):
            raise ValueError("source must be text")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("bounded nonempty attempt identity required")
        indices = list(indices)
        if (
            not indices
            or len(indices) > 100
            or any(type(i) is not int or not 0 <= i <= 99 for i in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("invalid indices")
        payload = {
            "source": source,
            "indices": indices,
            "limits": asdict(limits),
            "proposed_source": proposed_source,
        }
        if len(canonical_bytes(payload)) + 256 > MAX_MESSAGE:
            raise ValueError("request exceeds 1 MiB")
        fingerprint = compute_canonical_digest(
            {"request": payload, "runtime_sha256": self.runtime_sha256}
        )
        key = digest_bytes(request_id.encode())[7:]
        with self.lock:
            if self.closed:
                raise RuntimeError("worker pool is closed")
            if key in self.inflight:
                previous, future = self.inflight[key]
                if previous != fingerprint:
                    raise ValueError("attempt identity reused with different request")
                return future
            if not self.limit.acquire(blocking=False):
                raise Full("32-request bound reached")
            future = self.executor.submit(self._run, key, fingerprint, payload, limits)
            self.inflight[key] = (fingerprint, future)

        # register outside the lock: callbacks may run immediately
        def complete(_):
            with self.lock:
                self.inflight.pop(key, None)
            self.limit.release()

        future.add_done_callback(complete)
        return future

    def _run(self, key, fingerprint, payload, limits):
        path = ArtifactPath(self.root, key + ".json")
        if path.as_path().exists():
            record = load_json(path.as_path().read_bytes())
            if record["request_sha256"] != fingerprint:
                raise ValueError("attempt identity reused with different request")
            if record["evidence_sha256"] != compute_canonical_digest(record["evidence"]):
                raise ValueError("corrupt persisted execution evidence")
            return ExecutionEvidence(**record["evidence"])
        slot = self.slots.get()
        progress = None
        start = time.monotonic_ns()
        deadline = start + limits.deadline_ns
        broken = False
        try:
            if slot.process is None:
                slot.start()
                # Startup/replacement is pool overhead, not guest execution.
                start = time.monotonic_ns()
                deadline = start + limits.deadline_ns
            slot.channel.settimeout(max(1e-9, (deadline - time.monotonic_ns()) / 1e9))
            _send(slot.channel, {**payload, "deadline_ns": deadline})
            while True:
                message = _receive(slot.channel, deadline)
                if message["type"] == "progress":
                    progress = message["evidence"]
                elif message["type"] == "result":
                    result = ExecutionEvidence(**message["evidence"])
                    break
                else:
                    raise RuntimeError(message.get("reason", "invalid worker message"))
        except (TimeoutError, EOFError, OSError, ValueError, RuntimeError) as exc:
            broken = True
            timed_out = isinstance(exc, TimeoutError)
            if progress is not None:
                result = ExecutionEvidence(**progress)
            else:
                final = (
                    payload["proposed_source"]
                    if payload["proposed_source"] is not None
                    else payload["source"]
                )
                result = ExecutionEvidence(
                    digest_bytes(final.encode()),
                    payload["indices"],
                    fuel=None,
                    reference_steps=None,
                )
            result.outcome = (
                "execution_limit"
                if timed_out or "address_space_limit" in str(exc)
                else "runtime_failure"
            )
            result.reason = "external_candidate_deadline" if timed_out else f"worker_failure:{exc}"
            result.failure_stage = "execute"
            result.first_failure_index = payload["indices"][
                min(len(result.outputs), len(payload["indices"]) - 1)
            ]
            result.elapsed_ns = time.monotonic_ns() - start
        finally:
            if broken:
                slot.stop()  # Must reap before exposing a reusable slot.
            self.slots.put(slot)
        record = {
            "request_sha256": fingerprint,
            "evidence": result.to_dict(),
            "evidence_sha256": compute_canonical_digest(result.to_dict()),
        }
        atomic_write(path, canonical_bytes(record))
        return result

    def evaluate(self, source, indices, **kwargs):
        return self.submit(source, indices, **kwargs).result()

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        for slot in self.all_slots:
            slot.stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
