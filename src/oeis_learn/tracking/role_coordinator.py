"""Host-owned role views and kill deadlines. No Docker socket enters any role.

One 88-GiB learner/generator/evaluator plus <=8 one-GiB candidate workers.
Worker requests are exact source/indices/limits messages, never host commands.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import threading
from contextlib import ExitStack
from queue import Queue

from oeis_learn.experiments.artifacts import ArtifactPath, canonical_bytes, atomic_write, load_json
from oeis_learn.sandbox.pipeline import ExecutionEvidence, Limits, digest_bytes
from oeis_learn.tracking.foundation_container import container_command
from oeis_learn.tracking.foundation_metrics import EventLog

OPERATIONS = {"engineering", "diagnostic", "train", "generate", "evaluate", "probe", "smoke"}


def copy_view(root, files):
    """Host creates an enumerated capability view, never mounts a repository/cohort root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for name, source in files.items():
        target = ArtifactPath(root, name).as_path()
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError("role views require regular explicit files")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


def reclaim(name, process):
    try:
        removed = subprocess.run(["docker", "rm", "--force", name], timeout=2, capture_output=True)
        if removed.returncode:
            raise RuntimeError("Docker cgroup reclamation failed")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)


class DockerRunner:
    """Host candidate service; each request is durably issued before its container starts."""

    def __init__(self, root, image, *, workers=4):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("worker cap8")
        self.root, self.image = Path(root), image
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = EventLog(self.root / "events")
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.worker_count = workers
        self.mutex = threading.Lock()
        self.slots = Queue()
        self.services = []
        self.started = False

    def start(self):
        if self.started:
            return
        try:
            for number in range(self.worker_count):
                root = self.root / f"service-{uuid.uuid4().hex}"
                view, output = root / "input", root / "output"
                view.mkdir(parents=True)
                output.mkdir()
                name = "oeis-worker-" + uuid.uuid4().hex
                process = subprocess.Popen(
                    container_command(
                        role="worker",
                        image=self.image,
                        inputs=view,
                        output=output,
                        name=name,
                        command=["python", "-m", "oeis_learn.tracking.role_worker", "serve"],
                    ),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                slot = dict(name=name, process=process, view=view, output=output)
                self.services.append(slot)
            deadline = time.monotonic() + 20
            for slot in self.services:
                while not (slot["output"] / "ready.json").exists():
                    if time.monotonic() >= deadline or slot["process"].poll() is not None:
                        raise RuntimeError("bounded worker setup failed")
                    time.sleep(0.01)
                self.slots.put(slot)
            self.started = True
        except BaseException:
            self.close()
            raise

    def run_single(self, source, indices, *, request_id, limits=Limits(), deadline_ns=None):
        from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

        positions = list(indices)
        from oeis_learn.sandbox.wat_ast import parse

        parse(source)
        if (
            not positions
            or len(positions) > 100
            or any(type(i) is not int or not 0 <= i <= 99 for i in positions)
        ):
            raise ValueError("bounded rebased runtime request required")
        request = dict(
            source=source, indices=positions, limits=asdict(limits), request_id=request_id
        )
        key = digest(request)[7:]
        root = self.root / key
        root.mkdir(parents=True, exist_ok=True)
        event = "issue-" + digest(request_id)[7:]
        issued = dict(request_sha256=digest(request), issued_ns=time.time_ns())
        target = root / "result.json"
        with self.mutex:
            if event in self.log.events:
                if self.log.events[event]["request_sha256"] != issued["request_sha256"]:
                    raise ValueError("conflicting runtime request identity")
                completed = self.log.events.get("complete-" + digest(request_id)[7:])
                if completed is not None:
                    if not target.exists() or completed["result_sha256"] != digest(
                        load_json(target.read_bytes())
                    ):
                        raise ValueError("durable worker result changed")
                    return ExecutionEvidence(**load_json(target.read_bytes()))
                return self._lost(
                    source, positions, limits, "interrupted issued worker; no regeneration"
                )
            self.log.append(event, issued)
        start = time.monotonic_ns()
        deadline = (
            min(start + limits.deadline_ns, deadline_ns)
            if deadline_ns is not None
            else start + limits.deadline_ns
        )
        slot = None
        value = self._lost(source, positions, limits, "external_candidate_deadline")
        try:
            from queue import Empty

            try:
                slot = self.slots.get(timeout=max(0, deadline - time.monotonic_ns()) / 1e9)
            except Empty:
                pass
            if slot is not None:
                request_path = slot["view"] / (key + ".json")
                result_path = slot["output"] / (key + ".json")
                atomic_write(
                    ArtifactPath(slot["view"], request_path.name), canonical_bytes(request)
                )
                while time.monotonic_ns() < deadline:
                    if result_path.exists():
                        value = ExecutionEvidence(**load_json(result_path.read_bytes()))
                        if (
                            value.source_sha256 != digest_bytes(source.encode())
                            or value.indices != positions
                        ):
                            raise ValueError("worker response binding mismatch")
                        break
                    if slot["process"].poll() is not None:
                        raise RuntimeError("worker service exited")
                    time.sleep(0.001)
                else:
                    reclaim(slot["name"], slot["process"])
                    slot["reclaimed"] = True
                request_path.unlink(missing_ok=True)
                result_path.unlink(missing_ok=True)
        finally:
            if slot is not None and not slot.get("reclaimed"):
                self.slots.put(slot)
        value.elapsed_ns = time.monotonic_ns() - start
        atomic_write(ArtifactPath(root, "result.json"), canonical_bytes(value.to_dict()))
        with self.mutex:
            self.log.append(
                "complete-" + digest(request_id)[7:], dict(result_sha256=digest(value.to_dict()))
            )
        return value

    @staticmethod
    def _lost(source, indices, limits, reason):
        return ExecutionEvidence(
            digest_bytes(source.encode()),
            indices,
            fuel=limits.fuel_aggregate,
            reference_steps=limits.reference_aggregate,
            outcome="execution_limit",
            reason=reason,
        )

    def run_escalating(self, source, indices, **kwargs):
        from oeis_learn.evaluation.fuel_escalation import run_escalating

        return run_escalating(self, source, indices, **kwargs)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        failures = []
        for slot in self.services:
            if not slot.get("reclaimed"):
                try:
                    reclaim(slot["name"], slot["process"])
                    slot["reclaimed"] = True
                except BaseException as exc:
                    failures.append(exc)
        if failures:
            raise RuntimeError("worker cgroup reclamation failed") from failures[0]


class QueueRunner:
    """Learner-side transport with no Docker, evaluator or hidden-data capability."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_single(self, source, indices, *, request_id, limits=Limits(), deadline_ns=None):
        from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

        request = dict(
            source=source, indices=list(indices), request_id=request_id, limits=asdict(limits)
        )
        key = digest(request_id)[7:]
        atomic_write(ArtifactPath(self.root, key + ".request.json"), canonical_bytes(request))
        deadline = time.monotonic_ns() + limits.deadline_ns + 2_000_000_000
        if deadline_ns is not None:
            deadline = min(deadline, deadline_ns)
        path = self.root / (key + ".response.json")
        while time.monotonic_ns() < deadline:
            if path.exists():
                value = load_json(path.read_bytes())
                if value["request_sha256"] != digest(request):
                    raise ValueError("controller response binding mismatch")
                return ExecutionEvidence(**value["evidence"])
            time.sleep(0.01)
        return DockerRunner._lost(
            source, request["indices"], limits, "controller transport deadline"
        )

    def run_escalating(self, source, indices, **kwargs):
        from oeis_learn.evaluation.fuel_escalation import run_escalating

        return run_escalating(self, source, indices, **kwargs)

    def close(self):
        pass


class RoleCoordinator:
    def __init__(self, root, image, *, workers=4, render_node=None):
        self.root, self.image, self.render_node = Path(root), image, render_node
        self.root.mkdir(parents=True, exist_ok=True)
        self.runner = DockerRunner(self.root / "workers", image, workers=workers)
        self.log = EventLog(self.root / "roles")
        from oeis_learn.tracking.foundation_controller import run_lock

        self.locks = ExitStack()
        self.locks.enter_context(run_lock(self.root))
        self.active = threading.Lock()

    def run(self, operation, *, role, inputs, output, seconds, gpu=False):
        if not self.active.acquire(blocking=False):
            raise RuntimeError("only one main role may be live")
        try:
            return self._run(
                operation, role=role, inputs=inputs, output=output, seconds=seconds, gpu=gpu
            )
        finally:
            self.active.release()

    def _run(self, operation, *, role, inputs, output, seconds, gpu=False):
        if (
            operation not in OPERATIONS
            or type(seconds) not in (int, float)
            or not 0 < seconds <= 7200
        ):
            raise ValueError("bounded internal role operation required")
        # Only a single main role may exist at a time; invocation is synchronous.
        name = "oeis-role-" + uuid.uuid4().hex
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        command = ["python", "-m", "oeis_learn.tracking.role_worker", operation]
        argv = container_command(
            role=role,
            image=self.image,
            inputs=inputs,
            output=output,
            name=name,
            command=command,
            render_node=self.render_node if gpu else None,
            gpu=gpu,
        )
        queue = output / "runtime-queue"
        queue.mkdir()
        self.log.append(
            name,
            dict(
                operation=operation,
                role=role,
                seconds=seconds,
                image=self.image,
                started_ns=time.time_ns(),
            ),
        )
        start = time.monotonic()
        self.runner.start()
        if time.monotonic() >= start + seconds:
            raise TimeoutError("role setup exhausted external deadline")
        from oeis_learn.tracking.foundation_controller import ResourceGuard

        guard = ResourceGuard(self.root.parent, free_floor=20 << 30)
        process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        issued, futures = set(), {}
        low_samples, next_memory_sample = 0, start
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now >= start + seconds:
                    raise TimeoutError("external role deadline")
                if now >= next_memory_sample:
                    next_memory_sample = now + 1
                    resources = guard.check()
                    self.log.append(
                        name + "-resources-" + str(int(now - start)),
                        dict(
                            host=resources,
                            host_floor=16 << 30,
                            quota=200 << 30,
                            free_floor=20 << 30,
                        ),
                    )
                    try:
                        rows = dict(
                            l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines()
                        )
                        low_samples = (
                            low_samples + 1
                            if int(rows["MemAvailable"].split()[0]) * 1024 < 16 << 30
                            else 0
                        )
                    except (OSError, ValueError, KeyError):
                        raise RuntimeError("host reserve monitor unavailable")
                    if low_samples >= 2:
                        raise MemoryError("host available memory below16GiB for two samples")
                for path in sorted(queue.glob("*.request.json")):
                    if path.name in issued:
                        continue
                    if len(futures) >= 32:
                        raise RuntimeError("runtime request queue cap32")
                    if path.is_symlink() or path.stat().st_size > 1 << 20:
                        raise ValueError("invalid runtime request file")
                    request = load_json(path.read_bytes())
                    if set(request) != {"source", "indices", "request_id", "limits"}:
                        raise ValueError("runtime transport accepts no commands/paths/truth")
                    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

                    issued.add(path.name)
                    futures[path.name] = (
                        self.runner.executor.submit(
                            self.runner.run_single,
                            request["source"],
                            request["indices"],
                            request_id=name + "-" + request["request_id"],
                            limits=Limits(**request["limits"]),
                            deadline_ns=time.monotonic_ns()
                            + int(min(2, start + seconds - now) * 10**9),
                        ),
                        digest(request),
                    )
                for key, (future, binding) in list(futures.items()):
                    if future.done():
                        evidence = future.result().to_dict()
                        atomic_write(
                            ArtifactPath(queue, key.replace(".request", ".response")),
                            canonical_bytes(dict(request_sha256=binding, evidence=evidence)),
                        )
                        del futures[key]
                time.sleep(0.01)
            if process.returncode:
                raise RuntimeError("role container failed")
            return load_json((output / "result.json").read_bytes())
        finally:
            reclaim(name, process)
            for future, _ in futures.values():
                try:
                    future.result()
                except Exception:
                    pass
                #  # bounded worker deadline/reclamation, never leave a worker live
            self.log.append(
                name + "-stop",
                dict(elapsed_ns=int((time.monotonic() - start) * 1e9), reclaimed=True),
            )

    def close(self):
        try:
            self.runner.close()
        finally:
            self.locks.close()
