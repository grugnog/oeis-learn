#!/usr/bin/env python3
"""Rootless host coordinator, extracted from the pinned image by lab-gpu.

Standard library only. The host broker owns the GPU lease and outer deadline.
Roles receive private mounts; no container runtime socket enters a role.
"""

import fcntl
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/oeis_learn/fresh"))
from common import canonical, confined, digest, file_hash, inventory, read, write

PROFILES = {"smoke", "fresh-pilot-1h", "fresh-benchmark", "fresh-extended-12h"}


def clock():
    return {
        "utc_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


class Ledger:
    """Append-only accounting; crash gaps are charged, clean pauses are excluded."""

    def __init__(self, root, identity, seconds, *, now=None, started=None):
        self.root, self.now, self.identity, self.cap = (
            Path(root),
            now or clock,
            identity,
            int(seconds * 1e9),
        )
        self.path = self.root / "budget.jsonl"
        self.previous = None
        if self.path.exists():
            for number, line in enumerate(self.path.read_bytes().splitlines(keepends=True)):
                if not line.endswith(b"\n"):
                    raise ValueError("torn budget record; review retained evidence")
                value = json.loads(line)
                if (
                    value["identity"] != identity
                    or value["cap_ns"] != self.cap
                    or value["sequence"] != number
                ):
                    raise ValueError("budget identity/cap/sequence changed")
                if self.previous is not None:
                    expected = self.previous["charged_ns"] + self.delta(
                        self.previous, value["clock"]
                    )
                    if value["charged_ns"] != expected:
                        raise ValueError("budget high-water mark changed")
                elif value["charged_ns"] != 0:
                    raise ValueError("invalid initial budget")
                self.previous = value
        if self.previous is None and started is not None:
            self.record("setup", observed=started)
        self.record("paused" if self.previous is None else self.previous["status"])

    @staticmethod
    def delta(previous, now):
        before = previous["clock"]
        if now["utc_ns"] < before["utc_ns"]:
            raise ValueError("backward UTC clock")
        same = before["boot_id"] == now["boot_id"]
        utc = now["utc_ns"] - before["utc_ns"]
        delta = now["monotonic_ns"] - before["monotonic_ns"] if same else utc
        if delta < 0 or (same and abs(utc - delta) > 5_000_000_000):
            raise ValueError("inconsistent budget clock")
        return delta if previous["status"] not in {"paused", "complete"} else 0

    def record(self, status, *, observed=None):
        now = self.now() if observed is None else observed
        previous = self.previous
        value = {
            "identity": self.identity,
            "cap_ns": self.cap,
            "status": status,
            "clock": now,
            "sequence": 0 if previous is None else previous["sequence"] + 1,
            "charged_ns": (
                0 if previous is None else previous["charged_ns"] + self.delta(previous, now)
            ),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.previous = value
        write(self.root / "budget.json", value)
        return value

    def remaining(self):
        return max(
            0,
            (self.cap - self.previous["charged_ns"] - self.delta(self.previous, self.now())) / 1e9,
        )


def snapshot_file(root, name, destination, ref):
    """Open every workspace component without following concurrently swapped links."""
    parts = Path(name).parts
    if Path(name).is_absolute() or not parts or ".." in parts:
        raise ValueError("confined snapshot path required")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for i, part in enumerate(parts):
            new = os.open(
                part,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | os.O_NONBLOCK
                | (os.O_DIRECTORY if i < len(parts) - 1 else 0),
                dir_fd=fd,
            )
            os.close(fd)
            fd = new
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != ref["bytes"]:
            raise ValueError("snapshot requires an unchanged regular file")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.dup(fd), "rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        if file_hash(destination) != ref["sha256"]:
            raise ValueError(f"snapshot source changed: {name}")
    finally:
        os.close(fd)


def podman_command(policy, image, name, role, inputs, output, command):
    worker = role == "worker"
    if role not in {"learner", "generation", "evaluator", "worker", "probe"}:
        raise ValueError("unknown role")
    args = [
        policy["podman"],
        "--runtime=crun",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--label",
        "lab.foundation=" + policy["project"],
        "--network=none",
        "--read-only",
        "--read-only-tmpfs=false",
        "--image-volume=ignore",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--userns=keep-id",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--group-add=keep-groups",
        "--memory=" + ("1g" if worker else "88g"),
        "--memory-swap=" + ("1g" if worker else "88g"),
        "--cpus=" + ("1" if worker else "4"),
        "--pids-limit=" + ("64" if worker else "512"),
        "--ipc=private",
        "--shm-size=1g",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
        "--workdir=/work",
        "--mount",
        f"type=bind,src={inputs},target=/work/input,ro",
        "--mount",
        f"type=bind,src={output},target=/work/output,rw",
        "--env",
        "PYTHONPATH=/opt/oeis-build/src",
        "--env",
        "OMP_NUM_THREADS=" + ("1" if worker else "4"),
    ]
    if role in {"learner", "generation", "probe"}:
        args += ["--device=/dev/kfd", "--device=" + policy["renderNode"]]
    return args + ["--entrypoint", command[0], image, *command[1:]]


class Coordinator:
    def __init__(self, root, policy, image, ledger):
        self.root, self.policy, self.image, self.ledger = Path(root), policy, image, ledger
        self.root.mkdir(parents=True, exist_ok=True)
        self.live = {}
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=4)

    def spawn(self, role, inputs, output, operation, *, worker=False):
        output.mkdir(parents=True, exist_ok=True)
        name = "lab-foundation-" + self.policy["project"] + "-" + uuid.uuid4().hex
        command = (
            ["python", "-m", "oeis_learn.tracking.role_worker", "serve"]
            if worker
            else ["python", "-m", "oeis_learn.fresh.worker", operation]
        )
        args = podman_command(self.policy, self.image, name, role, inputs, output, command)
        write(
            output / "launch.json",
            {
                "role": role,
                "image": self.image,
                "inputs": inventory(inputs),
                "operation": operation,
                "workspace_mounted": False,
            },
        )
        with (
            (output / "stdout.log").open("wb") as stdout,
            (output / "stderr.log").open("wb") as stderr,
        ):
            os.fchmod(stdout.fileno(), 0o644)
            os.fchmod(stderr.fileno(), 0o644)
            process = subprocess.Popen(args, stdout=stdout, stderr=stderr)
        self.live[name] = process
        return name, process

    def reclaim(self, name):
        process = self.live.get(name)
        if process is None:
            return
        try:
            subprocess.run(
                [self.policy["podman"], "rm", "--force", "--time=0", "--ignore", name],
                check=True,
                timeout=30,
                capture_output=True,
            )
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        # A failed removal stays tracked for another cleanup attempt and the
        # broker's independent fallback; killing the client alone is insufficient.
        self.live.pop(name, None)

    def start_workers(self):
        if self.workers:
            return
        deadline = time.monotonic() + min(30, self.ledger.remaining())
        for n in range(4):
            root = self.root / "workers" / str(n)
            inputs, output = root / "input", root / "output"
            inputs.mkdir(parents=True)
            name, process = self.spawn("worker", inputs, output, "serve", worker=True)
            self.workers.append(
                {"name": name, "process": process, "inputs": inputs, "output": output}
            )
        while not all((w["output"] / "ready.json").exists() for w in self.workers):
            if time.monotonic() >= deadline or any(
                w["process"].poll() is not None for w in self.workers
            ):
                raise RuntimeError("native worker startup failed; inspect worker stderr")
            time.sleep(0.01)

    def execute(self, slot, request, timeout):
        self.start_workers()
        worker = self.workers[slot]
        key = uuid.uuid4().hex + ".json"
        write(worker["inputs"] / key, request)
        deadline = time.monotonic() + min(2, timeout)
        started = time.monotonic_ns()
        path = worker["output"] / key
        while not path.exists():
            if time.monotonic() >= deadline or worker["process"].poll() is not None:
                self.reclaim(worker["name"])
                import hashlib

                lost = {
                    "source_sha256": "sha256:"
                    + hashlib.sha256(request["source"].encode()).hexdigest(),
                    "indices": request["indices"],
                    "outputs": [],
                    "reference_outputs": [],
                    "fuel": None,
                    "reference_steps": None,
                    "elapsed_ns": time.monotonic_ns() - started,
                    "outcome": "execution_limit",
                    "reason": "external_candidate_deadline",
                }
                # Replacement setup remains charged to the containing role.
                root = self.root / "workers" / (str(slot) + "-" + uuid.uuid4().hex)
                inputs, output = root / "input", root / "output"
                inputs.mkdir(parents=True)
                name, process = self.spawn("worker", inputs, output, "serve", worker=True)
                replacement = {"name": name, "process": process, "inputs": inputs, "output": output}
                self.workers[slot] = replacement
                setup_deadline = time.monotonic() + min(30, self.ledger.remaining() - 5)
                while not (output / "ready.json").exists():
                    if time.monotonic() >= setup_deadline or process.poll() is not None:
                        raise RuntimeError("replacement native worker startup failed")
                    time.sleep(0.01)
                return lost
            time.sleep(0.001)
        result = read(path)
        import hashlib

        if (
            result["source_sha256"]
            != "sha256:" + hashlib.sha256(request["source"].encode()).hexdigest()
            or result["indices"] != request["indices"]
        ):
            raise ValueError("native response source/indices mismatch")
        (worker["inputs"] / key).unlink()
        path.unlink()
        return result

    def run(self, operation, role, inputs, output, seconds):
        allowance = min(seconds, self.ledger.remaining() - 5)
        if allowance <= 0:
            raise TimeoutError("run lifetime budget exhausted")
        if role == "evaluator":
            self.start_workers()
        name, process = self.spawn(role, inputs, output, operation)
        deadline, heartbeat = time.monotonic() + allowance, time.monotonic()
        pending, issued, free = {}, set(), set(range(4))
        try:
            while process.poll() is None or pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{operation} role deadline exhausted; inspect {output}")
                if time.monotonic() >= heartbeat:
                    heartbeat = time.monotonic() + 10
                    self.ledger.record(operation)
                    print(
                        f"RUN {operation}; remaining={self.ledger.remaining():.0f}s; log={output / 'stdout.log'}",
                        flush=True,
                    )
                    write(
                        self.root.parent / "status.json",
                        {
                            "status": "running",
                            "phase": operation,
                            "budget": self.ledger.previous,
                            "progress": (
                                read(output / "progress.json")
                                if (output / "progress.json").exists()
                                else None
                            ),
                        },
                    )
                    if shutil.disk_usage(self.root).free < 100 << 30:
                        raise OSError("host free disk below 100 GiB")
                    used = 0
                    for base, _, files in os.walk(self.root.parent):
                        for filename in files:
                            used += (Path(base) / filename).stat().st_size
                            if used > 200 << 30:
                                raise OSError("run disk quota200GiB exhausted")
                    memory = dict(
                        line.split(":", 1)
                        for line in Path("/proc/meminfo").read_text().splitlines()
                    )
                    if int(memory["MemAvailable"].split()[0]) * 1024 < 16 << 30:
                        raise MemoryError("host available memory below 16 GiB")
                queue = output / "runtime-queue"
                for path in sorted(queue.glob("*.request.json")):
                    if path.name in issued or not free:
                        continue
                    if path.is_symlink() or path.stat().st_size > 1 << 20:
                        raise ValueError("invalid bounded execution request")
                    request = read(path)
                    if set(request) != {"source", "indices", "request_id", "limits"}:
                        raise ValueError("worker requests cannot carry paths or commands")
                    completed = path.with_name(path.name.replace(".request", ".response"))
                    if completed.exists():
                        if read(completed)["request_sha256"] != digest(request):
                            raise ValueError("completed native response binding changed")
                        issued.add(path.name)
                        continue
                    slot = free.pop()
                    issued.add(path.name)
                    pending[path.name] = (
                        slot,
                        self.executor.submit(
                            self.execute, slot, request, deadline - time.monotonic()
                        ),
                        digest(request),
                    )
                for key, (slot, future, binding) in list(pending.items()):
                    if future.done():
                        write(
                            queue / key.replace(".request", ".response"),
                            {"request_sha256": binding, "evidence": future.result()},
                        )
                        del pending[key]
                        free.add(slot)
                time.sleep(0.01)
            if process.returncode:
                raise RuntimeError(
                    f"{operation} exited {process.returncode}; inspect {output / 'stderr.log'}"
                )
            return read(output / "result.json")
        finally:
            self.reclaim(name)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        failures = []
        with ThreadPoolExecutor(max_workers=5) as reaper:
            futures = [reaper.submit(self.reclaim, name) for name in list(self.live)]
            for future in futures:
                try:
                    future.result()
                # Attempt every cgroup even when interrupted during cleanup.
                except BaseException as exc:  # noqa: BLE001
                    failures.append(str(exc))
        if failures:
            raise RuntimeError("role reclamation failed: " + "; ".join(failures))


def copy_view(destination, values, files=None):
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in values.items():
        write(confined(destination, name), value)
    for name, source in (files or {}).items():
        if Path(source).is_dir():
            for relative, ref in inventory(source).items():
                snapshot_file(
                    source, relative, confined(destination, str(Path(name) / relative)), ref
                )
        else:
            snapshot_file(
                Path(source).parent,
                Path(source).name,
                confined(destination, name),
                {"bytes": Path(source).stat().st_size, "sha256": file_hash(source)},
            )
    return destination


def checkpoint_files(job, name):
    path = confined(job, name)
    manifest = read(path)
    base_name = manifest["base"]
    base = read(confined(path.parent, base_name))
    return path, base


def process_identity(pid):
    return {
        "pid": pid,
        "boot_id": clock()["boot_id"],
        "process_start": Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19],
    }


def publish(job, destination):
    """Publish enumerated reports/state, never private role inputs or hidden truth."""
    for parent in (destination, *destination.parents):
        if parent.is_symlink():
            raise ValueError("symlink publication directory")
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "run.json",
        "status.json",
        "budget.json",
        "budget.jsonl",
        "finalization.json",
        "preparation-report.json",
        "input-manifest.json",
        "issuance.json",
        "early-stop.json",
    ):
        path = job / name
        if path.exists():
            write(confined(destination, name), path.read_bytes())
    for root in (job / "roles").glob("*"):
        if "-input" in root.name or root.name == "workers":
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink() or "runtime-queue" in path.parts:
                continue
            target = confined(destination, "roles/" + str(path.relative_to(job / "roles")))
            target.parent.mkdir(parents=True, exist_ok=True)
            # Checkpoints are large: stream to a private temporary and atomically replace.
            temporary = target.parent / (".publish-" + uuid.uuid4().hex)
            with path.open("rb") as source, temporary.open("xb") as output:
                os.fchmod(output.fileno(), 0o644)
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(target)
    # GPU-owned directories can inherit other::--- through the shared host ACL.
    # Publish read/traverse bits only within this enumerated public output tree.
    for base, dirs, _ in os.walk(destination):
        for directory in [Path(base), *(Path(base) / name for name in dirs)]:
            if directory.is_symlink():
                raise ValueError("symlink publication directory")
            info = directory.stat()
            if info.st_uid == os.geteuid():
                directory.chmod(info.st_mode | 0o075)


def evaluation(job, coordinator, ledger, settings, checkpoint, split, label):
    roles = job / "roles"
    result_path = roles / (label + "-score") / "result.json"
    if result_path.exists():
        saved = read(result_path)
        if saved["identity"] != digest({k: v for k, v in saved.items() if k != "identity"}):
            raise ValueError("cached evaluation result changed")
        if saved["checkpoint_sha256"] != checkpoint_files(job, checkpoint)[1]["blob_sha256"]:
            raise ValueError("evaluation checkpoint changed")
        return saved
    path, base = checkpoint_files(job, checkpoint)
    truth = read(job / "inputs/truth.json")[split]
    protocol = settings["protocol"]
    if len(truth) > protocol["groups"]:
        truth = truth[: protocol["groups"]]
    visible = [{"group_id": g["group_id"], "values": g["values"][:20]} for g in truth]
    seal_path = roles / (label + "-generate") / "result.json"
    generation_settings = dict(
        settings["learner"],
        checkpoint=path.name,
        attempts=protocol["attempts"],
        generation_tokens=protocol["tokens"],
        per_group_seconds=protocol["per_group_seconds"],
        protocol_sha256=digest(protocol),
    )
    if not seal_path.exists():
        inputs = roles / (label + "-generate-input")
        if not inputs.exists():
            copy_view(
                inputs,
                {"settings.json": generation_settings, "visible.json": visible},
                {"checkpoints": path.parent},
            )
        ledger.record("generate")
        coordinator.run(
            "generate", "generation", inputs, seal_path.parent, min(600, ledger.remaining() - 10)
        )
    seal = read(seal_path)
    if (
        seal["checkpoint_sha256"] != base["blob_sha256"]
        or seal["protocol_sha256"] != digest(protocol)
        or seal["visible_sha256"] != digest(visible)
    ):
        raise ValueError("sealed candidates do not bind this checkpoint/protocol/visible set")
    inputs = roles / (label + "-score-input")
    if not inputs.exists():
        values = {"seal.json": seal, "truth.json": truth, "settings.json": {"split": split}}
        if split == "final":
            values["finalization.json"] = read(job / "finalization.json")
        copy_view(inputs, values)
    ledger.record("score")
    return coordinator.run(
        "score", "evaluator", inputs, result_path.parent, min(600, ledger.remaining() - 10)
    )


def execute(policy, image, argv, *, coordinator_factory=Coordinator):
    action = argv[0]
    workspace = Path(policy["workspace"])
    if action == "check":
        root = Path(policy["home"]) / "foundation-checks" / uuid.uuid4().hex
        ledger = Ledger(root, digest([image, str(root)]), 300)
        ledger.record("probe")
        coordinator = coordinator_factory(root / "roles", policy, image, ledger)
        try:
            results = []
            for role, gpu in (("probe", True), ("worker", False)):
                inputs = copy_view(root / (role + "-input"), {"settings.json": {"gpu": gpu}})
                results.append(
                    coordinator.run("probe", role, inputs, root / (role + "-output"), 120)
                )
        finally:
            try:
                ledger.record("cleanup")
            finally:
                coordinator.close()
        ledger.record("complete")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "roles": results,
                    "image": image,
                    "scope": "mount/device/memory probes; full spec007 kill/quota acceptance remains pending",
                }
            )
        )
        return 0
    output_name = argv[2] if action == "run" else argv[1]
    destination = confined(workspace, output_name)
    if not (output_name.startswith(("runs/", "tmp-"))):
        raise ValueError("run outputs must be inside runs/ or a top-level tmp-* directory")
    job = Path(policy["home"]) / "foundation" / digest(output_name)[7:]
    if action == "status":
        print(json.dumps(read(job / "status.json")))
        return 0
    if action == "cancel":
        active = read(job / "active.json")
        pid = active["pid"]
        if process_identity(pid) != active:
            raise ValueError("stale process identity; administrator recovery required")
        os.kill(pid, signal.SIGTERM)
        return 0
    job.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (job / "job.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == "run":
            profile = argv[3]
            if profile not in PROFILES:
                raise ValueError("unknown fresh profile")
            if profile == "fresh-extended-12h" and policy.get("foundationTimeout", 0) < 44100:
                raise ValueError("extended profile requires deployed foundationTimeout >=44100")
            if (job / "run.json").exists():
                old = read(job / "run.json")
                requested = read(confined(workspace, argv[1]) / "prepared.json")
                if (
                    old["image"] != image
                    or old["inputs_identity"] != requested["identity"]
                    or old["settings"]["profile"] != profile
                ):
                    raise ValueError("existing run image/inputs/profile differ; choose a new run")
                if read(job / "status.json")["status"] == "complete":
                    publish(job, destination)
                    return 0
                raise ValueError("existing run requires resume; budget cannot be reissued")
            source_name = argv[1]
            source = confined(workspace, source_name)
            manifest = read(source / "prepared.json")
            if manifest["identity"] != digest(
                {k: v for k, v in manifest.items() if k != "identity"}
            ):
                raise ValueError("prepared identity mismatch")
            if (
                len(manifest["files"]) > 50000
                or sum(r["bytes"] for r in manifest["files"].values()) > 10 << 30
            ):
                raise ValueError("prepared input snapshot cap exceeded")
            if destination.exists() and any(destination.iterdir()):
                raise ValueError("new run destination must be empty")
            issuance_path = job / "issuance.json"
            issuance = {"image": image, "inputs_identity": manifest["identity"], "profile": profile}
            if issuance_path.exists():
                frozen = read(issuance_path)
                if {k: v for k, v in frozen.items() if k != "clock"} != issuance:
                    raise ValueError("initial issuance changed; create a new run")
            else:
                write(issuance_path, dict(issuance, clock=clock()), immutable=True)
            inputs = job / "inputs"
            if inputs.exists():
                shutil.rmtree(inputs)  # Only an unissued private preparation, never a run/budget.
            inputs.mkdir(mode=0o700)
            for name, ref in manifest["files"].items():
                snapshot_file(source, name, confined(inputs, name), ref)
            write(job / "input-manifest.json", manifest, immutable=True)
            examples = read(inputs / "examples.json")
            if len(examples["training"]) < 128 or not examples["development"]:
                raise ValueError("admitted training/development input shortfall")
            config = manifest["config"]["run"]
            if profile == "fresh-extended-12h" and (
                config.get("profile") != profile
                or len({digest(row["values"]) for row in examples["training"]}) < 2048
            ):
                raise ValueError("extended training requires its config and 2048 admitted prompts")
            if profile == "fresh-pilot-1h" and config["training_seconds"] > 1800:
                raise ValueError("long training configuration requires the extended profile")
            settings = {
                "profile": profile,
                "seconds": {
                    "smoke": 1800,
                    "fresh-pilot-1h": 3600,
                    "fresh-benchmark": 600,
                    "fresh-extended-12h": 43200,
                }[profile],
                "training_seconds": 120 if profile == "smoke" else config["training_seconds"],
                "protocol": dict(
                    config["evaluation"],
                    groups=2 if profile == "smoke" else config["evaluation"]["groups"],
                ),
                "learner": {
                    "model": config["model"],
                    "seed": config["seed"],
                    "device": "cuda",
                    "batch_size": 4 if profile == "smoke" else config.get("batch_size", 4),
                    "length_buckets": (
                        False if profile == "smoke" else config.get("length_buckets", False)
                    ),
                    "learning_rate": 0.0003,
                    "checkpoint_every_seconds": 60,
                    "checkpoint_reserve_seconds": 20,
                    "free_disk_floor_bytes": 100 << 30,
                    "examples_sha256": digest(examples),
                },
            }
            run = {
                "version": "fresh-run/v1",
                "image": image,
                "inputs_identity": manifest["identity"],
                "settings": settings,
                "qualified": False,
                "historical_compatibility": "pending",
                "spec007_release": "pending",
            }
            settings["learner"]["contract_sha256"] = digest(run)
            run["identity"] = digest(run)
            write(job / "run.json", run, immutable=True)
            write(job / "preparation-report.json", read(inputs / "archive-preparation-report.json"))
        else:
            run = read(job / "run.json")
            if run["image"] != image:
                raise ValueError("image changed; continuation needs the original runtime")
            settings = run["settings"]
        ledger = Ledger(
            job, run["identity"], settings["seconds"], started=read(job / "issuance.json")["clock"]
        )
        if ledger.remaining() < 60:
            write(
                job / "status.json",
                {
                    "status": "interrupted",
                    "error": "lifetime budget exhausted",
                    "budget": ledger.previous,
                    "qualified": False,
                },
            )
            publish(job, destination)
            raise TimeoutError("less than60s remains; no budget reset permitted")
        if (job / "finalization.json").exists() and action == "resume":
            raise ValueError("finalized run cannot resume training")
        ledger.record("recovery" if action != "run" else "setup")
        coordinator = coordinator_factory(job / "roles", policy, image, ledger)
        active = process_identity(os.getpid())
        write(job / "active.json", active)
        try:
            roles = job / "roles"
            if settings["profile"] == "fresh-benchmark":
                benchmark_output = roles / "benchmark"
                if not (benchmark_output / "result.json").exists():
                    inputs = roles / "benchmark-input"
                    if not inputs.exists():
                        copy_view(
                            inputs,
                            {
                                "examples.json": read(job / "inputs/examples.json"),
                                "settings.json": settings["learner"],
                            },
                        )
                    coordinator.run(
                        "benchmark",
                        "learner",
                        inputs,
                        benchmark_output,
                        min(500, ledger.remaining() - 10),
                    )
                ledger.record("benchmark-finished")
                write(
                    job / "status.json",
                    {
                        "status": "cleanup",
                        "pending_status": "complete",
                        "benchmark": read(benchmark_output / "result.json"),
                        "budget": ledger.previous,
                        "qualified": False,
                    },
                )
                return 0
            stages = sorted(roles.glob("train-[0-9][0-9][0-9][0-9][0-9][0-9]/result.json"))
            latest = None
            if stages:
                last = stages[-1]
                latest = str(
                    (last.parent / "checkpoints" / read(last)["checkpoint"]).relative_to(job)
                )
            if action == "evaluate":
                split, selection = argv[2:]
                if split == "final":
                    latest = read(job / "finalization.json")["checkpoint"]
                elif selection == "latest":
                    if latest is None:
                        raise ValueError("no complete checkpoint")
                else:
                    raise ValueError("development evaluation supports latest checkpoint")
                label = (
                    "final"
                    if split == "final"
                    else split + "-" + checkpoint_files(job, latest)[1]["blob_sha256"][7:23]
                )
                evaluation(job, coordinator, ledger, settings, latest, split, label)
                ledger.record("evaluation-finished")
                write(
                    job / "status.json",
                    {
                        "status": "cleanup",
                        "pending_status": "complete" if split == "final" else "paused",
                        "checkpoint": latest,
                        "budget": ledger.previous,
                    },
                )
                return 0
            if action == "finalize":
                finalize(job, settings, argv[2])
                ledger.record("finalization-finished")
                write(
                    job / "status.json",
                    {
                        "status": "cleanup",
                        "pending_status": "finalized",
                        "checkpoint": read(job / "finalization.json")["checkpoint"],
                        "budget": ledger.previous,
                    },
                )
                return 0
            if action not in {"run", "resume"}:
                raise ValueError("unknown fresh action")
            # Initialize once and evaluate random weights before any updates.
            if latest is None:
                stage = roles / "train-000000"
                retained = [
                    p / "checkpoints"
                    for p in sorted(
                        roles.glob("train-000000-attempt-*"), key=lambda p: p.stat().st_mtime_ns
                    )
                    if list((p / "checkpoints").glob("qualified-[0-9]*.json"))
                ]
                inputs = roles / ("train-000000-input-" + uuid.uuid4().hex)
                copy_view(
                    inputs,
                    {
                        "examples.json": read(job / "inputs/examples.json"),
                        "settings.json": dict(
                            settings["learner"],
                            seconds=90,
                            update_cap=0,
                            charged_budget_ns=ledger.previous["charged_ns"],
                            ledger_sequence=ledger.previous["sequence"],
                        ),
                    },
                )
                if retained:
                    for name, ref in inventory(retained[-1]).items():
                        snapshot_file(
                            retained[-1], name, confined(inputs, "checkpoints/" + name), ref
                        )
                attempt = roles / ("train-000000-attempt-" + uuid.uuid4().hex)
                result = coordinator.run("train", "learner", inputs, attempt, 120)
                stage.mkdir(parents=True, exist_ok=True)
                shutil.copytree(attempt / "checkpoints", stage / "checkpoints", dirs_exist_ok=True)
                write(stage / "result.json", result)
                latest = str((stage / "checkpoints" / result["checkpoint"]).relative_to(job))
            initial_result = read(roles / "train-000000/result.json")
            initial_checkpoint = str(
                (roles / "train-000000/checkpoints" / initial_result["checkpoint"]).relative_to(job)
            )
            evaluation(
                job, coordinator, ledger, settings, initial_checkpoint, "development", "initial"
            )
            # Three smoke updates include a real boundary after update one.
            extended = settings["profile"] == "fresh-extended-12h"
            boundaries = (
                (1, 3)
                if settings["profile"] == "smoke"
                else ((None,) * 10 if extended else (None, None))
            )
            best_validation, stale = float("inf"), 0
            for number, cap in enumerate(boundaries, 1):
                stage = roles / f"train-{number:06d}"
                if (stage / "result.json").exists():
                    result = read(stage / "result.json")
                    latest = str((stage / "checkpoints" / result["checkpoint"]).relative_to(job))
                else:
                    # Failed role checkpoints are recoverable; never replace a committed generation.
                    attempts = sorted(
                        roles.glob(stage.name + "-attempt-*"), key=lambda p: p.stat().st_mtime_ns
                    )
                    retained = [
                        p / "checkpoints"
                        for p in attempts
                        if list((p / "checkpoints").glob("qualified-[0-9]*.json"))
                    ]
                    recovery = retained[-1] if retained else checkpoint_files(job, latest)[0].parent
                    seconds = min(
                        settings["training_seconds"] / len(boundaries),
                        ledger.remaining() - (1800 if extended else 900),
                    )
                    if settings["profile"] == "smoke":
                        seconds = min(60, ledger.remaining() - 300)
                    if seconds < 25:
                        raise TimeoutError(
                            "insufficient remaining budget for training and evaluation"
                        )
                    source = roles / (stage.name + "-input-" + uuid.uuid4().hex)
                    copy_view(
                        source,
                        {
                            "examples.json": read(job / "inputs/examples.json"),
                            "settings.json": dict(
                                settings["learner"],
                                seconds=seconds,
                                update_cap=cap or 1000000,
                                charged_budget_ns=ledger.previous["charged_ns"],
                                ledger_sequence=ledger.previous["sequence"],
                            ),
                        },
                        {"checkpoints": recovery},
                    )
                    # Every attempt has separate immutable output; stage pointers select successful attempts.
                    attempt = roles / (stage.name + "-attempt-" + uuid.uuid4().hex)
                    ledger.record("training")
                    result = coordinator.run("train", "learner", source, attempt, seconds + 30)
                    stage.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(
                        attempt / "checkpoints", stage / "checkpoints", dirs_exist_ok=True
                    )
                    write(stage / "result.json", result)
                    latest = str((stage / "checkpoints" / result["checkpoint"]).relative_to(job))
                evaluation(
                    job,
                    coordinator,
                    ledger,
                    settings,
                    latest,
                    "development",
                    f"development-{number}",
                )
                publish(job, destination)
                if extended:
                    loss = result.get("validation_loss")
                    if loss is None or not __import__("math").isfinite(loss):
                        raise ValueError("extended run needs finite structural validation loss")
                    if loss < best_validation - 0.01:
                        best_validation, stale = loss, 0
                    else:
                        stale += 1
                    if stale >= 2:
                        write(
                            job / "early-stop.json",
                            {
                                "reason": "two segments without validation improvement",
                                "best_validation_loss": best_validation,
                                "completed_updates": result["completed_updates"],
                            },
                        )
                        break
            if settings["profile"] == "smoke" and result["completed_updates"] != 3:
                raise ValueError("three actual smoke updates required")
            finalize(job, settings, "best-development")
            selected = read(job / "finalization.json")["checkpoint"]
            final = evaluation(job, coordinator, ledger, settings, selected, "final", "final")
            ledger.record("final-scored")
            write(
                job / "status.json",
                {
                    "status": "cleanup",
                    "pending_status": "complete",
                    "checkpoint": selected,
                    "completed_updates": result["completed_updates"],
                    "final_score": final,
                    "budget": ledger.previous,
                    "qualified": False,
                    "diagnostic_final_cohort_consumed": True,
                },
            )
            return 0
        except BaseException as exc:
            # Remain active: time until an interrupted run is explicitly recovered is charged.
            ledger.record("interrupted")
            write(
                job / "status.json",
                {
                    "status": "interrupted",
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": ledger.previous,
                    "qualified": False,
                },
            )
            raise
        finally:
            cleanup_error = None
            try:
                try:
                    ledger.record("cleanup")
                finally:
                    coordinator.close()
            except BaseException as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                status = read(job / "status.json")
                pending_status = status.pop("pending_status", None)
                if cleanup_error is not None:
                    status.update(status="interrupted", cleanup_error=cleanup_error)
                    status.setdefault("error", cleanup_error)
                elif pending_status is not None:
                    status["status"] = pending_status
                terminal = (
                    "complete"
                    if status["status"] == "complete"
                    else "paused" if status["status"] in {"paused", "finalized"} else "interrupted"
                )
                ledger.record(terminal)
                status["budget"] = ledger.previous
                if ledger.previous["charged_ns"] > ledger.cap:
                    status.update(
                        status="interrupted", error="cleanup exceeded frozen lifetime cap"
                    )
                    ledger.record("interrupted")
                    status["budget"] = ledger.previous
                write(job / "status.json", status)
                job.joinpath("active.json").unlink(missing_ok=True)
                publish(job, destination)
                if cleanup_error is None and ledger.previous["charged_ns"] > ledger.cap:
                    raise TimeoutError("cleanup exceeded frozen lifetime cap")


def finalize(job, settings, selection):
    if (job / "finalization.json").exists():
        return read(job / "finalization.json")
    if selection != "best-development":
        raise ValueError("predeclared best-development selection required")
    candidates = []
    for path in sorted((job / "roles").glob("development-*-score/result.json")):
        score = read(path)
        for stage in sorted(
            (job / "roles").glob("train-[0-9][0-9][0-9][0-9][0-9][0-9]/result.json")
        ):
            result = read(stage)
            if result["checkpoint_sha256"] == score["checkpoint_sha256"]:
                candidates.append(
                    (
                        len(score["solved_groups"]),
                        len(score["top_one_groups"]),
                        (
                            -result.get("validation_loss", float("inf"))
                            if settings["profile"] == "fresh-extended-12h"
                            else 0
                        ),
                        -result["completed_updates"],
                        str((stage.parent / "checkpoints" / result["checkpoint"]).relative_to(job)),
                    )
                )
    if not candidates:
        raise ValueError("no complete development evaluation for checkpoint selection")
    checkpoint = max(candidates)[-1]
    final = {
        "version": "fresh-finalization/v1",
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_files(job, checkpoint)[1]["blob_sha256"],
        "protocol_sha256": digest(settings["protocol"]),
        "truth_sha256": digest(
            read(job / "inputs/truth.json")["final"][: settings["protocol"]["groups"]]
        ),
        "criterion": "solved groups, top-one groups, then "
        + (
            "structural validation loss, then earliest update"
            if settings["profile"] == "fresh-extended-12h"
            else "earliest update"
        ),
        "selected_using": "development",
        "diagnostic": True,
        "final_cohort_consumed_on_evaluation": True,
    }
    write(job / "finalization.json", final, immutable=True)
    return final


def main():
    policy, image, argv = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3:]
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    return execute(policy, image, argv)


if __name__ == "__main__":
    raise SystemExit(main())
