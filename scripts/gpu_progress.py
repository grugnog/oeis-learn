#!/usr/bin/env python3
"""Temporary resumable VM driver. See gpu_progress.md; no runtime installation."""

from __future__ import annotations
import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
HOST_STEPS = {
    "host-containment",
    "baseline",
    "overflow-gpu",
    "measurement",
    "release-handoff",
    "end-to-end",
}
ISOLATED_ROLES_BLOCKER = (
    "lab-gpu supports image ensure and one workspace-mounted exec container only; "
    "it exposes no Docker/Podman API, host shell, or isolated role launch operation. "
    "This stage needs the owning host-side role launcher with separate private mounts. "
    "Host launcher integration is required; installing Docker in the VM/GPU image or "
    "using the lab-gpu SSH identity as a Docker endpoint cannot provide it."
)


class Blocked(RuntimeError):
    pass


@dataclass(frozen=True)
class Step:
    name: str
    needs: tuple[str, ...]
    seconds: int
    gpu: bool = False
    inputs: tuple[str, ...] = ()


STEPS = (
    Step("inventory", (), 60),
    Step("image", ("inventory",), 3600),
    Step("gpu-environment", ("image",), 120, True),
    Step("cpu-regressions", ("inventory",), 3600),
    Step("gpu-smoke", ("gpu-environment", "cpu-regressions"), 1800, True),
    Step("gpu-learning", ("gpu-smoke",), 7200, True),
    Step("preflight", ("gpu-smoke",), 600, True, ("image_id",)),
    Step("legacy-cpu", ("cpu-regressions",), 3600, inputs=("legacy_cases",)),
    Step("legacy-gpu", ("legacy-cpu", "gpu-smoke"), 3600, True, ("legacy_cases",)),
    Step("phase13-inputs", ("cpu-regressions",), 3600),
    Step(
        "release-source",
        ("cpu-regressions",),
        3600,
        inputs=("source_manifest", "cohort_config", "archive", "holdouts"),
    ),
    Step("host-containment", ("gpu-environment",), 1800, True, ("image_id",)),
    Step("baseline", ("preflight", "host-containment", "release-source"), 7200, True),
    Step(
        "overflow-gpu", ("phase13-inputs", "release-source", "gpu-learning", "baseline"), 3600, True
    ),
    Step("measurement", ("baseline", "gpu-learning", "legacy-gpu", "overflow-gpu"), 28800, True),
    Step("release-handoff", ("measurement",), 7200, True),
    Step("end-to-end", ("release-handoff",), 43200, True),
)
TASK_CHECKS = {}
for tasks, checks in (
    (
        "T005 T007 T023 T024 T040 T042 T043 T045 T046 T047 T049 T115",
        "cpu-regressions gpu-smoke host-containment preflight baseline",
    ),
    ("T062 T066 T067 T078 T079 T116", "cpu-regressions gpu-learning"),
    ("T093 T113", "cpu-regressions release-source"),
    ("T095 T114 T117", "cpu-regressions phase13-inputs legacy-cpu legacy-gpu"),
    ("T096 T097 T118", "cpu-regressions phase13-inputs overflow-gpu measurement"),
    ("T098 T099 T100 T119 T120", "cpu-regressions measurement"),
    ("T102 T104 T105 T121", "cpu-regressions release-handoff end-to-end"),
):
    TASK_CHECKS.update({t: checks.split() for t in tasks.split()})
FIELDS = {
    "image_id",
    "docker_host",
    "preflight_config",
    "source_manifest",
    "cohort_config",
    "archive",
    "holdouts",
    "legacy_cases",
    "final_seconds",
}
STEP_CONFIG = {
    "phase13-inputs": ("archive", "holdouts"),
    "gpu-environment": ("image_id",),
    "host-containment": (),
    "preflight": ("preflight_config",),
    "baseline": ("image_id",),
    "overflow-gpu": ("image_id",),
    "measurement": ("image_id",),
    "release-handoff": ("image_id",),
    "end-to-end": ("image_id", "final_seconds"),
}
PROBES = {"inventory", "image", "gpu-environment"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def tree_hashes(path, *, source=False):
    path = Path(path)
    if path.is_file():
        return {path.name: file_hash(path)}
    files = []
    for directory, children, names in os.walk(path):
        children[:] = [
            name
            for name in children
            if name != "__pycache__"
            and not (
                source
                and (
                    name in {"target", "build", "dist", ".pytest_cache"}
                    or name.endswith(".egg-info")
                )
            )
        ]
        files.extend(Path(directory) / name for name in names if not name.endswith(".pyc"))
    return {str(p.relative_to(path)): file_hash(p) for p in sorted(files) if p.is_file()}


def task_inventory(root=ROOT):
    rows = []
    for status, task, description in re.findall(
        r"^- \[(/| )\] (T\d+) (.+)$",
        (root / "specs/007-experiment-foundation/tasks.md").read_text(),
        re.M,
    ):
        if int(task[1:]) < 115:
            if task not in TASK_CHECKS:
                raise ValueError(f"Unmapped earlier unfinished task {task}; review tasks.md")
            rows.append(
                dict(
                    task=task,
                    status=status,
                    description=description,
                    checks=TASK_CHECKS[task],
                    acceptance="pending_actual_evidence",
                )
            )
    return rows


def input_hashes(path, seen=None):
    seen = set() if seen is None else seen
    path = Path(path).resolve()
    if path in seen or not path.exists():
        return {}
    seen.add(path)
    result = {str(path): tree_hashes(path)}
    if path.is_file() and path.suffix in (".json", ".yaml", ".yml"):
        try:
            if path.suffix == ".json":
                value = read(path)
            else:
                import yaml

                value = yaml.safe_load(path.read_text())
        except (ValueError, ImportError):
            return result

        def visit(v):
            if isinstance(v, dict):
                for k, c in v.items():
                    if k in {
                        "path",
                        "checkpoint",
                        "config",
                        "dockerfile",
                        "dependency_lock",
                    } and isinstance(c, str):
                        result.update(input_hashes(path.parent / c, seen))
                    elif isinstance(c, (dict, list)):
                        visit(c)
            elif isinstance(v, list):
                for c in v:
                    visit(c)

        visit(value)
    return result


def load_settings(path=None):
    cfg = read(path) if path else {}
    if not isinstance(cfg, dict) or set(cfg) - FIELDS:
        raise ValueError("Unknown settings fields")
    cfg = dict(cfg)
    cfg.setdefault("final_seconds", 3600)
    if type(cfg["final_seconds"]) is not int or not 600 <= cfg["final_seconds"] <= 43200:
        raise ValueError("final_seconds must be an integer in 600..43200")
    if cfg.get("image_id") and not re.fullmatch(r"sha256:[0-9a-f]{64}", cfg["image_id"]):
        raise ValueError("Actual sha256 image ID required")
    # Accept null from old example files, but never send Docker requests through
    # the forced-command SSH broker or seek an endpoint from inside its sandbox.
    if cfg.get("docker_host") is not None:
        raise ValueError(
            "docker_host is unsupported by the lab-gpu broker. " + ISOLATED_ROLES_BLOCKER
        )
    cfg.pop("docker_host", None)
    for name in FIELDS - {"image_id", "docker_host", "final_seconds"}:
        if cfg.get(name):
            cfg[name] = str((ROOT / cfg[name]).resolve())
    return cfg


def source_identity(root=ROOT):
    names = (
        "src",
        "scripts",
        "tests",
        "configs/foundation",
        "docker/dev",
        "crates/oeis_wasm_evaluator",
        "specs/007-experiment-foundation",
        ".specify/memory",
        "Dockerfile",
        "gpu-project.json",
        "setup.py",
        "MANIFEST.in",
        "pyproject.toml",
        "uv.lock",
        "data/sample_oeis",
    )
    return digest({n: tree_hashes(root / n, source=True) for n in names})


def step_identity(step, cfg, source, receipts):
    keys = set(step.inputs) | set(STEP_CONFIG.get(step.name, ()))
    values = {k: cfg.get(k) for k in sorted(keys)}
    refs = {
        k: input_hashes(v)
        for k, v in values.items()
        if k not in {"image_id", "docker_host", "final_seconds"} and v
    }
    return digest(
        dict(
            source=source,
            settings=values,
            refs=refs,
            dependencies={n: receipts[n]["artifact_identity"] for n in step.needs},
        )
    )


def cached_success(receipt, identity, out):
    return bool(
        receipt
        and receipt.get("status") == "PASS"
        and receipt.get("identity") == identity
        and receipt.get("files") == tree_hashes(out)
    )


def success_receipt(identity, out, result, elapsed, step):
    files = tree_hashes(out)
    semantic = result.get("runtime", result) if step.name in PROBES else files
    return dict(
        status="PASS",
        identity=identity,
        artifact_identity=digest(semantic),
        files=files,
        issued=True,
        elapsed_seconds=elapsed,
    )


def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise Blocked(f"Active worker/controller lock: {path}")
    return f


def require_idle(out):
    out = Path(out)
    if not out.exists():
        return
    f = lock(out / "worker.lock")
    f.close()
    p = out / "reclamation.json"
    if p.exists() and read(p)["status"] != "complete":
        raise Blocked("Earlier lease cleanup failed; host administrator recovery required")


def final_remaining(state, limit, now=None):
    now = time.time() if now is None else now
    path = state / "final-budget.json"
    if path.exists():
        record = read(path)
        if record["limit_seconds"] != limit:
            raise Blocked("Final budget cannot change after issuance")
        if now < record["last_observed_utc"]:
            raise Blocked("Final wall clock regressed")
    else:
        record = dict(started_utc=now, limit_seconds=limit)
    record["last_observed_utc"] = now
    write(path, record)
    remaining = record["started_utc"] + limit - now
    if remaining <= 40:
        raise Blocked("Final lifetime budget exhausted; offline/retry time remains charged")
    return remaining


def supervise(argv, log, seconds):
    start = time.monotonic()
    process = None
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("ab") as stream:
        stream.write((json.dumps([str(a) for a in argv]) + "\n").encode())
        stream.flush()
        try:
            process = subprocess.Popen(
                [str(a) for a in argv],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            next_message = start + 30
            while process.poll() is None:
                now = time.monotonic()
                if now - start >= seconds:
                    return 124
                if now >= next_message:
                    print(f"  running {now-start:.0f}s; log: {log}", flush=True)
                    next_message = now + 30
                time.sleep(0.2)
            return process.returncode
        except KeyboardInterrupt:
            return 130
        finally:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)


def worker_argv(step, settings, state, seconds):
    args = [
        SCRIPT,
        "_watchdog" if step.gpu else "_worker",
        step.name,
        settings,
        state,
        str(seconds),
    ]
    return ["lab-gpu", "exec", "--", "python", *args] if step.gpu else [sys.executable, *args]


def remote_watchdog(name, settings, state, seconds):
    if name in HOST_STEPS:
        raise Blocked(ISOLATED_ROLES_BLOCKER)
    state = Path(state)
    out = state / name
    out.mkdir(parents=True, exist_ok=True)
    held = lock(out / "worker.lock")
    lease = uuid.uuid4().hex
    write(
        out / "lease.json",
        dict(
            lease=lease,
            backend="lab-gpu",
            cleanup_owner="host broker",
            pid=os.getpid(),
            issued_utc=time.time(),
            seconds=seconds,
        ),
    )
    if name == "end-to-end":
        budget = read(state / "final-budget.json")
        seconds = min(seconds, budget["started_utc"] + budget["limit_seconds"] - time.time())
    try:
        code = supervise(
            [sys.executable, SCRIPT, "_worker", name, settings, state, max(1, seconds - 40)],
            out / "worker.log",
            max(1, seconds - 40),
        )
    finally:
        # The host broker removes its rootless container and restores inference.
        # The project process has no container API or authority to clear a stale
        # host lease. supervise() handles only this worker's process group.
        held.close()
    return code


def check_junit(path, allowed_skips=()):
    cases = ET.parse(path).findall(".//testcase")
    if not cases:
        raise ValueError(f"Empty JUnit: {path}")
    for case in cases:
        if case.find("failure") is not None or case.find("error") is not None:
            raise ValueError(f"Failed test {case.attrib} in {path}")
        if case.find("skipped") is not None and case.attrib.get("name") not in allowed_skips:
            raise ValueError(f"Unexpected skipped test {case.attrib} in {path}")
    return len(cases)


def execute_worker(name, settings, state, seconds):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.modules.setdefault("gpu_progress", sys.modules[__name__])
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    from gpu_progress_steps import execute

    out = Path(state) / name
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(name, read(settings), Path(state), out, float(seconds))
        write(out / "step-result.json", dict(status="PASS", result=result))
        return 0
    except Blocked as e:
        write(out / "blocked.json", dict(error=str(e)))
        print(f"BLOCKED: {e}", file=sys.stderr, flush=True)
        return 3


def run(cfg, state, until=None):
    state.mkdir(parents=True, exist_ok=True)
    held = lock(state / "controller.lock")
    try:
        return _run(cfg, state, until)
    finally:
        held.close()


def _run(cfg, state, until):
    cfg = dict(cfg)
    task_inventory()
    source = source_identity()
    settings = state / ("settings-" + digest(cfg) + ".json")
    write(settings, cfg)
    path = state / "receipts.json"
    receipts = read(path) if path.exists() else {}
    statuses = {}
    errors = {}
    all_pass = True
    for step in STEPS:
        out = state / step.name
        previous = receipts.get(step.name)
        identity = None
        start = time.monotonic()
        issued = False
        code = None
        try:
            if step.name in HOST_STEPS:
                raise Blocked(ISOLATED_ROLES_BLOCKER)
            missing = [n for n in step.needs if statuses.get(n) not in {"PASS", "SKIP"}]
            if missing:
                raise Blocked("Prerequisites not successful: " + ", ".join(missing))
            missing = [n for n in step.inputs if not cfg.get(n)]
            if missing:
                raise Blocked("Missing settings: " + ", ".join(missing))
            identity = step_identity(step, cfg, source, receipts)
            require_idle(out)
            if step.name not in PROBES and cached_success(previous, identity, out):
                statuses[step.name] = "SKIP"
                print(f"SKIP {step.name}: verified prior success", flush=True)
            else:
                if step.name == "measurement" and (
                    (previous and previous.get("issued")) or any(out.rglob("measurement.json"))
                ):
                    raise Blocked(
                        "Measurement was already issued; preserve its immutable disposition for review"
                    )
                if (
                    step.name == "end-to-end"
                    and previous
                    and previous.get("issued")
                    and previous.get("identity") != identity
                ):
                    raise Blocked("Cannot change final inputs after issuance")
                seconds = (
                    final_remaining(state, cfg["final_seconds"])
                    if step.name == "end-to-end"
                    else step.seconds
                )
                if out.exists() and step.name != "end-to-end":
                    history = state / "history"
                    history.mkdir(exist_ok=True)
                    out.rename(history / (step.name + "-" + str(time.time_ns())))
                out.mkdir(parents=True, exist_ok=True)
                (out / "worker.lock").touch(exist_ok=True)
                receipts[step.name] = dict(status="RUNNING", identity=identity, issued=True)
                write(path, receipts)
                issued = True
                print(
                    f'RUN {step.name}: cap {seconds:.0f}s; log {out / "transport.log"}', flush=True
                )
                launched = time.time()
                code = supervise(
                    worker_argv(step, settings, state, seconds),
                    out / "transport.log",
                    # Reserve the local 20s interrupt + 5s kill/reap grace inside
                    # the final lifetime limit, rather than adding it afterward.
                    (
                        max(1, seconds - 25)
                        if step.name == "end-to-end"
                        else seconds + (60 if step.gpu else 0)
                    ),
                )
                finished = time.time()
                if code == 75 and not (out / "lease.json").exists():
                    issued = False
                if code == 0:
                    result = read(out / "step-result.json")["result"]
                    if step.name == "image":
                        actual = result["runtime"]["image_id"]
                        if cfg.get("image_id") and cfg["image_id"] != actual:
                            raise Blocked("Configured image_id differs from lab-gpu image ensure")
                        cfg["image_id"] = actual
                        settings = state / ("settings-" + digest(cfg) + ".json")
                        write(settings, cfg)
                    if (
                        step.name == "gpu-environment"
                        and not launched - 5 <= result["observed_utc"] <= finished + 5
                    ):
                        raise Blocked(
                            "VM and broker wall clocks disagree; final budget cannot be trusted"
                        )
                    receipts[step.name] = success_receipt(
                        identity, out, result, time.monotonic() - start, step
                    )
                    statuses[step.name] = "PASS"
                    print(f"PASS {step.name}", flush=True)
                elif code in (3, 75):
                    raise Blocked(f'Worker blocked (exit {code}); inspect {out / "transport.log"}')
                else:
                    raise RuntimeError(f'Worker exit {code}; inspect {out / "transport.log"}')
        except Exception as e:
            status = (
                "BLOCKED" if isinstance(e, Blocked) else ("INTERRUPTED" if code == 130 else "FAIL")
            )
            statuses[step.name] = status
            errors[step.name] = str(e)
            all_pass = False
            # Keep a valid success when a dependency is temporarily unavailable.
            # Never rebind an issued final run to changed inputs on a blocked retry.
            if not (previous and previous.get("status") == "PASS" and not issued) and not (
                step.name == "end-to-end" and previous and previous.get("issued") and not issued
            ):
                receipts[step.name] = dict(
                    status=status,
                    identity=identity,
                    issued=issued or bool(previous and previous.get("issued")),
                    error=str(e),
                )
                if code == 75 and not issued:
                    receipts[step.name]["issued"] = False
            print(f"{status} {step.name}: {e}", flush=True)
        write(path, receipts)
        write(
            state / "summary.json",
            dict(statuses=statuses, errors=errors, updated_utc=time.time(), qualified=False),
        )
        if code == 130:
            return 130
        if until == step.name:
            break
    return 0 if all_pass else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"_worker", "_watchdog"}:
        kind, name, settings, state, seconds = sys.argv[1:]
        try:
            if kind == "_worker":
                return execute_worker(name, settings, state, seconds)
            if kind == "_watchdog":
                return remote_watchdog(name, settings, state, float(seconds))
        except Blocked as e:
            print(str(e), file=sys.stderr)
            return 3
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--state-dir", default="tmp-gpu-progress")
    parser.add_argument("--until", choices=[s.name for s in STEPS])
    parser.add_argument("--reclaim", choices=[s.name for s in STEPS])
    args = parser.parse_args()
    if args.plan:
        for s in STEPS:
            runner = (
                "BLOCKED: isolated host roles unavailable"
                if s.name in HOST_STEPS
                else ("lab-gpu" if s.gpu else "VM")
            )
            print(f'{s.name}: {runner}; cap {s.seconds}s; needs {",".join(s.needs) or "-"}')
        print(json.dumps(task_inventory(), indent=2))
        return 0
    state = (ROOT / args.state_dir).resolve()
    if state.parent != ROOT or not state.name.startswith("tmp-"):
        parser.error("State must be a top-level tmp-* directory inside the checkout")
    if args.reclaim:
        parser.error(
            "Broker cleanup belongs to the host administrator. Follow grugnog/lab "
            "docs/lab/gpu-runner.md Lifecycle and recovery; do not clear a live host lease. "
            "Old Docker role leases require review at their originally recorded host endpoint."
        )
    try:
        return run(load_settings(args.config), state, args.until)
    except (ValueError, Blocked) as e:
        parser.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
