"""Independent heartbeat/deadline controller for one bounded learner.

The outer process is a watchdog for the controller itself. This is a process
deadline boundary, not a claim of container filesystem or device isolation.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import signal
import shutil
import time

from oeis_learn.experiments.artifacts import canonical_bytes, load_json
from oeis_learn.tracking.budget_ledger import BudgetLedger, ACTIVE, clock, elapsed
from oeis_learn.tracking.foundation_metrics import EventLog, replace_json, resource_metrics
from oeis_learn.tracking.training_checkpoint import tree_bytes


@contextmanager
def run_lock(root):
    with (Path(root) / "controller.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("run already has an active controller") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _send(connection, obj):
    raw = canonical_bytes(obj)
    if len(raw) > 1 << 20:
        raise ValueError("controller message exceeds 1 MiB")
    connection.send_bytes(raw)


def _receive(connection):
    return load_json(connection.recv_bytes(1 << 20))


def _learner(connection, root, resume, stop_after):
    from oeis_learn.rl.foundation_sft import FoundationTrainer

    try:
        learner = FoundationTrainer(root, resume=resume)

        def phase(name):
            _send(connection, dict(kind="phase", phase=name))
            return _receive(connection)

        if not resume:
            learner.save(phase("checkpoint"))  # random state is recoverable before first update
        path = None
        while learner.completed_update < learner.cfg.data["training"]["updates"]:
            phase("training")
            result = learner.update()
            _send(connection, result)
            _receive(connection)
            complete = learner.completed_update == learner.cfg.data["training"]["updates"]
            path = learner.save(phase("checkpoint"), final=complete)
            _send(
                connection,
                dict(kind="checkpoint", path=str(path), completed_update=learner.completed_update),
            )
            _receive(connection)
            if stop_after is not None and learner.completed_update >= stop_after:
                break
        _send(
            connection,
            dict(
                kind="done",
                completed_update=learner.completed_update,
                checkpoint=str(path) if path else None,
            ),
        )
    except BaseException as exc:
        _send(connection, dict(kind="error", error=f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _reap(process):
    if process.is_alive():
        process.terminate()
        process.join(0.5)
    if process.is_alive():
        process.kill()
        process.join(1.5)
    if process.is_alive():
        raise RuntimeError("learner reclamation exceeded two seconds")


class ResourceGuard:
    def __init__(
        self,
        root,
        *,
        free_floor,
        disk_usage=shutil.disk_usage,
        quota=200 << 30,
        host_floor=16 << 30,
    ):
        self.root, self.free_floor, self.disk_usage, self.quota = (
            Path(root),
            free_floor,
            disk_usage,
            quota,
        )
        self.low_memory = 0
        self.host_floor = host_floor

    def check(self):
        if tree_bytes(self.root) >= self.quota or self.disk_usage(self.root).free < self.free_floor:
            raise OSError("run disk quota/free-space floor exhausted")
        resources = resource_metrics("cpu")
        memory = resources["host_available"]
        self.low_memory = (
            self.low_memory + 1
            if memory["state"] == "measured" and memory["value"] < self.host_floor
            else 0
        )
        if self.low_memory >= 2:
            raise OSError("host available memory below frozen floor for two samples")
        return resources


def _controller(root, resume, stop_after, child_target=_learner):
    os.setsid()
    root = Path(root)
    run = load_json((root / "run.json").read_bytes())
    ledger = BudgetLedger(root, run["run_id"], run["budget_ns"])
    ledger.transition("recovery" if resume else "validation")
    log = EventLog(root / "events", quota_bytes=1 << 30)
    guard = ResourceGuard(
        root, free_floor=run["free_disk_floor_bytes"], host_floor=run["host_available_floor_bytes"]
    )
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=child_target, args=(child, root, resume, stop_after))
    done = None
    try:
        guard.check()
        if resume:
            from oeis_learn.tracking.training_checkpoint import CheckpointStore
            from oeis_learn.rl.foundation_sft import runtime_record

            runtime = runtime_record(run["device"], run=run)
            if load_json((root / "checkpoints/runtime.json").read_bytes()) != runtime:
                raise ValueError("incompatible runtime; checkpoint recovery blocked")
            _, checkpoint = CheckpointStore(root / "checkpoints").recover(expected_runtime=runtime)
            ledger.check_checkpoint(checkpoint.manifest)
            log.append(
                f"resume-{ledger.state['sequence']}",
                dict(
                    kind="resume",
                    completed_update=checkpoint.manifest["completed_update"],
                    charged_budget_ns=ledger.state["charged_budget_ns"],
                    abandoned_updates="events after selected checkpoint are uncommitted",
                ),
            )
        process.start()
        child.close()
        heartbeat = time.monotonic() + 1
        while True:
            remaining = ledger.remaining_ns()
            if remaining <= 0:
                raise TimeoutError("arm elapsed deadline exhausted")
            if time.monotonic() >= heartbeat:
                ledger.heartbeat()
                resources = guard.check()
                log.append(
                    f"resources-{ledger.state['sequence']}",
                    dict(kind="resources", resources=resources),
                )
                heartbeat = time.monotonic() + 1
            if not parent.poll(min(0.1, remaining / 1e9)):
                if not process.is_alive():
                    raise RuntimeError("learner exited without completion")
                continue
            message = _receive(parent)
            kind = message["kind"]
            if kind == "phase":
                _send(parent, ledger.transition(message["phase"]))
            elif kind in ("update", "checkpoint"):
                # Attempt IDs include durable controller sequence so retried work
                # is distinguishable; committed progress is the selected manifest.
                log.append(
                    f"{kind}-{ledger.state['sequence']}-{message['completed_update']}", message
                )
                _send(parent, ledger.snapshot())
            elif kind == "done":
                done = message
                break
            else:
                raise RuntimeError(message.get("error", "unexpected learner message"))
        process.join(2)
        _reap(process)
        ledger.transition("completed" if done["completed_update"] == 3 else "paused")
        report = dict(
            status=ledger.state["status"],
            qualified=False,
            purpose="diagnostic",
            run_id=run["run_id"],
            completed_update=done["completed_update"],
            ledger=ledger.snapshot(),
            isolation="unqualified_process_boundary",
            diagnostic_small_host=run["diagnostic_small_host"],
        )
        replace_json(root / "training-report.json", report)
    except BaseException as exc:
        if process.pid is not None:
            _reap(process)
        ledger.transition("failed")
        replace_json(
            root / "training-report.json",
            dict(
                status="interrupted",
                qualified=False,
                error=f"{type(exc).__name__}: {exc}",
                ledger=ledger.snapshot(),
            ),
        )
    finally:
        parent.close()
        child.close()


def remaining_on_disk(root):
    """Read-only watchdog allowance includes an active controller's crash gap."""
    root = Path(root)
    run = load_json((root / "run.json").read_bytes())
    paths = sorted((root / "budget-events").glob("*.jsonl"))
    if not paths:
        return run["budget_ns"]
    # Ignore only an incomplete final line; the controller will quarantine it.
    last = None
    for path in paths:
        with path.open("rb") as stream:
            for line in stream:
                if line.endswith(b"\n"):
                    last = load_json(line)["data"]
    if last is None:
        raise ValueError("ledger has no durable start")
    pending = elapsed(last["clock"], clock()) if last["status"] in ACTIVE else 0
    return max(0, run["budget_ns"] - last["charged_budget_ns"] - pending)


def run_training(
    root, *, resume=False, stop_after=None, child_target=_learner, _controller_target=_controller
):
    root = Path(root).resolve()
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= 3):
        raise ValueError("stop-after-update must be 1, 2 or 3")
    with run_lock(root):
        if (root / "finalization.json").exists():
            raise ValueError("finalized run cannot resume training")
        if not resume and (root / "budget-events").exists():
            raise ValueError("existing ledger requires resume")
        allowance = remaining_on_disk(root) / 1e9
        if allowance <= 0:
            raise TimeoutError("arm budget exhausted; resume cannot refund time")
        process = mp.get_context("spawn").Process(
            target=_controller_target, args=(root, resume, stop_after, child_target)
        )
        process.start()
        process.join(allowance + 2)
        if process.is_alive():
            # Kill the process group, including a blocked learner, even if the
            # heartbeat controller itself is wedged. No inherited host group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _reap(process)
            raise TimeoutError("external controller deadline exhausted")
        if process.exitcode != 0:
            raise RuntimeError(f"controller failed with exit code {process.exitcode}")
        report = load_json((root / "training-report.json").read_bytes())
        if report["status"] == "interrupted":
            raise RuntimeError(report["error"])
        from oeis_learn.tracking.training_checkpoint import CheckpointStore

        path, checkpoint = CheckpointStore(root / "checkpoints").recover()
        return dict(
            report,
            checkpoint=str(path),
            checkpoint_sha256=checkpoint.manifest["blob_sha256"],
            result_path=str(root / "training-report.json"),
        )
