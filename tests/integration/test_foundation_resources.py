"""Bounded fault injection; no host disk or memory exhaustion."""

import time
from types import SimpleNamespace
import pytest
from tests.helpers.foundation_training import prepare, trainer
from oeis_learn.rl.foundation_sft import prepare_run
from oeis_learn.tracking.foundation_controller import ResourceGuard, run_lock, run_training
from oeis_learn.tracking.foundation_container import container_command


def _hang(connection, root, resume, stop_after):
    while True:
        time.sleep(0.1)


def _hung_controller(root, resume, stop_after, child_target):
    import os
    from oeis_learn.experiments.artifacts import load_json
    from oeis_learn.tracking.budget_ledger import BudgetLedger

    os.setsid()
    run = load_json((root / "run.json").read_bytes())
    ledger = BudgetLedger(root, run["run_id"], run["budget_ns"])
    ledger.transition("training")
    while True:
        time.sleep(0.1)


def test_outer_deadline_covers_hung_controller(training_inputs, tmp_path):
    from oeis_learn.tracking.foundation_controller import remaining_on_disk

    root = tmp_path / "run"
    prepare_run(
        *training_inputs,
        root,
        device="cpu",
        diagnostic=True,
        diagnostic_small_host=True,
        budget_ns=3_000_000_000,
    )
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="external controller"):
        run_training(root, _controller_target=_hung_controller)
    assert time.monotonic() - start < 8
    assert remaining_on_disk(root) == 0


def test_checkpoint_quota_stops_without_new_latest(training_inputs, tmp_path):
    root = tmp_path / "run"
    run = prepare(root, training_inputs)
    learner = trainer(root)
    learner.update()
    learner.store.quota = 1
    with pytest.raises(OSError, match="quota"):
        learner.save(dict(run_id=run["run_id"], sequence=1, charged_budget_ns=1))
    assert not (root / "checkpoints/latest.json").exists()


def test_fake_disk_and_two_low_memory_samples(tmp_path, monkeypatch):
    guard = ResourceGuard(tmp_path, free_floor=100, disk_usage=lambda _: SimpleNamespace(free=99))
    with pytest.raises(OSError, match="disk"):
        guard.check()
    guard = ResourceGuard(tmp_path, free_floor=100, disk_usage=lambda _: SimpleNamespace(free=1000))
    monkeypatch.setattr(
        "oeis_learn.tracking.foundation_controller.resource_metrics",
        lambda _: {"host_available": {"state": "measured", "value": 1}},
    )
    guard.check()
    with pytest.raises(OSError, match="two samples"):
        guard.check()


def test_controller_kills_hung_learner_and_keeps_ledger(training_inputs, tmp_path):
    root = tmp_path / "run"
    prepare_run(
        *training_inputs,
        root,
        device="cpu",
        diagnostic=True,
        diagnostic_small_host=True,
        budget_ns=3_000_000_000,
    )
    start = time.monotonic()
    with pytest.raises((RuntimeError, TimeoutError), match="deadline"):
        run_training(root, child_target=_hang)
    assert time.monotonic() - start < 8
    assert list((root / "budget-events").glob("*.jsonl"))
    with pytest.raises((RuntimeError, TimeoutError), match="budget"):
        run_training(root, resume=True)


def test_run_has_single_controller(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="active controller"):
            with run_lock(tmp_path):
                pass


def test_role_mount_plan_never_exposes_workspace_socket_or_gpu_to_workers(tmp_path):
    inputs, output = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    from oeis_learn.container_runtime import Engine

    args = dict(
        engine=Engine("docker", executable="docker", details={}),
        role="worker",
        image="sha256:" + "a" * 64,
        inputs=inputs,
        output=output,
        name="test",
        command=["python", "-V"],
    )
    command = container_command(**args)
    for flag in (
        "--network=none",
        "--read-only",
        "--memory=1073741824",
        "--memory-swap=1073741824",
        "--pids-limit=64",
        "--cpus=1",
        "--cap-drop=ALL",
    ):
        assert flag in command
    assert "--device" not in command
    assert not any("docker.sock" in x or "privileged" in x for x in command)
    assert len([x for x in command if x.startswith("type=bind")]) == 2
    with pytest.raises(ValueError, match="nonoverlapping"):
        container_command(**dict(args, inputs=tmp_path))
    with pytest.raises(ValueError, match="image ID"):
        container_command(**dict(args, image="rocm:latest"))
    with pytest.raises(ValueError, match="forbidden"):
        container_command(**dict(args, render_node="/dev/dri/renderD128"))


@pytest.fixture
def mismatched_cpu_threads(monkeypatch):
    import torch

    previous = torch.get_num_threads()
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_actual_controller_pause_resume_and_read_only_inspect(
    training_inputs, tmp_path, capsys, mismatched_cpu_threads
):
    import json
    from oeis_learn.cli.main import cli
    from oeis_learn.tracking.run_manager import inspect_foundation_run

    root = tmp_path / "run"
    assert (
        cli(
            [
                "foundation",
                "train",
                "--config",
                str(training_inputs[0]),
                "--pool",
                str(training_inputs[1]),
                "--run-dir",
                str(root),
                "--device",
                "cpu",
                "--diagnostic",
                "--diagnostic-small-host",
                "--stop-after-update",
                "1",
                "--json",
            ]
        )
        == 0
    )
    import torch

    assert json.loads((root / "checkpoints/runtime.json").read_text())["cpu_threads"] == 4
    # The resume controller must also normalize its newly spawned runtime.
    torch.set_num_threads(3)
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "paused" and first["completed_update"] == 1
    before = {str(p): p.read_bytes() for p in root.rglob("*.json*") if p.is_file()}
    report = inspect_foundation_run(root)
    assert report["lifecycle"] == "paused" and report["qualified"] is False
    assert {str(p): p.read_bytes() for p in root.rglob("*.json*") if p.is_file()} == before
    assert cli(["foundation", "resume", "--run-dir", str(root), "--json"]) == 0
    final = json.loads(capsys.readouterr().out)
    assert final["status"] == "completed" and final["completed_update"] == 3
    assert final["ledger"]["charged_budget_ns"] >= first["ledger"]["charged_budget_ns"]


def test_worker_oom_is_reaped_and_replaced(tmp_path):
    from oeis_learn.sandbox.worker_pool import WorkerPool
    from tests.helpers.worker_faults import oom_worker

    with WorkerPool(tmp_path, _worker_target=oom_worker) as pool:
        old = pool.all_slots[0].process.pid
        failure = pool.evaluate("i256.zero", [0], request_id="oom")
        assert failure.outcome == "execution_limit"
        assert "address_space" in failure.reason
        replacement = pool.evaluate("i256.const 1", [0], request_id="replacement")
        assert replacement.outputs == ["1"], replacement.to_dict()
        assert pool.all_slots[0].process.pid != old


@pytest.mark.docker
def test_actual_container_worker_mount_and_device_isolation(tmp_path):
    import os
    import subprocess

    image = os.environ.get("OEIS_FOUNDATION_TEST_IMAGE")
    if not image:
        pytest.skip("explicit pinned OEIS_FOUNDATION_TEST_IMAGE required")
    from oeis_learn.container_runtime import resolve_engine

    engine = resolve_engine()
    inputs, output = tmp_path / "visible", tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    (inputs / "sentinel").write_text("readable")
    secret = tmp_path / "private-truth"
    secret.write_text("never mounted")
    script = (
        "from pathlib import Path; import os; "
        "assert Path('/work/input/sentinel').read_text() == 'readable'; "
        "assert not Path('/dev/kfd').exists(); assert not Path('/run/docker.sock').exists(); "
        f"assert not Path({str(secret)!r}).exists(); "
        "assert not os.access('/work/input', os.W_OK); "
        "Path('/work/output/pass').write_text('isolated')"
    )
    command = container_command(
        engine=engine,
        role="worker",
        image=image,
        inputs=inputs,
        output=output,
        name="oeis-foundation-isolation-test",
        command=["python", "-c", script],
    )
    try:
        result = subprocess.run(command, timeout=30, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert (output / "pass").read_text() == "isolated"
    finally:
        engine.remove("oeis-foundation-isolation-test")


pytest_plugins = ["tests.helpers.foundation_training"]


def test_engine_removal_timeout_still_reaps_local_client(tmp_path, monkeypatch):
    import subprocess
    from oeis_learn.tracking.foundation_container import launch

    class Client:
        killed = False

        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired("docker run", timeout)
            return -9

        def poll(self):
            return None if not self.killed else -9

        def kill(self):
            self.killed = True

    client = Client()

    def failed_remove(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker rm", 2)

    monkeypatch.setattr(
        "oeis_learn.tracking.foundation_container.container_command", lambda **_: ["docker"]
    )
    monkeypatch.setattr(
        "oeis_learn.tracking.foundation_container.subprocess.Popen", lambda _: client
    )
    monkeypatch.setattr("oeis_learn.tracking.foundation_container.subprocess.run", failed_remove)
    with pytest.raises(subprocess.TimeoutExpired):
        from oeis_learn.container_runtime import Engine

        launch(seconds=1, engine=Engine("docker", executable="docker", details={}))
    assert client.killed
