"""Transport/isolation checks; actual container acceptance is explicitly gated."""

import os
import shutil
import subprocess
import pytest
from oeis_learn.experiments.artifacts import (
    canonical_bytes,
    load_json,
    compute_canonical_digest as digest,
)
from oeis_learn.tracking.role_coordinator import (
    copy_view,
    QueueRunner,
    DockerRunner,
    RoleCoordinator,
)
from oeis_learn.tracking.foundation_container import container_command
from oeis_learn.container_runtime import Engine
from oeis_learn.sandbox.pipeline import Limits


def test_enumerated_views_reject_paths_and_symlinks(tmp_path):
    private = tmp_path / "private.json"
    private.write_text("private")
    visible = tmp_path / "visible.json"
    visible.write_text("visible")
    view = copy_view(tmp_path / "view", {"visible.json": visible})
    assert list(view.iterdir()) == [view / "visible.json"]
    with pytest.raises(ValueError):
        copy_view(tmp_path / "bad", {"../private.json": private})
    alias = tmp_path / "alias"
    alias.symlink_to(private)
    with pytest.raises(ValueError, match="regular"):
        copy_view(tmp_path / "alias-view", {"alias": alias})


def test_transport_zero_deadline_charges_and_response_binding(tmp_path):
    runner = QueueRunner(tmp_path)
    lost = runner.run_single("i256.const 1", range(20), request_id="x", deadline_ns=0)
    assert lost.outcome == "execution_limit" and lost.fuel == 50000000
    request = load_json(next(tmp_path.glob("*.request.json")).read_bytes())
    response = tmp_path / (digest("x")[7:] + ".response.json")
    response.write_bytes(
        canonical_bytes(dict(request_sha256=digest(request), evidence=lost.to_dict()))
    )
    assert runner.run_single("i256.const 1", range(20), request_id="x").fuel == lost.fuel
    with pytest.raises(ValueError, match="binding|immutable"):
        runner.run_single("i256.const 2", range(20), request_id="x")


def test_durable_worker_result_replay_and_tampering_without_docker(tmp_path):
    runner = DockerRunner(tmp_path, "sha256:" + "a" * 64)
    try:
        value = runner.run_single("i256.const 7", range(20), request_id="request", deadline_ns=0)
        assert value.fuel == 50000000
        assert (
            runner.run_single("i256.const 7", range(20), request_id="request").to_dict()
            == value.to_dict()
        )
        with pytest.raises(ValueError, match="conflicting"):
            runner.run_single("i256.const 8", range(20), request_id="request")
        target = next(tmp_path.glob("*/result.json"))
        target.write_text("{}")
        with pytest.raises(ValueError, match="changed"):
            runner.run_single("i256.const 7", range(20), request_id="request")
    finally:
        runner.close()


def test_coordinator_excludes_second_controller_and_bounds_operations(tmp_path):
    root = tmp_path / "controller"
    first = RoleCoordinator(root, "sha256:" + "a" * 64)
    try:
        with pytest.raises(RuntimeError, match="active controller"):
            RoleCoordinator(root, "sha256:" + "a" * 64)
        with pytest.raises(ValueError, match="bounded"):
            first.run(
                "arbitrary", role="learner", inputs=tmp_path, output=tmp_path / "out", seconds=1
            )
    finally:
        first.close()


def test_deadline_reclaims_then_replaces_without_regenerating_request(tmp_path, monkeypatch):
    import threading
    import time
    from oeis_learn.tracking import role_coordinator as module

    class Process:
        def poll(self):
            return None

    runner = DockerRunner(tmp_path, "sha256:" + "a" * 64, workers=1)
    view, output = tmp_path / "input", tmp_path / "output"
    view.mkdir()
    output.mkdir()
    dead = dict(name="timed-out", process=Process(), view=view, output=output)
    runner.services.append(dead)
    runner.slots.put(dead)
    replaced = threading.Event()
    calls = []
    monkeypatch.setattr(module, "reclaim", lambda name, process, **kwargs: calls.append(name))

    def replace():
        assert dead["reclaimed"] and calls == ["timed-out"]
        replaced.set()

    monkeypatch.setattr(runner, "_replace", replace)
    try:
        result = runner.run_single(
            "i256.const 7",
            range(20),
            request_id="deadline",
            deadline_ns=time.monotonic_ns() + 10_000_000,
        )
        assert replaced.wait(2)
        assert result.outcome == "execution_limit" and result.fuel == 50000000
        assert not list(view.iterdir()) and runner.slots.empty()
        assert (
            runner.run_single("i256.const 7", range(20), request_id="deadline").to_dict()
            == result.to_dict()
        )
        assert calls == ["timed-out"]
    finally:
        runner.close()


def test_role_command_is_confined_and_cpu_generation_is_explicit(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    argv = container_command(
        role="generation",
        image="sha256:" + "a" * 64,
        inputs=inputs,
        output=output,
        name="fixture",
        command=["python", "-m", "oeis_learn.tracking.role_worker", "generate"],
        gpu=False,
        engine=Engine("docker", executable="docker", details={}),
    )
    assert (
        "--network=none" in argv and "--memory=" + str(16 << 30) in argv and "--read-only" in argv
    )
    assert not any("docker.sock" in arg or "--device" in arg for arg in argv)
    with pytest.raises(ValueError, match="GPU capability"):
        container_command(
            role="evaluator",
            image="sha256:" + "a" * 64,
            inputs=inputs,
            output=output,
            name="fixture",
            command=["python"],
            gpu=True,
        )


def test_actual_container_role_views_and_reclamation(tmp_path):
    image = os.environ.get("OEIS_ROLE_TEST_IMAGE")
    if not image:
        pytest.skip("actual engine image required; CPU fixtures do not substitute")
    from oeis_learn.container_runtime import resolve_engine

    engine = resolve_engine()
    assert shutil.which(engine.executable), "explicit engine acceptance requires its CLI"
    # Execute a real isolated worker with an enumerated read-only view. This test
    # checks cgroup mounts/reclamation, not the workstation's memory reserve.
    source = "i256.const 7"
    inputs = tmp_path / "input"
    inputs.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (inputs / "request.json").write_bytes(
        canonical_bytes(
            dict(
                source=source,
                indices=list(range(100)),
                request_id="actual",
                limits=Limits().__dict__,
            )
        )
    )
    name = "oeis-role-acceptance-" + tmp_path.name
    argv = container_command(
        engine=engine,
        role="worker",
        image=image,
        inputs=inputs,
        output=output,
        name=name,
        command=["python", "-m", "oeis_learn.tracking.role_worker", "candidate"],
    )
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, (stdout, stderr)
        record = load_json((output / "result.json").read_bytes())
        assert record["outcome"] is None and record["outputs"] == ["7"] * 100
        assert (
            record["reference_outputs"] == record["outputs"]
            and record["identities"]["backend"] == "rust"
        )
        info = load_json(subprocess.check_output([engine.executable, "inspect", name]))[0]
        assert info["HostConfig"]["NetworkMode"] == "none"
        assert info["HostConfig"]["ReadonlyRootfs"] and info["HostConfig"]["Memory"] == 1 << 30
        assert {m["Destination"] for m in info["Mounts"] if m["Type"] == "bind"} == {
            "/work/input",
            "/work/output",
        }
        assert next(m for m in info["Mounts"] if m["Destination"] == "/work/input")["RW"] is False
    finally:
        from oeis_learn.tracking.role_coordinator import reclaim

        reclaim(name, process, engine=engine)
    assert subprocess.run([engine.executable, "inspect", name], capture_output=True).returncode != 0
    # A persistent worker retains its runtime cache and is reclaimed as a cgroup.
    runner = DockerRunner(tmp_path / "live", image, workers=1, engine=engine)
    try:
        runner.start()
        for index in range(2):
            value = runner.run_single(source, range(100), request_id=f"cache-{index}")
            assert value.outcome is None and value.outputs == ["7"] * 100
        assert value.phase_timings_ns and all(v >= 0 for v in value.phase_timings_ns.values())
    finally:
        runner.close()


def test_reclamation_is_bounded_and_reaps_the_client(monkeypatch):
    from unittest.mock import Mock
    from oeis_learn.tracking import role_coordinator as module

    engine = Mock()
    process = Mock()
    process.poll.return_value = None
    module.reclaim("fixture", process, engine=engine)
    engine.remove.assert_called_once_with("fixture")
    process.kill.assert_called_once()
    process.wait.assert_called_once_with(timeout=5)
