"""Portable host commands run directly through the selected user's engine."""

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oeis_learn.container_runtime import Engine

spec = importlib.util.spec_from_file_location(
    "foundation_local", Path(__file__).parents[2] / "scripts/foundation_local.py"
)
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)
IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def host(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/foundation_host.py").write_text("fixture")
    adapter = tmp_path / "src/oeis_learn/container_runtime.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("fixture")
    engine = Mock(spec=Engine)
    engine.name, engine.executable = "podman", "podman"
    engine.facts.return_value = {
        "memory_bytes": 32 << 30,
        "cpus": 8,
        "rootless": True,
        "selinux": False,
        "runtime": "crun",
        "version": "test",
    }
    engine.image.return_value = IMAGE
    engine.build.return_value = IMAGE
    engine.command.side_effect = lambda **kw: ["podman", "run", kw["name"]]
    with patch.object(local, "ROOT", tmp_path), patch.object(local, "Engine", return_value=engine):
        yield local.Host(local.parser().parse_args(["--image", IMAGE, "image", "ensure"]))


def test_image_inspection_is_cached_and_immutable(host):
    assert host.ensure() == host.ensure() == IMAGE
    host.engine.image.assert_called_once_with(IMAGE)
    host.engine.build.assert_not_called()


def test_cpu_launch_has_workspace_and_cache_but_no_gpu(host):
    with patch.object(local.subprocess, "run", return_value=Mock(returncode=0)) as run:
        host.cpu("prepare", "--output", "inputs")
    options = host.engine.command.call_args.kwargs
    assert options["image"] == IMAGE
    assert options["argv"] == [
        "python",
        "/opt/oeis-build/scripts/foundation_tasks.py",
        "prepare",
        "--output",
        "inputs",
    ]
    assert options["mounts"][0] == (local.ROOT, "/workspace", True)
    assert not options["network"] and not options.get("gpu", False)
    assert options["environment"]["OEIS_WORKSPACE"] == "/workspace"
    assert run.call_args.kwargs["stdin"] == subprocess.DEVNULL
    host.engine.remove.assert_called_once_with(options["name"])


@pytest.mark.parametrize("argv,network", [(("prepare",), True), (("data", "fetch"), False)])
def test_only_bulk_acquisition_gets_network(host, argv, network):
    with pytest.raises(ValueError):
        host.cpu(*argv, network=network)
    host.engine.command.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        KeyboardInterrupt(),
        subprocess.TimeoutExpired("engine", 5),
        subprocess.CalledProcessError(7, "engine"),
    ],
)
def test_cpu_failure_always_reclaims_container(host, error):
    with patch.object(local.subprocess, "run", side_effect=error), pytest.raises(type(error)):
        host.cpu("check", "--inputs", "inputs", "--inputs-only")
    host.engine.remove.assert_called_once()


def test_pipeline_directly_starts_bounded_coordinator_and_cleans_labels(host):
    process = Mock(pid=1234)
    process.wait.return_value = 0
    process.poll.return_value = 0
    with patch.object(local.subprocess, "Popen", return_value=process) as spawn:
        host.pipeline("run", "inputs", "runs/new", "fresh-extended-12h")
    command = spawn.call_args.args[0]
    assert command[0] == local.sys.executable
    assert command[1] == str(local.ROOT / "scripts/foundation_host.py")
    policy = json.loads(command[2])
    assert policy["state_root"] == str(local.ROOT / "run-state")
    assert policy["engine"] == "podman"
    assert command[3:] == [IMAGE, "run", "inputs", "runs/new", "fresh-extended-12h"]
    process.wait.assert_called_once_with(timeout=43245)
    assert host.engine.cleanup.call_count == 2
    assert spawn.call_args.kwargs["start_new_session"]


def test_watchdog_terminates_group_and_reclaims_containers(host):
    process = Mock(pid=1234)
    process.wait.side_effect = [subprocess.TimeoutExpired("controller", 600), 0]
    process.poll.return_value = None
    with (
        patch.object(local.subprocess, "Popen", return_value=process),
        patch.object(local.os, "killpg") as kill,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        host.pipeline("check")
    kill.assert_called_once_with(1234, local.signal.SIGTERM)
    assert host.engine.cleanup.call_count == 2


def test_control_and_pack_do_not_need_an_engine(tmp_path):
    from oeis_learn.fresh.common import write

    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "Engine", side_effect=AssertionError("engine used")),
    ):
        write(local.state_path("runs/new") / "status.json", {"status": "paused"})
        assert local.main(["status", "--run", "runs/new"]) == 0


def test_paths_cannot_escape_workspace(host):
    with pytest.raises(ValueError):
        local.relative("../outside")


def test_main_training_routes_through_cpu_check_and_direct_pipeline(tmp_path):
    with patch.object(local, "ROOT", tmp_path), patch.object(local, "Host") as factory:
        assert (
            local.main(
                [
                    "--engine",
                    "docker",
                    "train",
                    "--profile",
                    "fresh-extended-12h",
                    "--inputs",
                    "inputs",
                    "--run",
                    "runs/new",
                ]
            )
            == 0
        )
        factory.return_value.cpu.assert_called_once_with(
            "check", "--inputs", "inputs", "--inputs-only", timeout=1800
        )
        factory.return_value.pipeline.assert_called_once_with(
            "run", "inputs", "runs/new", "fresh-extended-12h"
        )
