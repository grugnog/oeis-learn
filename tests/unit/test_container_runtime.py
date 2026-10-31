import subprocess
from unittest.mock import Mock, patch

import pytest

from oeis_learn.container_runtime import Engine, normalized_image

IMAGE = "sha256:" + "a" * 64


@pytest.mark.parametrize("name", ["docker", "podman"])
def test_startup_diagnostics_preserve_command_and_containment(name):
    engine = Engine(name, executable=name, details={})
    options = dict(image=IMAGE, name="diagnostic", mounts=[], argv=["python", "-c", "pass"])
    ordinary = engine.command(**options)
    diagnostic = engine.command(**options, startup_diagnostics=True)
    if name == "podman":
        assert diagnostic[:3] == ["podman", "--log-level=debug", "run"]
        assert diagnostic[:1] + diagnostic[2:] == ordinary
    else:
        assert diagnostic == ordinary


@pytest.mark.parametrize("name", ["docker", "podman"])
def test_role_bounds_and_mounts_are_portable(tmp_path, name):
    source, output = tmp_path / "inputs", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    engine = Engine(name, executable=name, details={})
    command = engine.command(
        image=IMAGE,
        name="oeis-test",
        mounts=[(source, "/work/input", False), (output, "/work/output", True)],
        argv=["python", "worker.py"],
    )
    assert command[0] == name and "--network=none" in command
    assert f"type=bind,src={source},dst=/work/input,readonly" in command
    assert "--privileged" not in command and "--runtime=crun" not in command
    assert not any("sock" in value for value in command)
    assert command[-2:] == [IMAGE, "worker.py"]


def test_podman_rootless_mapping_and_supplementary_groups():
    engine = Engine(
        "podman",
        executable="podman",
        details={"host": {"security": {"rootless": True}, "ociRuntime": {"name": "crun"}}},
    )
    assert "--userns=keep-id" in engine.user_args()
    assert "--group-add=keep-groups" in engine.user_args()


@pytest.mark.parametrize("name", ["docker", "podman"])
def test_reclamation_is_engine_specific_and_bounded(name):
    engine = Engine(name, executable=name, details={})
    with patch.object(engine, "call") as call:
        engine.remove("oeis-test")
    args = call.call_args.args[0]
    assert ("--time=0" in args) == (name == "podman")
    assert call.call_args.kwargs["timeout"] == 30


def test_inspect_normalizes_actual_image_identity():
    engine = Engine("podman", executable="podman", details={})
    with patch.object(engine, "call", return_value=Mock(stdout=("a" * 64).encode())):
        assert engine.image("tag") == IMAGE
    with pytest.raises(ValueError):
        normalized_image("tag:latest")


def test_cleanup_does_not_swallow_engine_connection_errors():
    engine = Engine("docker", executable="docker", details={})
    with patch.object(
        engine,
        "call",
        side_effect=subprocess.CalledProcessError(1, ["docker"], stderr=b"daemon unavailable"),
    ):
        with pytest.raises(RuntimeError, match="reclamation"):
            engine.remove("oeis-test")


def test_auto_detect_tries_next_usable_connection():
    def call(engine, *args, **kwargs):
        if engine.name == "docker":
            raise RuntimeError("daemon unavailable")
        return {"host": {}}

    with (
        patch("shutil.which", side_effect=lambda name: "/bin/" + name),
        patch.object(Engine, "info", call),
    ):
        engine = Engine("auto")
    assert engine.name == "podman" and engine.executable == "/bin/podman"


def test_rootless_docker_maps_namespace_root_to_caller():
    engine = Engine("docker", executable="docker", details={"SecurityOptions": ["name=rootless"]})
    assert engine.user_args() == ["--user", "0:0"]


@pytest.mark.parametrize("name", ["docker", "podman"])
def test_gpu_devices_are_explicit_and_cpu_cannot_receive_them(tmp_path, name):
    engine = Engine(name, executable=name, details={})
    options = dict(
        image=IMAGE, name="test", mounts=[(tmp_path, "/work/input", False)], argv=["python"]
    )
    with patch.object(engine, "device_args", return_value=["--device", "/dev/kfd"]) as devices:
        assert "--device" in engine.command(**options, gpu=True, render_node="/dev/dri/renderD128")
        devices.assert_called_once_with("/dev/dri/renderD128")
    with pytest.raises(ValueError, match="forbidden"):
        engine.command(**options, gpu=False, render_node="/dev/dri/renderD128")


def test_cleanup_filters_exact_run_and_attempts_all_reclamations():
    engine = Engine("docker", executable="docker", details={})
    with (
        patch.object(engine, "call", return_value=Mock(stdout=b"oeis-one\noeis-two\n")) as call,
        patch.object(engine, "remove", side_effect=[RuntimeError("failed"), None]) as remove,
    ):
        with pytest.raises(RuntimeError, match="cleanup failed"):
            engine.cleanup("token")
    assert "label=oeis.run=token" in call.call_args.args[0]
    assert remove.call_count == 2
