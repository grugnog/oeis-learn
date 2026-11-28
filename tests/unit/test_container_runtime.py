import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from oeis_learn.container_runtime import Engine, normalized_image

IMAGE = "sha256:" + "a" * 64


def test_build_progress_does_not_corrupt_the_host_json_result(tmp_path, capfd):
    engine_cli = tmp_path / "engine"
    engine_cli.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = build ]; then\n'
        "  echo '[1/2] STEP 1/3: FROM base'\n"
        "  echo 'build warning' >&2\n"
        f"else\n  echo '{IMAGE}'\nfi\n"
    )
    engine_cli.chmod(0o700)
    engine = Engine("podman", executable=str(engine_cli))
    print(json.dumps({"image": engine.build(tmp_path)}))
    out, err = capfd.readouterr()
    assert json.loads(out) == {"image": IMAGE}
    assert "[1/2] STEP" in err and "build warning" in err


def test_monitored_build_checks_the_supervised_image_id_and_rechecks_storage(tmp_path):
    engine_cli = tmp_path / "podman"
    engine_cli.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = build ]; then\n'
        '  while [ "$1" != --iidfile ]; do shift; done\n'
        f'  echo {IMAGE} > "$2"\n'
        "  sleep 1.25\n"
        f"else echo {IMAGE}; fi\n"
    )
    engine_cli.chmod(0o700)
    engine = Engine("podman", executable=str(engine_cli), details={"host": {"serviceIsRemote": False}})
    observations = []
    evidence = tmp_path / "image-build"
    iid = evidence / "image-id"
    assert engine.build(tmp_path, seconds=8, evidence=evidence, iidfile=iid,
                        check_resources=lambda: observations.append(True)) == IMAGE
    assert len(observations) >= 2 and normalized_image(iid.read_text()) == IMAGE
    assert json.loads((evidence / "cleanup.json").read_bytes())["acknowledged"] is True
    with patch.object(engine, "image", return_value="sha256:" + "b" * 64), pytest.raises(
        ValueError, match="built image ID differs"
    ):
        other = tmp_path / "second"
        engine.build(tmp_path, seconds=8, evidence=other, iidfile=other / "image-id")


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


def test_runtime_environment_uses_explicit_mount_and_rejects_escape(tmp_path):
    from oeis_learn.container_runtime import private_runtime_environment
    assert private_runtime_environment(tmp_path)['HOME']=='/work/output/.runtime/home'
    assert private_runtime_environment(tmp_path,destination='/output/learner')['TMPDIR']=='/output/learner/.runtime/tmp'
    for destination in ('relative','/output/../inputs','/'):
        with pytest.raises(ValueError):private_runtime_environment(tmp_path,destination=destination)
    (tmp_path/'.runtime/tmp').rmdir();(tmp_path/'.runtime/tmp').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='symlink'):private_runtime_environment(tmp_path)


@pytest.mark.parametrize('name', ['docker', 'podman'])
def test_explicit_readonly_files_preserve_mount_and_network_boundary(tmp_path, name):
    source = tmp_path / 'original.json'
    source.write_text('{}')
    engine = Engine(name, executable=name, details={})
    args = dict(image=IMAGE, name='custodian', argv=['python'], mounts=[(source, '/work/inputs/original.json', False)])
    command = engine.command(**args)
    assert f'type=bind,src={source},dst=/work/inputs/original.json,readonly' in command
    assert '--network=none' in command and '--read-only' in command
    with pytest.raises(ValueError, match='read-only regular file'):
        engine.command(**dict(args, mounts=[(source, '/work/inputs/original.json', True)]))
    alias = tmp_path / 'alias.json'
    alias.symlink_to(source)
    with pytest.raises(ValueError, match='read-only regular file'):
        engine.command(**dict(args, mounts=[(alias, '/work/inputs/original.json', False)]))
    import os
    pipe = tmp_path / 'pipe'
    os.mkfifo(pipe)
    with pytest.raises(ValueError, match='read-only regular file'):
        engine.command(**dict(args, mounts=[(pipe, '/work/inputs/original.json', False)]))


@pytest.mark.parametrize("remote", [False, True])
def test_recorded_build_keeps_log_tails_and_requires_local_termination(tmp_path, remote):
    engine_cli = tmp_path / "podman"
    engine_cli.write_text(
        "#!/bin/sh\n"
        "python3 -c 'print(\"x\" * 100000)'\n"
        "echo 'fixture final build step'\n"
        "echo 'fixture layer could not be committed' >&2\n"
        "exit 125\n"
    )
    engine_cli.chmod(0o700)
    engine = Engine(
        "podman", executable=str(engine_cli), details={"host": {"serviceIsRemote": remote}}
    )
    evidence = tmp_path / "build-logs"
    with pytest.raises(RuntimeError, match="fixture layer could not be committed"):
        engine.build(tmp_path, seconds=10, evidence=evidence)
    process = json.loads((evidence / "process.json").read_bytes())
    assert process["returncode"] == 125 and process["cleanup_failure"] is None
    assert "fixture final build step" in (evidence / "stdout.log").read_text()
    assert (evidence / "stdout.log").stat().st_size <= 65536
    assert json.loads((evidence / "cleanup.json").read_bytes())["acknowledged"] == (not remote)
