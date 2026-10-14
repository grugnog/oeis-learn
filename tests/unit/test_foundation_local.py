"""Host transport must bypass SSH/SmolVM without bypassing the GPU broker."""

import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

spec = importlib.util.spec_from_file_location(
    "foundation_local", Path(__file__).parents[2] / "scripts/foundation_local.py"
)
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)
IMAGE = "sha256:" + "a" * 64


def policy(root):
    return {
        "project": "oeis-learn",
        "user": "labgpu-oeis-learn",
        "workspace": str(root),
        "home": "/var/lib/lab-gpu/oeis-learn",
        "foundation": True,
        "python": "/nix/store/python/bin/python3",
        "script": "/nix/store/broker/lab-gpu.py",
        "env": "/nix/store/coreutils/bin/env",
        "podman": "/nix/store/podman/bin/podman",
        "path": "/run/wrappers/bin:/nix/store/tools/bin",
        "cpus": 8,
        "buildMemory": "12g",
    }


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setattr(local.os, "getuid", lambda: 9801)
    with patch.object(
        local.pwd, "getpwnam", return_value=SimpleNamespace(pw_uid=9801, pw_gid=9801)
    ):
        return local.Host(policy(tmp_path))


def trusted_policy_file(path):
    result = Mock(wraps=path)
    result.stat.return_value = SimpleNamespace(st_uid=0, st_mode=path.stat().st_mode)
    return result


def test_load_policy_binds_checkout_and_installed_code(tmp_path):
    path = tmp_path / "gpu-runner.json"
    path.write_text(json.dumps({"oeis-learn": policy(tmp_path)}))
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "POLICY", trusted_policy_file(path)),
    ):
        assert local.load_policy("oeis-learn")["workspace"] == str(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(foundation=False),
        lambda p: p.update(workspace="/some/other/checkout"),
        lambda p: p.update(user="root"),
        lambda p: p.update(python="/workspace/python"),
        lambda p: p.update(script="/workspace/scripts/foundation_host.py"),
        lambda p: p.update(env="/nix/store/../env"),
    ],
)
def test_policy_rejects_wrong_workspace_account_or_mutable_root_code(tmp_path, mutation):
    value = policy(tmp_path)
    mutation(value)
    path = tmp_path / "gpu-runner.json"
    path.write_text(json.dumps({"oeis-learn": value}))
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "POLICY", trusted_policy_file(path)),
        pytest.raises(ValueError),
    ):
        local.load_policy("oeis-learn")


def test_writable_policy_and_missing_host_configuration_reject(tmp_path):
    path = tmp_path / "gpu-runner.json"
    with patch.object(local, "POLICY", path), pytest.raises(ValueError, match="lab host"):
        local.load_policy("oeis-learn")
    path.write_text(json.dumps({"oeis-learn": policy(tmp_path)}))
    path.chmod(0o666)
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "POLICY", trusted_policy_file(path)),
        pytest.raises(ValueError, match="root-owned"),
    ):
        local.load_policy("oeis-learn")


def test_image_ensure_reuses_one_immutable_identity(host):
    with patch.object(
        local.subprocess, "run", return_value=Mock(stdout=(IMAGE + "\n").encode())
    ) as run:
        assert host.ensure() == host.ensure() == IMAGE
    run.assert_called_once()
    command = run.call_args.args[0]
    assert command == [
        "sudo",
        "-n",
        "--",
        host.policy["python"],
        "-I",
        host.policy["script"],
        "broker",
        "oeis-learn",
    ]
    assert json.loads(run.call_args.kwargs["input"]) == {
        "version": 1,
        "op": "ensure",
        "cwd": ".",
        "argv": [],
    }


@pytest.mark.parametrize("value", ["tag:latest", "a" * 63, "sha256:" + "g" * 64])
def test_mutable_or_malformed_image_identity_rejects(host, value):
    with (
        patch.object(host, "broker", return_value=Mock(stdout=value.encode())),
        pytest.raises(ValueError, match="immutable"),
    ):
        host.ensure()


@pytest.mark.parametrize("operation", ["fetch", "prepare", "seal"])
def test_cpu_container_uses_native_bind_and_no_gpu_or_transport_socket(host, operation):
    network = operation == "fetch"
    argv = (
        ["data", "fetch"]
        if network
        else (
            ["seal-inputs", "--inputs", "inputs", "--accept-shortfall"]
            if operation == "seal"
            else ["prepare", "--output", "tmp-host"]
        )
    )
    with (
        patch.object(host, "ensure", return_value=IMAGE),
        patch.object(local.subprocess, "run", return_value=Mock(returncode=0)) as run,
    ):
        host.cpu_worker(IMAGE, argv, 600)
    command = run.call_args_list[0].args[0]
    assert command[0] == host.policy["podman"] and "sudo" not in command
    assert f"type=bind,src={host.policy['workspace']},target=/workspace,rw" in command
    assert "--network=" + ("slirp4netns" if network else "none") in command
    assert "--userns=keep-id" in command and "9801:9801" in command
    assert not any(
        "--device" in arg or "docker.sock" in arg or "podman.sock" in arg for arg in command
    )
    assert "ssh" not in command and "smolvm" not in command and "--privileged" not in command
    assert command[-(2 + len(argv)) :] == [IMAGE, "/workspace/scripts/foundation_run.py", *argv]
    name = command[command.index("--name") + 1]
    assert run.call_args_list[-1].args[0][-4:] == ["rm", "--force", "--ignore", name]


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(7, ["podman"]),
        subprocess.TimeoutExpired(["podman"], 30),
        KeyboardInterrupt(),
    ],
)
def test_cpu_error_timeout_or_interruption_reclaims_container(host, error):
    with (
        patch.object(host, "ensure", return_value=IMAGE),
        patch.object(local.subprocess, "run", side_effect=[error, Mock(returncode=0)]) as run,
        pytest.raises(type(error)),
    ):
        host.cpu_worker(IMAGE, ["prepare"], 600)
    assert run.call_count == 2
    assert "--force" in run.call_args_list[1].args[0]


def test_cleanup_failure_is_reported(host):
    with (
        patch.object(host, "ensure", return_value=IMAGE),
        patch.object(
            local.subprocess,
            "run",
            side_effect=[Mock(returncode=0), subprocess.CalledProcessError(1, ["rm"])],
        ),
        pytest.raises(RuntimeError, match="cleanup failed"),
    ):
        host.cpu_worker(IMAGE, ["prepare"], 600)


def test_cpu_uses_one_sudo_transition_with_project_worker_cleanup(host):
    with (
        patch.object(host, "ensure", return_value=IMAGE),
        patch.object(local.subprocess, "run", return_value=Mock(returncode=0)) as run,
    ):
        host.cpu("prepare", "--output", "tmp-host", timeout=600)
    run.assert_called_once()
    command = run.call_args.args[0]
    assert command[:5] == ["sudo", "-n", "-u", "labgpu-oeis-learn", "--"]
    assert "HOME=/var/lib/lab-gpu/oeis-learn" in command
    assert "XDG_RUNTIME_DIR=/run/user/9801" in command
    assert command[-4:-1] == ["oeis-learn", IMAGE, "600"]
    assert json.loads(command[-1]) == ["prepare", "--output", "tmp-host"]
    assert "_cpu-worker" in command and run.call_args.kwargs["timeout"] == 645


@pytest.mark.parametrize(
    "argv",
    [
        ["train"],
        ["check", "--inputs", "tmp-inputs"],
        ["data", "other"],
        ["seal-inputs", "--inputs", "inputs"],
    ],
)
def test_cpu_worker_rejects_nested_gpu_or_unknown_operations(host, argv):
    with pytest.raises(ValueError):
        host.cpu_worker(IMAGE, argv, 600)


def test_cpu_worker_rejects_wrong_account_and_unexpected_network(host):
    with (
        patch.object(local.os, "getuid", return_value=0),
        pytest.raises(ValueError, match="rootless"),
    ):
        host.cpu_worker(IMAGE, ["prepare"], 600)
    with pytest.raises(ValueError, match="networking"):
        host.cpu("prepare", network=True)


def test_cleanup_error_does_not_hide_original_failure(host, capsys):
    original = subprocess.CalledProcessError(17, ["prepare"])
    with (
        patch.object(
            local.subprocess,
            "run",
            side_effect=[original, subprocess.CalledProcessError(1, ["rm"])],
        ),
        pytest.raises(subprocess.CalledProcessError) as caught,
    ):
        host.cpu_worker(IMAGE, ["prepare"], 600)
    assert caught.value.returncode == 17
    assert "cleanup failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "operation,extra",
    [
        ("status", []),
        ("cancel", []),
        ("resume", []),
        ("evaluate", ["development", "latest"]),
        ("finalize", ["best-development"]),
    ],
)
def test_control_dispatch_has_no_cpu_container_image_ensure_or_ssh(tmp_path, operation, extra):
    host = Mock()
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "load_policy", return_value=policy(tmp_path)),
        patch.object(local, "Host", return_value=host),
    ):
        assert local.main([operation, "--run", "runs/test"]) == 0
    host.broker.assert_called_once_with("foundation", operation, "runs/test", *extra)
    host.cpu.assert_not_called()
    host.ensure.assert_not_called()


@pytest.mark.parametrize("operation,profile", [("train", "fresh-pilot-1h"), ("smoke", "smoke")])
def test_training_checks_inputs_in_cpu_container_then_uses_private_broker(
    tmp_path, operation, profile
):
    host = Mock()
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "load_policy", return_value=policy(tmp_path)),
        patch.object(local, "Host", return_value=host),
    ):
        assert local.main([operation, "--inputs", "tmp-inputs", "--run", "runs/test"]) == 0
    assert host.mock_calls == [
        ("cpu", ("check", "--inputs", "tmp-inputs", "--inputs-only"), {"timeout": 600}),
        ("broker", ("foundation", "run", "tmp-inputs", "runs/test", profile), {}),
    ]


@pytest.mark.parametrize("operation", ["data", "prepare"])
def test_cpu_commands_do_not_acquire_gpu_lease(tmp_path, operation):
    host = Mock()
    argv = {
        "data": ["data", "fetch"],
        "prepare": ["prepare"],
    }[operation]
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "load_policy", return_value=policy(tmp_path)),
        patch.object(local, "Host", return_value=host),
    ):
        assert local.main(argv) == 0
    host.broker.assert_not_called()
    assert host.cpu.call_args.args[0] == operation
    assert host.cpu.call_args.kwargs.get("network", False) == (operation == "data")


def test_pack_needs_no_sudo_policy_image_or_container(tmp_path):
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "load_policy") as policy_loader,
        patch.object(local, "pack_run") as pack,
    ):
        assert local.main(["pack", "--run", "runs/test", "--output", "tmp-analysis/test.zip"]) == 0
    pack.assert_called_once_with("runs/test", "tmp-analysis/test.zip", checkpoint=False)
    policy_loader.assert_not_called()


def test_real_host_analysis_pack_is_standard_library_only_and_excludes_truth(tmp_path):
    run = tmp_path / "runs/test"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"identity": "test-run"}))
    (run / "status.json").write_text(json.dumps({"status": "interrupted"}))
    (run / "truth.json").write_text(json.dumps({"private": "DO NOT EXPORT"}))
    with patch.object(local, "ROOT", tmp_path), patch.object(local.subprocess, "run") as process:
        result = local.pack_run("runs/test", "tmp-analysis/test.zip")
    process.assert_not_called()
    with zipfile.ZipFile(tmp_path / "tmp-analysis/test.zip") as archive:
        assert "truth.json" not in archive.namelist()
        assert json.loads(archive.read("analysis.json"))["status"]["status"] == "interrupted"
    assert result["sha256"].startswith("sha256:")


def test_training_success_packs_after_broker_without_new_sudo_transition(tmp_path):
    (tmp_path / "runs/test").mkdir(parents=True)
    (tmp_path / "runs/test/run.json").write_text("{}")
    host = Mock()
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "load_policy", return_value=policy(tmp_path)),
        patch.object(local, "Host", return_value=host),
        patch.object(local, "pack_run") as pack,
    ):
        assert local.main(["train", "--inputs", "tmp-inputs", "--run", "runs/test"]) == 0
    assert host.cpu.call_count == host.broker.call_count == 1
    pack.assert_called_once_with("runs/test", "tmp-analysis/test.zip")


def test_training_failure_is_retained_if_analysis_pack_also_fails(tmp_path, capsys):
    (tmp_path / "runs/test").mkdir(parents=True)
    (tmp_path / "runs/test/run.json").write_text("{}")
    host = Mock()
    host.broker.side_effect = subprocess.CalledProcessError(17, ["broker"])
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "load_policy", return_value=policy(tmp_path)),
        patch.object(local, "Host", return_value=host),
        patch.object(local, "pack_run", side_effect=OSError("pack unavailable")),
    ):
        assert local.main(["train", "--inputs", "tmp-inputs", "--run", "runs/test"]) == 17
    assert "Analysis packaging also failed" in capsys.readouterr().err


def test_paths_outside_host_checkout_reject(tmp_path):
    with (
        patch.object(local, "ROOT", tmp_path / "checkout"),
        pytest.raises(ValueError, match="registered host"),
    ):
        local.relative(tmp_path / "outside")


def test_resume_rejects_new_budget_and_settings():
    with pytest.raises(SystemExit):
        local.main(["resume", "--run", "runs/test", "--seconds", "7200"])
