"""Compiler storage remains private, disk-backed and separate from evidence."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from oeis_learn.container_runtime import Engine, private_runtime_environment
from oeis_learn.tracking.foundation_container import container_command

spec = importlib.util.spec_from_file_location(
    "storage_host", Path(__file__).parents[2] / "scripts/foundation_host.py"
)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
IMAGE = "sha256:" + "a" * 64


def test_runtime_directories_are_private_and_do_not_share_caches(tmp_path):
    for role in ("learner", "generation"):
        output = tmp_path / role
        output.mkdir()
        env = private_runtime_environment(output)
        assert env["TMPDIR"] == env["TMP"] == env["TEMP"]
        for value in set(env.values()):
            path = output / Path(value).relative_to("/work/output")
            assert path.is_dir() and path.stat().st_mode & 0o777 == 0o700
        (output / ".runtime/cache/compiled").write_bytes(role.encode())
        assert private_runtime_environment(output) == env  # safe repeat, cache retained
    assert (tmp_path / "learner/.runtime/cache/compiled").read_bytes() == b"learner"


@pytest.mark.parametrize(
    "relative", [".runtime", ".runtime/tmp", ".runtime/home", ".runtime/cache"]
)
def test_runtime_directory_symlinks_never_escape_the_role(tmp_path, relative):
    output, private = tmp_path / "output", tmp_path / "private"
    output.mkdir()
    private.mkdir()
    link = output / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        private_runtime_environment(output)
    assert not list(private.iterdir())


@pytest.mark.parametrize("launcher", ["fresh", "generic"])
@pytest.mark.parametrize("engine_name", ["docker", "podman"])
def test_both_launchers_keep_scratch_in_the_only_writable_mount(tmp_path, launcher, engine_name):
    inputs, output = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    engine = Engine(engine_name, executable=engine_name, details={})
    if launcher == "fresh":
        with patch.object(host.Engine, "from_config", return_value=engine):
            argv = host.container_command({}, IMAGE, "test", "worker", inputs, output, ["python"])
    else:
        argv = container_command(
            role="worker",
            image=IMAGE,
            inputs=inputs,
            output=output,
            name="test",
            command=["python"],
            engine=engine,
        )
    assert argv.count("--mount") == 2
    assert "--network=none" in argv and "--read-only" in argv
    assert not any("--device" in arg for arg in argv)
    assert f"type=bind,src={inputs},dst=/work/input,readonly" in argv
    assert f"type=bind,src={output},dst=/work/output" in argv
    env = dict(argv[n + 1].split("=", 1) for n, arg in enumerate(argv) if arg == "--env")
    assert env["HOME"].startswith("/work/output/.runtime/")
    assert env["TMPDIR"] == "/work/output/.runtime/tmp"
    assert env["XDG_CACHE_HOME"] == "/work/output/.runtime/cache"


def test_publication_excludes_compiler_cache_but_keeps_launch_diagnostics(tmp_path):
    job, published = tmp_path / "job", tmp_path / "run"
    role = job / "roles/train-attempt"
    role.mkdir(parents=True)
    private_runtime_environment(role)
    (role / ".runtime/cache/secret.json").write_text("cache is not evidence")
    (role / "launch.json").write_text('{"storage":{"output_free_bytes":123}}')
    (role / "stderr.log").write_text("LLVM ERROR")
    host.publish(job, published)
    assert (published / "roles/train-attempt/launch.json").read_bytes() == (
        role / "launch.json"
    ).read_bytes()
    assert (published / "roles/train-attempt/stderr.log").read_text() == "LLVM ERROR"
    assert not list(published.rglob(".runtime"))
