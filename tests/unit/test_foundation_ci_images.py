"""Exercise image provenance, registry permissions and engine-neutral reuse."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ci_images", ROOT / "scripts/ci/images.py")
images = importlib.util.module_from_spec(spec)
spec.loader.exec_module(images)
REVISION = "a" * 40
REFERENCE = "ghcr.io/owner/repo@sha256:" + "b" * 64


@pytest.fixture
def build():
    env = dict(
        CI_IMAGE_TARGET="qualification",
        GITHUB_REPOSITORY="Owner/Repo",
        GITHUB_RUN_ID="42",
        GITHUB_RUN_ATTEMPT="2",
        GITHUB_EVENT_NAME="pull_request",
        GITHUB_ACTOR="maintainer",
        GITHUB_REF="refs/pull/1/merge",
    )
    event = dict(
        repository=dict(default_branch="main"),
        pull_request=dict(
            head=dict(repo=dict(full_name="Owner/Repo")),
            base=dict(sha="c" * 40),
        ),
    )
    return env, event


def test_pr_images_are_unique_and_cannot_update_shared_cache(build):
    env, event = build
    value = images.plan(env, event)
    assert value["publish"] == "true" and value["transport"] == "registry"
    assert value["tag"] == "ghcr.io/owner/repo:ci-qualification-42-2"
    assert "type=registry" in value["cache_from"]
    assert "type=registry" not in value["cache_to"]
    assert "qualification-main" not in value["tags"]
    retry = images.plan(dict(env, GITHUB_RUN_ATTEMPT="3"), event)
    assert retry["tag"] != value["tag"]


@pytest.mark.parametrize("fork,actor", [(True, "contributor"), (False, "dependabot[bot]")])
def test_read_only_prs_share_archive_without_publishing(build, fork, actor):
    env, event = build
    event = deepcopy(event)
    if fork:
        event["pull_request"]["head"]["repo"]["full_name"] = "fork/Repo"
    value = images.plan(dict(env, GITHUB_ACTOR=actor), event)
    assert value["transport"] == "archive" and value["publish"] == "false"
    assert "type=registry" not in value["cache_to"]


@pytest.mark.parametrize("event_name", ["push", "workflow_dispatch"])
def test_only_default_branch_refreshes_registry_cache(build, event_name):
    env, event = build
    env = dict(env, GITHUB_EVENT_NAME=event_name, GITHUB_REF="refs/heads/main")
    main = images.plan(env, event)
    assert "type=registry" in main["cache_to"]
    assert main["tags"].splitlines() == [main["tag"], "ghcr.io/owner/repo:qualification-main"]
    env["GITHUB_REF"] = "refs/heads/feature"
    assert "type=registry" not in images.plan(env, event)["cache_to"]
    env["CI_IMAGE_TARGET"] = "gpu"
    assert ":buildcache-gpu-main" in images.plan(env, event)["cache_from"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("GITHUB_EVENT_NAME", "pull_request_target"),
        ("CI_IMAGE_TARGET", "unknown"),
        ("GITHUB_REPOSITORY", "owner/repo\nmalicious-output=true"),
        ("GITHUB_RUN_ID", "42\nmalicious-output=true"),
    ],
)
def test_invalid_build_identity_is_rejected(build, key, value):
    env, event = build
    with pytest.raises(ValueError):
        images.plan(dict(env, **{key: value}), event)


@pytest.fixture
def engine_calls(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=REVISION + "\n")

    monkeypatch.setattr(images.subprocess, "run", run)
    monkeypatch.setenv("GITHUB_ACTOR", "maintainer")
    return calls


@pytest.mark.parametrize("engine", ["docker", "podman"])
def test_pull_uses_digest_and_removes_credentials_before_tests(engine_calls, engine):
    images.load_image(engine, "registry", REFERENCE, "ci-local", REVISION, token="test-token")
    assert [args[1] for args, _ in engine_calls] == ["login", "pull", "logout", "image", "tag"]
    assert all(args[0] == engine for args, _ in engine_calls)
    assert engine_calls[1][0] == [engine, "pull", REFERENCE]
    assert engine_calls[0][1]["input"] == "test-token\n"
    assert all("test-token" not in arg for args, _ in engine_calls for arg in args)


def test_failed_pull_still_logs_out(engine_calls, monkeypatch):
    original = images.subprocess.run

    def fail(argv, **kwargs):
        if argv[1] == "pull":
            raise subprocess.CalledProcessError(1, argv)
        return original(argv, **kwargs)

    monkeypatch.setattr(images.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        images.load_image("podman", "registry", REFERENCE, "ci-local", REVISION, token="token")
    assert engine_calls[-1][0] == ["podman", "logout", "ghcr.io"]
    assert not any(args[1] == "tag" for args, _ in engine_calls)


@pytest.mark.parametrize("engine", ["docker", "podman"])
def test_fork_archive_load_never_logs_in(engine_calls, engine, tmp_path):
    archive = tmp_path / "image.tar"
    archive.touch()
    images.load_image(
        engine,
        "archive",
        "ghcr.io/owner/repo:ci-42",
        "ci-local",
        REVISION,
        archive=archive,
        token="unused-token",
    )
    assert [args[1] for args, _ in engine_calls] == ["load", "image", "tag"]
    assert engine_calls[0][0] == [engine, "load", "--input", str(archive)]


def test_stale_image_cannot_be_retagged(engine_calls):
    with pytest.raises(ValueError, match="source revision"):
        images.load_image("docker", "registry", REFERENCE, "ci-local", "f" * 40)
    assert not any(args[1] == "tag" for args, _ in engine_calls)


@pytest.mark.parametrize("reference", ["ghcr.io/owner/repo:latest", "--help", "bad@sha256:x"])
def test_registry_reuse_requires_immutable_reference(engine_calls, reference):
    with pytest.raises(ValueError):
        images.load_image("docker", "registry", reference, "ci-local", REVISION)
    assert not engine_calls


def test_missing_archive_fails_before_engine_launch(engine_calls, tmp_path):
    with pytest.raises(ValueError, match="archive is missing"):
        images.load_image(
            "podman",
            "archive",
            "ghcr.io/owner/repo:ci-42",
            "ci-local",
            REVISION,
            archive=tmp_path / "missing.tar",
        )
    assert not engine_calls


def test_recipe_diff_includes_build_inputs_and_fails_closed(monkeypatch):
    calls = []
    code = 0

    def run(argv):
        calls.append(argv)
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(images.subprocess, "run", run)
    assert images.recipe_changed("base", "head") is False
    code = 1
    assert images.recipe_changed("base", "head") is True
    assert calls[0][:6] == ["git", "diff", "--quiet", "base", "head", "--"]
    assert {
        "Dockerfile",
        ".dockerignore",
        "uv.lock",
        "docker/dev",
        "crates",
    } <= set(calls[0])
    code = 128
    with pytest.raises(RuntimeError, match="build changes"):
        images.recipe_changed("missing-base", "head")


def test_real_git_diff_distinguishes_code_from_recipe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    git("init", "--quiet")
    git("config", "user.name", "CI test")
    git("config", "user.email", "ci-test@example.invalid")
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "source.py").write_text("pass\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "code")
    assert images.recipe_changed(base, "HEAD") is False
    (tmp_path / "Dockerfile").write_text("FROM scratch\nLABEL test=true\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "recipe")
    assert images.recipe_changed(base, "HEAD") is True
