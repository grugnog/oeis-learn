import hashlib
import resource
import shutil
import sqlite3
import subprocess
from unittest.mock import patch

import pytest

from oeis_learn.fresh import bulk
from oeis_learn.fresh.common import file_hash, read, write


def entry(aid="A000001", offset=-2, keywords="sign,easy"):
    return f"%I {aid}\n%S {aid} 0,1,2\n%O {aid} {offset},3\n%K {aid} {keywords}\n"


@pytest.mark.parametrize("offset", [-10, 0, 1])
def test_authoritative_metadata_and_explicit_offset(offset):
    meta = bulk.metadata(entry(offset=offset), "A000001", "pinned-source")
    assert meta["offset"] == offset
    assert meta["keywords"] == ["sign", "easy"]
    assert not meta["finite"] and not meta["table"]
    assert bulk.metadata(entry(keywords="fini,tabl"), "A000001", "source")["finite"]


@pytest.mark.parametrize(
    "payload",
    [
        "%O A000001 0,1\n",
        entry().replace("-2,3", "zero,3"),
        entry() + "%O A000001 0,1\n",
        entry().replace("%S A000001", "%S A000002"),
    ],
)
def test_missing_conflicting_foreign_metadata_rejects(payload):
    with pytest.raises(ValueError):
        bulk.metadata(payload, "A000001", "source")


def test_indexed_horizon_preserves_negative_offsets_signs_and_gaps(tmp_path):
    path = tmp_path / "b.txt"
    path.write_text("# comment\n-2 -7\n-1 +0\n1 9\n200 7\n")
    assert bulk.indexed_terms(path, -2) == {-2: "-7", -1: "0", 1: "9"}
    for content in (
        "0 1\n0 1\n",
        "1 1\n0 0\n",
        "0 1.5\n",
        "version https://git-lfs.github.com/spec/v1\n",
    ):
        path.write_text(content)
        with pytest.raises(ValueError):
            bulk.indexed_terms(path, 0)


def test_import_replays_hashes_counts_and_retains_gaps(tmp_path):
    cache, output = tmp_path / "cache", tmp_path / "import"
    repo = cache / "oeisdata"
    rows = []
    for n, count in ((1, 120), (2, 50)):
        aid = f"A{n:06d}"
        seq = repo / "seq/A000" / (aid + ".seq")
        terms = repo / "files/A000" / f"b{n:06d}.txt"
        seq.parent.mkdir(parents=True, exist_ok=True)
        terms.parent.mkdir(parents=True, exist_ok=True)
        seq.write_text(entry(aid, -2))
        terms.write_text("".join(f"{i - 2} {-i}\n" for i in range(count)))
        rows.append(
            {
                "aid": aid,
                "path": str(terms.relative_to(repo)),
                "sha256": file_hash(terms),
                "bytes": terms.stat().st_size,
                "entry_sha256": file_hash(seq),
                "metadata": bulk.metadata(seq.read_text(), aid, "pinned"),
            }
        )
    plan = {"selected": rows, "snapshot": {"commit": "a" * 40}}
    write(cache / "fetch-plan.json", plan)
    write(cache / "fetch-result.json", {"plan_sha256": file_hash(cache / "fetch-plan.json")})
    report = bulk.import_snapshot(cache, output)
    assert report["records"] == 1
    assert report["horizons"]["complete_50"] == 2
    assert report["horizons"]["complete_100"] == 1
    assert report["exclusions"] == {"incomplete100": 1}
    manifest = (output / "manifest.json").read_bytes()
    bulk.import_snapshot(cache, output)
    assert (output / "manifest.json").read_bytes() == manifest
    (repo / rows[0]["path"]).write_text("0 1\n")
    with pytest.raises(ValueError, match="changed bulk"):
        bulk.import_snapshot(cache, output)


def test_fetch_is_batched_pinned_and_rerun_uses_verified_objects(tmp_path):
    repo = tmp_path / "oeisdata"
    seq = repo / "seq/A000/A000001.seq"
    path = repo / "files/A000/b000001.txt"
    seq.parent.mkdir(parents=True)
    path.parent.mkdir(parents=True)
    seq.write_text(entry())
    payload = b"-2 -1\n-1 0\n0 1\n"
    import hashlib

    sha = hashlib.sha256(payload).hexdigest()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {len(payload)}\n"
    path.write_text(pointer)
    (repo / "time.txt").write_text("2026-10-11")
    write(
        tmp_path / "snapshot.json",
        {"repository": bulk.REPOSITORY, "commit": "a" * 40, "exported_at": "2026-10-11"},
    )
    calls = []

    def fake_git(root, *args, capture=False):
        calls.append(args)
        if args[:2] == ("remote", "get-url"):
            return bulk.REPOSITORY + "\n"
        if args[:2] == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        if args[0] == "ls-tree":
            return (
                f"100644 blob {bulk.blob_hash(seq.read_bytes())}\tseq/A000/A000001.seq\0"
                + f"100644 blob {bulk.blob_hash(pointer.encode())}\tfiles/A000/b000001.txt\0"
            )
        if "fetch" in args:
            assert "--include=files/A000/b000001.txt" in args
            assert args[-1] == "a" * 40 and "--all" not in args
        if args[:2] == ("lfs", "checkout"):
            path.write_bytes(payload)
        return ""

    config = {
        "revision": None,
        "seed": 1,
        "max_records": 1,
        "max_download_bytes": 1000,
        "free_disk_floor_bytes": 0,
    }
    with patch.object(bulk, "git", fake_git):
        bulk.fetch(config, tmp_path)
        count = len([c for c in calls if "fetch" in c])
        bulk.fetch(config, tmp_path)
        assert len([c for c in calls if "fetch" in c]) == count == 1
        with pytest.raises(ValueError, match="already pinned"):
            bulk.fetch(dict(config, revision="b" * 40), tmp_path)
    assert read(tmp_path / "fetch-result.json")["complete"]


@pytest.mark.parametrize("custom_hooks_path", [False, True])
def test_fetch_preserves_existing_hooks_and_resumes_with_real_lfs(
    tmp_path, monkeypatch, custom_hooks_path
):
    if shutil.which("git-lfs") is None:
        pytest.skip("requires git-lfs, installed in both project images")
    # Use actual Git/LFS configuration and hooks without inheriting host config.
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    repo = tmp_path / "oeisdata"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    bulk.git(repo, "remote", "add", "origin", bulk.REPOSITORY)
    bulk.git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "--allow-empty",
        "-m",
        "fixture",
    )
    commit = bulk.git(repo, "rev-parse", "HEAD", capture=True).strip()
    hooks = repo / ("kept-hooks" if custom_hooks_path else ".git/hooks")
    hooks.mkdir(exist_ok=True)
    if custom_hooks_path:
        bulk.git(repo, "config", "--local", "core.hooksPath", "kept-hooks")
        content = b"#!/bin/sh\n# Existing custom hook must not be overwritten.\nexit 0\n"
    else:
        # The older LFS hook reported by the HIP run is deliberately retained.
        content = (
            b"#!/bin/sh\ncommand -v git-lfs >/dev/null 2>&1 || { "
            b'echo "git-lfs not found" >&2; exit 2; }\ngit lfs pre-push "$@"\n'
        )
    hook = hooks / "pre-push"
    hook.write_bytes(content)
    hook.chmod(0o755)
    payload = repo / "files/A000/b000001.txt"
    payload.parent.mkdir(parents=True)
    payload.write_text("0 1\n1 2\n")
    write(tmp_path / "snapshot.json", {"repository": bulk.REPOSITORY, "commit": commit})
    plan = {
        "selected": [
            {
                "path": str(payload.relative_to(repo)),
                "bytes": payload.stat().st_size,
                "sha256": file_hash(payload),
            }
        ]
    }
    write(tmp_path / "fetch-plan.json", plan, immutable=True)
    config = {"revision": commit, "free_disk_floor_bytes": 0}
    plan_hash = file_hash(tmp_path / "fetch-plan.json")
    for _ in range(2):
        assert bulk.fetch(config, tmp_path) == plan
        assert hook.read_bytes() == content and hook.stat().st_mode & 0o777 == 0o755
        assert file_hash(tmp_path / "fetch-plan.json") == plan_hash
        assert read(tmp_path / "fetch-result.json") == {
            "plan_sha256": plan_hash,
            "complete": True,
            "files": 1,
        }
    assert (
        bulk.git(repo, "config", "--local", "--get", "filter.lfs.process", capture=True).strip()
        == "git-lfs filter-process"
    )


@pytest.fixture
def local_lfs_cache(tmp_path, monkeypatch):
    if shutil.which("git-lfs") is None:
        pytest.skip("requires git-lfs, installed in both project images")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_LFS_SKIP_SMUDGE", "1")
    source, remote, cache = tmp_path / "source", tmp_path / "remote.git", tmp_path / "cache"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    bulk.git(source, "lfs", "install", "--local", "--skip-repo")
    (source / ".gitattributes").write_text("files/** filter=lfs diff=lfs merge=lfs -text\n")
    payloads = {
        "files/A000/b000001.txt": b"0 1\n1 2\n",
        "files/A373/b373464.txt": b"0 7\n1 9\n",
        "files/A999/b999999.txt": b"0 99\n1 99\n",
    }
    for name, data in payloads.items():
        path = source / name
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
    bulk.git(source, "add", ".")
    bulk.git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    bulk.git(source, "remote", "add", "origin", str(remote))
    bulk.git(source, "push", "-q", "origin", "HEAD")
    bulk.git(source, "lfs", "push", "--all", "origin")
    cache.mkdir()
    repo = cache / "oeisdata"
    subprocess.run(["git", "clone", "-q", str(remote), str(repo)], check=True)
    commit = bulk.git(repo, "rev-parse", "HEAD", capture=True).strip()
    monkeypatch.setattr(bulk, "REPOSITORY", str(remote))
    write(cache / "snapshot.json", {"repository": str(remote), "commit": commit})
    rows = [
        {"path": name, "bytes": len(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}
        for name, data in list(payloads.items())[:2]
    ]
    write(cache / "fetch-plan.json", {"selected": rows}, immutable=True)
    # One object was already hydrated by an earlier partial attempt.
    (repo / rows[0]["path"]).write_bytes(payloads[rows[0]["path"]])
    return cache, repo, commit, rows, payloads


@pytest.mark.parametrize(
    "failure", [None, "missing_remote_object", "changed_worktree", "stale_plan"]
)
def test_fetch_recovers_unresolved_pointer_via_real_lfs_and_preserves_cache(
    local_lfs_cache, monkeypatch, failure
):
    cache, repo, commit, rows, payloads = local_lfs_cache
    row = rows[1]
    path = repo / row["path"]
    if failure == "missing_remote_object":
        oid = row["sha256"].removeprefix("sha256:")
        remote = cache.parent / "remote.git"
        (remote / "lfs/objects" / oid[:2] / oid[2:4] / oid).unlink()
    elif failure == "changed_worktree":
        path.write_text("local modification\n")
    elif failure == "stale_plan":
        rows[1] = dict(row, bytes=row["bytes"] + 1)
        write(cache / "fetch-plan.json", {"selected": rows})
    before = path.read_bytes()
    retained = {
        p: file_hash(p)
        for p in (cache / "snapshot.json", cache / "fetch-plan.json", repo / rows[0]["path"])
    }
    original_git, fetches = bulk.git, []

    def incomplete_batch(root, *args, **kwargs):
        if args[:4] == ("-c", "lfs.fetchrecentalways=false", "lfs", "fetch"):
            fetches.append(args)
            return ""  # Observed success-with-missing-object symptom; no network.
        return original_git(root, *args, **kwargs)

    monkeypatch.setattr(bulk, "git", incomplete_batch)
    config = {"revision": commit, "free_disk_floor_bytes": 0}
    if failure:
        with pytest.raises(ValueError, match="LFS .*failed|differs from pinned"):
            bulk.fetch(config, cache)
        assert path.read_bytes() == before
        assert not (cache / "fetch-result.json").exists()
    else:
        bulk.fetch(config, cache)
        assert path.read_bytes() == payloads[row["path"]]
        assert read(cache / "fetch-result.json")["complete"]
        bulk.fetch(config, cache)
        assert len(fetches) == 1
    assert all(file_hash(p) == sha for p, sha in retained.items())
    assert bulk.pointer(repo / "files/A999/b999999.txt") is not None
    assert not list(repo.glob("files/*/.lfs-recovery-*"))


def import_fixture(cache):
    repo = cache / "oeisdata"
    seq = repo / "seq/A000/A000001.seq"
    terms = repo / "files/A000/b000001.txt"
    seq.parent.mkdir(parents=True)
    terms.parent.mkdir(parents=True)
    seq.write_text(entry())
    terms.write_text("".join(f"{i} {i}\n" for i in range(-2, 118)))
    write(
        cache / "fetch-plan.json",
        {
            "snapshot": {"commit": "a" * 40},
            "selected": [
                {
                    "aid": "A000001",
                    "path": "files/A000/b000001.txt",
                    "sha256": file_hash(terms),
                    "entry_sha256": file_hash(seq),
                    "metadata": bulk.metadata(seq.read_text(), "A000001", "fixture"),
                }
            ],
        },
    )
    write(cache / "fetch-result.json", {"plan_sha256": file_hash(cache / "fetch-plan.json")})
    return terms


@pytest.mark.parametrize("failure", [None, "changed_source", "empty_source"])
def test_import_closes_sqlite_on_success_and_failure(tmp_path, failure):
    cache, output = tmp_path / "cache", tmp_path / "import"
    terms = import_fixture(cache)
    if failure == "changed_source":
        terms.write_text("tampered")
    elif failure == "empty_source":
        plan = read(cache / "fetch-plan.json")
        plan["selected"] = []
        write(cache / "fetch-plan.json", plan)
        write(cache / "fetch-result.json", {"plan_sha256": file_hash(cache / "fetch-plan.json")})
    connect, connections = sqlite3.connect, []

    def retain_connection(*args, **kwargs):
        db = connect(*args, **kwargs)
        connections.append(db)  # Prevent GC from hiding missing explicit closure.
        return db

    with patch.object(bulk.sqlite3, "connect", retain_connection):
        if failure:
            with pytest.raises(ValueError):
                bulk.import_snapshot(cache, output)
        else:
            bulk.import_snapshot(cache, output)
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_repeated_import_under_low_open_file_limit(tmp_path):
    cache, output = tmp_path / "cache", tmp_path / "import"
    import_fixture(cache)
    connect, connections = sqlite3.connect, []

    def retain_connection(*args, **kwargs):
        db = connect(*args, **kwargs)
        connections.append(db)
        return db

    previous = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = min(64, previous[0])
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, previous[1]))
        with patch.object(bulk.sqlite3, "connect", retain_connection):
            for _ in range(128):
                assert bulk.import_snapshot(cache, output)["records"] == 1
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, previous)
        for db in connections:
            db.close()
