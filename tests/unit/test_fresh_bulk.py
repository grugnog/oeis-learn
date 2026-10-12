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
        terms.write_text("".join(f"{i-2} {-i}\n" for i in range(count)))
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
