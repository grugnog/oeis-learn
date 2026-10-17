"""Host-only bundle recovery must not lose gate evidence under trace pressure."""

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

from oeis_learn.fresh.bundle import pack
from oeis_learn.fresh.common import digest, file_hash, inventory, write


def run_fixture(root):
    write(root / "run.json", {"identity": digest("bundle-test"), "controls": {"max_updates": 3000}})
    selected = "roles/train-000030/checkpoints/qualified-00000001.json"
    write(root / "status.json", {"status": "complete", "checkpoint": selected})
    write(root / "finalization.json", {"checkpoint": selected})
    write(root / selected, {"base": "qualified-00000001-base.json"})
    blob = root / selected.replace(".json", ".pt")
    write(blob, b"weights must be opt-in")
    write(
        blob.with_name("qualified-00000001-base.json"),
        {"blob_path": blob.name, "blob_sha256": file_hash(blob)},
    )
    return selected


def test_trace_pressure_preserves_root_and_selected_evidence(tmp_path):
    root, output = tmp_path / "run", tmp_path / "recovered.zip"
    selected = run_fixture(root)
    trace = {"tokens": "1" * (900 << 10), "splits": {"training": {"exact": 30}}}
    write(root / "learnability.json", {"learnability_passed": False, "diagnostic": trace})
    write(root / "learnability-examples.json", {"training": [{"tokens": [1, 2, 3]}]})
    write(
        root / "status.json", {"status": "complete", "checkpoint": selected, "final_score": trace}
    )
    for i in range(31):
        write(root / f"roles/train-{i:06}/result.json", {"diagnostic": trace})
    write(root / "roles/train-000030/diagnostic.json", trace)
    # Metrics are aggregated even if the corresponding raw file is omitted.
    write(root / "roles/train-000000/metrics.jsonl", b'{"loss":2,"nonpadding_tokens":8}\n')
    before = inventory(root)
    result = pack(root, output)
    assert result["omissions"]
    with zipfile.ZipFile(output) as archive:
        receipt = json.loads(archive.read("bundle.json"))
        required = {
            "run.json",
            "status.json",
            "finalization.json",
            "learnability.json",
            "learnability-examples.json",
            selected,
            selected.replace(".json", "-base.json"),
            "roles/train-000030/result.json",
            "roles/train-000030/diagnostic.json",
        }
        assert required <= set(archive.namelist())
        assert not required & {row["path"] for row in receipt["omissions"]}
        for name in required:
            assert archive.read(name) == (root / name).read_bytes()
        assert sum(info.file_size for info in archive.infolist()) <= 25 << 20
        for name, ref in receipt["files"].items():
            payload = archive.read(name)
            assert len(payload) == ref["bytes"]
            assert "sha256:" + hashlib.sha256(payload).hexdigest() == ref["sha256"]
        assert json.loads(archive.read("analysis.json"))["nonpadding_tokens"] == 8
        assert not any(name.endswith(".pt") for name in archive.namelist())
    assert inventory(root) == before


def test_required_evidence_over_cap_fails_without_replacing_archive(tmp_path):
    root, output = tmp_path / "run", tmp_path / "analysis.zip"
    run_fixture(root)
    write(root / "learnability.json", {"diagnostic": "x" * 5000})
    output.write_bytes(b"previous archive")
    with pytest.raises(ValueError, match="cannot retain required evidence"):
        pack(root, output, max_bytes=4096)
    assert output.read_bytes() == b"previous archive"


def test_generated_inventory_counts_towards_cap(tmp_path):
    root, output = tmp_path / "run", tmp_path / "analysis.zip"
    run_fixture(root)
    for i in range(30):
        write(root / f"roles/history-{i:03}/diagnostic.json", {"tokens": "x" * 150})
    pack(root, output, max_bytes=9000)
    with zipfile.ZipFile(output) as archive:
        assert sum(info.file_size for info in archive.infolist()) <= 9000
        receipt = json.loads(archive.read("bundle.json"))
        assert any(row["reason"] == "bundle byte cap" for row in receipt["omissions"])
        assert "run.json" in receipt["files"] and "status.json" in receipt["files"]


def test_weight_bundle_is_opt_in_and_verifies_selected_blob(tmp_path):
    root, output = tmp_path / "run", tmp_path / "analysis.zip"
    selected = run_fixture(root)
    pack(root, output, checkpoint=True)
    blob = selected.replace(".json", ".pt")
    with zipfile.ZipFile(output) as archive:
        assert archive.read(blob) == b"weights must be opt-in"
    write(root / blob, b"corrupted")
    with pytest.raises(ValueError, match="blob mismatch"):
        pack(root, output, checkpoint=True)


def test_pack_cli_needs_neither_engine_nor_gpu():
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="tmp-bundle-cli-", dir=repo) as temp:
        folder = Path(temp)
        run_fixture(folder / "run")
        result = subprocess.run(
            [
                sys.executable,
                str(repo / "scripts/foundation_run.py"),
                "--engine",
                "docker",
                "pack",
                "--run",
                str((folder / "run").relative_to(repo)),
                "--output",
                str((folder / "analysis.zip").relative_to(repo)),
            ],
            check=False,
            cwd=folder,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (folder / "analysis.zip").is_file()


def test_selected_manifest_cannot_export_private_view(tmp_path):
    root, output = tmp_path / "run", tmp_path / "analysis.zip"
    run_fixture(root)
    secret = "roles/train-input/checkpoints/qualified.json"
    write(root / "status.json", {"status": "interrupted", "checkpoint": secret})
    (root / "finalization.json").unlink()
    write(root / secret, {"base": "truth.json"})
    write(root / "roles/train-input/checkpoints/truth.json", {"secret": "must not export"})
    pack(root, output)
    with zipfile.ZipFile(output) as archive:
        assert not any("train-input" in name or "truth.json" in name for name in archive.namelist())
