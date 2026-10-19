"""Persisted integration transitions reuse Phase 2 identity and budget rules."""

import pytest
from oeis_learn.fresh.bigint_config import resolve
from oeis_learn.fresh.bigint_state import DurableRun, CheckpointStore
from oeis_learn.fresh.bigint_inputs import visible_record
from oeis_learn.fresh.bigint_bundle import pack


def test_visible_view_never_exports_continuations_or_rebases_indices():
    row = dict(
        id="sample",
        family="family",
        stage="A",
        indices=list(map(str, range(-7, 93))),
        values=[str(n) for n in range(20)] + ["987654321"] * 80,
        source="(program (return $index))",
        parameter_source="(program (return $index))",
    )
    record = visible_record(row, "concrete")
    assert record["indices"][0] == "-7" and len(record["values"]) == 20
    assert "987654321" not in str(record)
    assert len(record["horizon_indices"]) == 100


def test_debit_checkpoint_resume_and_analysis_do_not_refund_or_change_parent(tmp_path):
    cfg = resolve(dict(version="bigint-bootstrap/v1"))
    manifest = dict(identity="sha256:" + "1" * 64, split_sha256="sha256:" + "2" * 64)
    run = DurableRun.create(
        tmp_path / "run", cfg, manifest, image="sha256:" + "3" * 64, device="cpu"
    )
    run.start("first", 10)
    run.finish("first", "interrupted")
    before = run.ledger.remaining
    resumed = DurableRun.resume(tmp_path / "run", cfg, image="sha256:" + "3" * 64)
    assert resumed.ledger.remaining == before
    with pytest.raises(ValueError, match="image"):
        DurableRun.resume(tmp_path / "run", cfg, image="sha256:" + "4" * 64)
    states = {
        k: dict(value=k)
        for k in ("weights", "optimizer", "rng", "data_order", "selection", "curriculum", "replay")
    }
    store = CheckpointStore(tmp_path / "run/checkpoints", run.contract)
    saved = store.save(states, run.ledger, state="interrupted")
    loaded = store.load(saved)
    assert loaded["states"] == states and loaded["ledger"].remaining == before
    blob = next((tmp_path / "run/checkpoints").glob("*/weights.pt"))
    blob.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact"):
        store.load(saved)


def test_core_export_precedes_details_and_excludes_private_inputs(tmp_path):
    cfg = resolve(dict(version="bigint-bootstrap/v1"))
    run = DurableRun.create(
        tmp_path / "run",
        cfg,
        dict(identity="sha256:" + "1" * 64, split_sha256="sha256:" + "2" * 64),
        image="sha256:" + "3" * 64,
        device="cpu",
    )
    from oeis_learn.fresh.common import write

    for name in ("summary.json", "checkpoint-identity.json", "comparison.json"):
        write(run.root / name, dict(status="fixture", final_access="sealed"))
    (run.root / "private").mkdir()
    (run.root / "private/truth.json").write_text("NEVER EXPORT")
    with pytest.raises(ValueError, match="16"):
        pack(run.root, tmp_path / "too-small.zip", max_bytes=1 << 20)
    result = pack(run.root, tmp_path / "report.zip", max_bytes=16 << 20)
    import zipfile

    with zipfile.ZipFile(tmp_path / "report.zip") as archive:
        assert "summary.json" in archive.namelist() and "bundle-manifest.json" in archive.namelist()
        assert all("private" not in name for name in archive.namelist())
    assert result["complete_core"]
