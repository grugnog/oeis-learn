"""Available real input bytes cannot be promoted by a diagnostic driver's labels."""

import shutil
import pytest

from scripts import verify_phase13


@pytest.fixture
def input_checkout(tmp_path, monkeypatch):
    # The user's live checkout may contain actual historical weights. Never
    # assume they are absent or modify them to satisfy this negative test.
    checkout = tmp_path / "checkout"
    for name in (
        "configs/foundation/generic_sampler.yaml",
        "scripts/extract_oeis_data.py",
        "data/sample_oeis/stripped",
    ):
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verify_phase13.ROOT / name, target)
    historical = checkout / "runs/historical/config.yaml"
    historical.parent.mkdir(parents=True)
    historical.write_text("model:\n  d_model: 8\n")
    monkeypatch.setattr(verify_phase13, "ROOT", checkout)

    def diagnostic(strata, output, **kwargs):
        assert strata["random_generic"] and not strata["model_samples"]
        assert not strata["verified_solutions"] and kwargs["provenance_packets"]
        return {"report_sha256": verify_phase13.digest(strata)}

    monkeypatch.setattr(verify_phase13, "diagnose_overflow", diagnostic)
    return checkout


def test_cpu_input_driver_preserves_source_and_trained_artifact_blockers(tmp_path, input_checkout):
    report = verify_phase13.verify_inputs(tmp_path / "output", count=4)
    assert report["qualified"] is False
    assert report["tasks"]["T113"] == "pending_external_inputs"
    assert report["blockers"]["authoritative_source"] and report["blockers"]["trained_artifacts"]
    assert report["blockers"]["hardware"] and report["historical_configurations"]
    assert report["historical_configurations"][0]["parent_weights"] == []
    assert report["historical_configurations"][0]["status"] == "missing_trained_weights"


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".ckpt"])
def test_available_checkpoint_bytes_are_inventoried_without_qualifying_release(
    tmp_path, input_checkout, suffix
):
    weights = input_checkout / "runs/historical/checkpoints" / ("parent" + suffix)
    weights.parent.mkdir()
    weights.write_bytes(b"test inventory bytes, not a qualified trained model")
    before = weights.read_bytes()
    report = verify_phase13.verify_inputs(tmp_path / "output", count=4)
    assert report["qualified"] is False
    assert report["tasks"]["T113"] == "pending_external_inputs"
    assert report["blockers"]["trained_artifacts"] == []
    assert all(
        report["blockers"][name]
        for name in ("authoritative_source", "release_archive", "development", "hardware")
    )
    historical = report["historical_configurations"][0]
    assert historical["status"] == "available_for_parity"
    assert historical["parent_weights"] == [
        dict(
            path=weights.relative_to(input_checkout).as_posix(),
            sha256=verify_phase13.compute_file_hash(weights),
        )
    ]
    assert weights.read_bytes() == before


def test_unassociated_weights_and_noncheckpoint_files_do_not_remove_blocker(
    tmp_path, input_checkout
):
    (input_checkout / "runs/historical/notes.txt").write_text("historical notes")
    orphan = input_checkout / "runs/without-config"
    orphan.mkdir()
    (orphan / "parent.pt").write_bytes(b"unassociated inventory bytes")
    report = verify_phase13.verify_inputs(tmp_path / "output", count=4)
    assert report["blockers"]["trained_artifacts"]
    assert len(report["historical_configurations"]) == 1
    assert report["historical_configurations"][0]["parent_weights"] == []
