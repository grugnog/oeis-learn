"""Available real input bytes cannot be promoted by a diagnostic driver's labels."""

from scripts import verify_phase13


def test_cpu_input_driver_preserves_source_and_trained_artifact_blockers(tmp_path, monkeypatch):
    def diagnostic(strata, output, **kwargs):
        assert strata["random_generic"] and not strata["model_samples"]
        assert not strata["verified_solutions"] and kwargs["provenance_packets"]
        return {"report_sha256": verify_phase13.digest(strata)}

    monkeypatch.setattr(verify_phase13, "diagnose_overflow", diagnostic)
    report = verify_phase13.verify_inputs(tmp_path, count=4)
    assert report["qualified"] is False
    assert report["tasks"]["T113"] == "pending_external_inputs"
    assert report["blockers"]["authoritative_source"] and report["blockers"]["trained_artifacts"]
    assert report["blockers"]["hardware"] and report["historical_configurations"]
