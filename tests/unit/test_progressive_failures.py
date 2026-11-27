import json
import pytest
from oeis_learn.pipeline_validation.progressive_evidence import publish, failure


@pytest.mark.parametrize(
    "state,science",
    [("COMPLETE", "NO_MATERIAL_GAIN"), ("PARTIAL", "INCOMPLETE"), ("FAILED", "INVALID")],
)
def test_publication_never_upgrades_science(tmp_path, state, science):
    r = publish(
        tmp_path,
        operation="fixture",
        mechanical_status=state,
        scientific_status=science,
        reason="expected",
        missing=["device"],
    )
    assert r["scientific_status"] == science and r["qualification_status"] == "PENDING"
    with pytest.raises(ValueError):
        publish(tmp_path, operation="fixture", mechanical_status="COMPLETE", reason="overwrite")


def test_first_failure_and_collector_cause_retained_without_truth_leak(tmp_path):
    try:
        raise ValueError("native mismatch for private value 918273645")
    except ValueError as exc:
        r = failure(
            tmp_path,
            "audit-language",
            exc,
            producer="native-worker",
            private_values=["918273645"],
            collector_errors=[{"type": "OSError", "message": "disk full"}],
        )
    assert "918273645" not in json.dumps(r)
    assert "native mismatch" in r["producer_error"]["message"]
    assert r["producer_error"]["type"] == "ValueError"
    assert r["collector_errors"][0]["message"] == "disk full"
    assert "918273645" in (tmp_path / r["producer_error"]["local_traceback"]["path"]).read_text()
