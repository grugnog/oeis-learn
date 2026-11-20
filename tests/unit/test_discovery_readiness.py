import json
import zipfile
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, reference
from oeis_learn.generalization.progressive.readiness import build, from_request
from oeis_learn.pipeline_validation.identity import seal
from tests.unit.test_progressive_cohorts import fixture


def test_missing_evidence_reports_all_four_categories_without_scoring(tmp_path):
    c, e = fixture(tmp_path, counts=(1, 1, 0, 1))
    r = build(tmp_path, c, e, [], [])
    assert r["status"] == "INCOMPLETE"
    assert set(r["strata"]) == {"parameter", "composition", "unseen_structure", "oeis"}
    assert r["strata"]["unseen_structure"]["availability"]["tasks"] == 0
    assert "retained_D_evaluation" in r["missing"]
    assert r["historical_exposure"] == e["identity"]
    assert r["scientific_status"] == "UNMEASURED"


def test_command_report_reconciles_and_exports_only_public_summary(tmp_path):
    from oeis_learn.pipeline_validation.progressive_commands import main

    c, e = fixture(tmp_path)
    atomic_write(tmp_path / "cohort.json", c)
    atomic_write(tmp_path / "exposure.json", e)
    request = seal(
        {
            "version": "capability-readiness-request/v1",
            "cohort": reference(tmp_path, "cohort.json"),
            "exposure": reference(tmp_path, "exposure.json"),
            "coverage": [],
            "evaluations": [],
        }
    )
    atomic_write(tmp_path / "run/capability-inputs.json", request)
    args = SimpleNamespace(operation="report", run="run", output="summary.zip")
    assert main(args, tmp_path, None) == 2
    with zipfile.ZipFile(tmp_path / "summary.zip") as archive:
        assert set(archive.namelist()) == {"readiness.json", "result.json"}
        report = json.loads(archive.read("readiness.json"))
        assert report["status"] == "INCOMPLETE"
        assert b"values" not in archive.read("readiness.json")
    request["evaluations"] = [{}]
    with pytest.raises(ValueError):
        from_request(tmp_path, request)


def test_forged_score_or_missing_comparison_is_rejected(tmp_path):
    c, e = fixture(tmp_path)
    atomic_write(
        tmp_path / "candidate.json", seal({"version": "discovery-candidates/v1", "records": {}})
    )
    atomic_write(
        tmp_path / "score.json",
        seal(
            {"version": "discovery-scoring/v1", "candidates": "sha256:" + "0" * 64, "outcomes": {}}
        ),
    )
    with pytest.raises(ValueError, match="binding"):
        build(
            tmp_path,
            c,
            e,
            [],
            [
                {
                    "candidates": reference(tmp_path, "candidate.json"),
                    "scores": reference(tmp_path, "score.json"),
                }
            ],
        )
