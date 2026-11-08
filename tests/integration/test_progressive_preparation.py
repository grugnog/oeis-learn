"""Real projected custodian operation, with no model or outcomes in census."""

from tests.unit.test_progressive_cohorts import fixture
from tests.contract.test_progressive_roles import artifact
from oeis_learn.generalization.progressive.projections import project
from oeis_learn.pipeline_validation.progressive_roles import REGISTRY
from oeis_learn.pipeline_validation.progressive_role import execute


def test_projected_installed_census_writes_deficits_without_scoring(tmp_path):
    c, e = fixture(tmp_path)
    cohort = artifact(tmp_path, "cohort-input.json", "cohort", "custodian", c)
    exposure = artifact(tmp_path, "exposure-input.json", "exposure", "custodian", e)
    project(tmp_path, "inputs", "census", {"sources": [cohort, exposure]}, registry=REGISTRY)
    output = tmp_path / "out"
    output.mkdir()
    result = execute("census", inputs=tmp_path / "inputs", output=output)
    assert result["mechanical_status"] == "PARTIAL" and result["scientific_status"] == "UNMEASURED"
    assert set(result["missing_evidence"]) == {
        "parameter",
        "composition",
        "unseen_structure",
        "oeis",
    }
    assert (output / "census.json").is_file()
