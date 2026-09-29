"""Contract test for Progressive Test Harness Reporting Schema."""

import json
from oeis_learn.data.models import ProgressiveTierResult, ProgressiveValidationReport
from oeis_learn.rl.progressive import validate_tier_0


def test_progressive_harness_contract_and_schema(tmp_path):
    tier0_res = validate_tier_0()
    assert tier0_res.tier == 0
    assert tier0_res.tier_name == "TIER_0_STATIC_UNIT"
    assert tier0_res.latency_seconds > 0.0
    assert tier0_res.passed is True
    assert "trap_latency_ms" in tier0_res.metrics

    report = ProgressiveValidationReport(
        harness_version="2.0.0",
        overall_passed=True,
        max_authorized_tier=0,
        tier_results=[tier0_res],
    )

    report_dict = report.to_dict()
    assert report_dict["harness_version"] == "2.0.0"
    assert report_dict["overall_passed"] is True
    assert report_dict["max_authorized_tier"] == 0
    assert len(report_dict["tier_results"]) == 1

    # Verify JSON serializability
    out_file = tmp_path / "prog_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["harness_version"] == "2.0.0"
    assert loaded["tier_results"][0]["tier_name"] == "TIER_0_STATIC_UNIT"
