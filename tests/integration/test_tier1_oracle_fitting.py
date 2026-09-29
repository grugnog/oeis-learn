"""Integration test for Tier 1 Oracle Reference Solution Fitting."""

from oeis_learn.rl.progressive import validate_tier_1


def test_tier1_oracle_fitting_converges():
    res = validate_tier_1()
    assert res.tier == 1
    assert res.tier_name == "TIER_1_ORACLE_SFT"
    assert res.passed is True
    assert res.latency_seconds < 120.0
    assert "final_oracle_ppl" in res.metrics
    assert res.metrics["final_oracle_ppl"] < 1.25
