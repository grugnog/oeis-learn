"""Integration test for Tier 3 Synthetic Micro-Cohort Curriculum Progression."""

from oeis_learn.rl.progressive import validate_tier_3


def test_tier3_micro_cohort_progression():
    res = validate_tier_3()
    assert res.tier == 3
    assert res.tier_name == "TIER_3_MICRO_COHORT"
    assert res.passed is True
    assert res.latency_seconds < 2700.0
    assert "micro_cohort_competence" in res.metrics
    assert "final_acr" in res.metrics
    assert res.metrics["final_acr"] <= 0.40
