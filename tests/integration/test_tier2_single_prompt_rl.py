"""Integration test for Tier 2 Single-Prompt RL Policy Convergence."""

from oeis_learn.rl.progressive import validate_tier_2


def test_tier2_single_prompt_rl_convergence():
    res = validate_tier_2()
    assert res.tier == 2
    assert res.tier_name == "TIER_2_SINGLE_PROMPT_RL"
    assert res.passed is True
    assert res.latency_seconds < 600.0
    assert "final_pass_rate" in res.metrics
    assert "final_acr" in res.metrics
    assert res.metrics["final_acr"] <= 0.50
