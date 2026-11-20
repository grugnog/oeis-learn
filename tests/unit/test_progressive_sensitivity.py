"""Synthetic operating characteristics cannot use confirmation observations."""

import pytest


def scenario(**patch):
    return {
        "effect_ppm": 20000,
        "censor_ppm": 0,
        "family_dependence_ppm": 0,
        "family_sizes": [4, 16],
        "seed_disagreement": False,
        "comparisons": 1,
        "claim": "capability",
        "groups": 16,
    } | patch


def test_default_grid_contains_required_effects_censoring_and_optional_looks():
    from oeis_learn.generalization.progressive.sensitivity import default_design

    d = default_design(seed=13, repeats=1)
    assert {s["effect_ppm"] for s in d["scenarios"]} == {0, 20000, 50000}
    assert {s["censor_ppm"] for s in d["scenarios"]} == {0, 20000, 50000, 100000}
    assert {s["claim"] for s in d["scenarios"]} == {"capability", "efficiency"}
    assert {s["comparisons"] for s in d["scenarios"]} == {1, 4}
    assert any(s["seed_disagreement"] for s in d["scenarios"])


def test_synthetic_records_have_prospective_membership_and_paired_controls():
    from oeis_learn.generalization.progressive.sensitivity import simulate

    rows, roster, seeds = simulate(scenario(effect_ppm=0), seed=13)
    outcomes = {(r["seed"], r["arm"], r["task"]): r["outcome"] for r in rows}
    assert all(
        outcomes[s, "candidate", r["task"]] == outcomes[s, "baseline", r["task"]]
        for s in seeds
        for r in roster
    )
    assert len(roster) == 2 * 16 * 10 + 20
    assert all(r["task"].startswith("synthetic/") for r in roster)
    assert (rows, roster, seeds) == simulate(scenario(effect_ppm=0), seed=13)


def test_report_exposes_all_rates_assumptions_and_no_research_authority():
    from oeis_learn.generalization.progressive.sensitivity import run

    d = {"seed": 13, "repeats": 1, "scenarios": [scenario(effect_ppm=0)]}
    r = run(d)
    s = r["scenarios"][0]
    assert r["kind"] == "DIAGNOSTIC_ONLY"
    assert r["assumptions"] and r["no_confirmation_inputs"] is True
    assert s["false_advancement_rate"] == 0
    assert s["missed_gain_rate"] is None
    assert s["inconclusive_rate"] == 0
    assert sum(s["verdict_counts"].values()) == 1
    assert r == run(d)


def test_reject_real_outcomes_and_posthoc_thresholds():
    from oeis_learn.generalization.progressive.sensitivity import run, simulate

    d = {"seed": 13, "repeats": 1, "scenarios": [scenario()]}
    for extra in ({"outcomes": []}, {"confirmation_scores": []}, {"gain_threshold": 0.0}):
        with pytest.raises(ValueError):
            run(d | extra)
    for patch in (
        {"effect_ppm": 30000},
        {"censor_ppm": True},
        {"family_sizes": []},
        {"groups": 0},
        {"comparisons": 501},
    ):
        with pytest.raises(ValueError):
            simulate(scenario(**patch), seed=13)
