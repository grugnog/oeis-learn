"""Independent T019 specifications still awaiting the T025 decision implementation."""

import pytest


@pytest.mark.parametrize(
    "completion,seed_effects,expected",
    [
        (0.949, [0.02, 0.02], "INCOMPLETE"),
        (1.0, [0.06, -0.02], "INCONCLUSIVE"),
        (1.0, [0.001, 0.001], "NO_MATERIAL_GAIN"),
        (1.0, [0.02, 0.02], "CONFIRMED_GAIN"),
    ],
)
def test_two_pp_mean_is_insufficient_with_negative_seed_or_949_completion(
    completion, seed_effects, expected
):
    from oeis_learn.generalization.progressive.decisions import classify

    # Every scalar here is independently selected; no production reducer computes expectations.
    r = classify(
        mean_gain=sum(seed_effects) / 2,
        seed_effects=seed_effects,
        completion_by_arm_seed_stratum=[completion] * 8,
        retention_losses=[0.0, 0.0],
        minimum_groups=16,
        lower=0.001 if expected == "CONFIRMED_GAIN" else -0.01,
        upper=0.01 if expected == "NO_MATERIAL_GAIN" else 0.08,
        integrity=True,
        claim="capability",
    )
    assert r == expected


def test_sparse_groups_per_seed_retention_and_efficiency_only():
    from oeis_learn.generalization.progressive.decisions import classify

    base = {
        "mean_gain": 0.03,
        "seed_effects": [0.03, 0.03],
        "completion_by_arm_seed_stratum": [1.0] * 8,
        "retention_losses": [0.0, 0.021],
        "minimum_groups": 16,
        "lower": 0.01,
        "upper": 0.05,
        "integrity": True,
        "claim": "capability",
    }
    assert classify(**base) != "CONFIRMED_GAIN"
    assert (
        classify(**(base | {"retention_losses": [0.0, 0.0], "minimum_groups": 15})) == "INCOMPLETE"
    )
    assert (
        classify(
            **(
                base
                | {
                    "retention_losses": [0.0, 0.0],
                    "mean_gain": 0.0,
                    "seed_effects": [0.0, 0.0],
                    "lower": -0.005,
                    "claim": "efficiency",
                    "cost_reduction": 0.2,
                }
            )
        )
        == "CONFIRMED_EFFICIENCY"
    )


def fixture():
    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.generalization.progressive.targets import defaults
    from oeis_learn.pipeline_validation.identity import seal
    from tests.unit.test_progressive_metrics import paired_panel, rows

    data = paired_panel(candidate_success=4)
    for seed in (1, 2):
        for arm in ("baseline", "candidate"):
            data += rows(seed, arm, "retention", "familiar", 4, 4)
    roster = [
        {k: r[k] for k in ("task", "group", "stratum")}
        for r in data
        if r["seed"] == 1 and r["arm"] == "baseline"
    ]
    design = {"claim": "capability", "bootstrap_seed": 13, "comparisons": 1, "links": [], "costs": None}
    p = defaults()
    p["payload"]["selected"] = True
    target = next(t for t in p["payload"]["targets"] if t["id"] == "gain")
    target["evidence"].append(digest(design))
    return data, roster, design, seal(p)


def test_sealed_decision_recomputes_and_separately_checks_action():
    from oeis_learn.generalization.progressive.decisions import publish, verify
    from oeis_learn.pipeline_validation.identity import seal

    data, roster, design, policy = fixture()
    kwargs = {"policy": policy, "target_id": "gain", "action": "RETAIN_BASELINE", "remaining": {}}
    result = publish(
        data,
        roster,
        [1, 2],
        design,
        created_at="2026-11-08T00:00:00Z",
        purpose="acceptance",
        track="synthetic_strict",
        **kwargs,
    )
    assert result["payload"]["evidence_verdict"] == "CONFIRMED_GAIN"
    assert verify(result, data, roster, [1, 2], design, **kwargs) == result
    result["payload"]["evidence_verdict"] = "INCONCLUSIVE"
    with pytest.raises(ValueError):
        verify(seal(result), data, roster, [1, 2], design, **kwargs)
    with pytest.raises(ValueError):
        publish(
            data,
            roster,
            [1, 2],
            design,
            created_at="2026-11-08T00:00:00Z",
            purpose="acceptance",
            track="synthetic_strict",
            **(kwargs | {"action": "ADVANCE_REGISTERED"}),
        )


def test_design_and_threshold_cannot_be_selected_after_outcomes():
    from oeis_learn.generalization.progressive.decisions import publish
    from oeis_learn.pipeline_validation.identity import seal

    data, roster, design, policy = fixture()
    kwargs = {
        "target_id": "gain",
        "action": "RETAIN_BASELINE",
        "remaining": {},
        "created_at": "2026-11-08T00:00:00Z",
        "purpose": "acceptance",
        "track": "synthetic_strict",
    }
    with pytest.raises(ValueError, match="registered"):
        publish(data, roster, [1, 2], design | {"comparisons": 2}, policy=policy, **kwargs)
    next(t for t in policy["payload"]["targets"] if t["id"] == "gain")["threshold"] = "0"
    with pytest.raises(ValueError, match="threshold"):
        publish(data, roster, [1, 2], design, policy=seal(policy), **kwargs)


@pytest.mark.parametrize(
    "upper,expected",
    [
        (-0.0001, "REGRESSION"),
        (0, "NO_MATERIAL_GAIN"),
        (0.0199, "NO_MATERIAL_GAIN"),
        (0.02, "INCONCLUSIVE"),
    ],
)
def test_strict_upper_bounds_and_precedence(upper, expected):
    values = {
        "mean_gain": 0,
        "seed_effects": [0, 0],
        "completion_by_arm_seed_stratum": [1.0] * 8,
        "retention_losses": [0, 0],
        "minimum_groups": 16,
        "lower": -0.1,
        "upper": upper,
        "integrity": True,
        "claim": "capability",
    }
    from oeis_learn.generalization.progressive.decisions import classify

    assert classify(**values) == expected
    assert classify(**(values | {"integrity": False})) == "INVALID"
    assert classify(**(values | {"minimum_groups": 15})) == "INCOMPLETE"


def test_efficiency_requires_each_primary_stratum():
    from oeis_learn.generalization.progressive.decisions import classify

    values = {
        "mean_gain": 0,
        "seed_effects": [0, 0],
        "completion_by_arm_seed_stratum": [1.0] * 8,
        "retention_losses": [0, 0],
        "minimum_groups": 16,
        "lower": -0.005,
        "upper": 0.05,
        "integrity": True,
        "claim": "efficiency",
        "cost_reduction": 0.2,
        "stratum_lowers": [-0.005, -0.011],
    }
    assert classify(**values) == "INCONCLUSIVE"
    assert classify(**(values | {"stratum_lowers": [-0.01, -0.005]})) == "CONFIRMED_EFFICIENCY"


def test_production_efficiency_uses_per_stratum_conservative_bounds():
    from oeis_learn.generalization.progressive.decisions import recompute

    data, roster, design, _ = fixture()
    for row in data:
        if row["stratum"] == "composition" and row["arm"] == "baseline":
            row["outcome"] = "success"
    metrics, verdict = recompute(
        data,
        roster,
        [1, 2],
        design | {"claim": "efficiency", "costs": {"baseline": 100, "candidate": 80}},
    )
    assert verdict == "CONFIRMED_EFFICIENCY"
    assert metrics["efficiency_stratum_lowers"] == {"composition": 0.0, "unseen_structure": 1.0}
