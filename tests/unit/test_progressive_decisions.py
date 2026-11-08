"""Independent T019 specifications still awaiting the T025 decision implementation."""

import pytest

pending = pytest.mark.xfail(
    strict=True, raises=ModuleNotFoundError, reason="T025 decision implementation remains pending"
)


@pending
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


@pending
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
