"""T019 independent examples for the T024–T025 implementation contract.

These specification tests intentionally xfail only while the later metrics/decision
modules are absent. They are not software acceptance of those later tasks; adding
those modules without satisfying the cases makes the strict markers fail.
"""

import pytest

pending = pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="T024/T025 are outside T001–T023; metric/gate behavior not implemented",
)


def rows(seed, arm, stratum, group, n, success, unresolved=0):
    assert success + unresolved <= n
    return [
        dict(
            seed=seed,
            arm=arm,
            stratum=stratum,
            group=group,
            task=f"{stratum}/{group}/{i}",
            outcome="success"
            if i < success
            else "unresolved"
            if i >= n - unresolved
            else "failure",
        )
        for i in range(n)
    ]


@pending
def test_family_imbalance_all_task_denominator_and_censoring():
    from oeis_learn.generalization.progressive.metrics import summarize

    data = rows(1, "candidate", "composition", "small", 2, 2) + rows(
        1, "candidate", "composition", "large", 8, 0, 2
    )
    r = summarize(data)
    assert r["all_task_yield"] == 0.2
    assert r["family_macro"] == 0.5
    assert r["completion"] == 0.8
    assert r["success_bounds"] == (0.2, 0.4)


@pending
def test_fixed_endpoint_equal_seeds_unequal_stratum_sizes_and_linked_groups():
    from oeis_learn.generalization.progressive.metrics import compare

    data = []
    for seed in [1, 2]:
        for stratum, n in [("composition", 10), ("unseen_structure", 100)]:
            for group in range(16):
                data += rows(seed, "baseline", stratum, str(group), n, 0)
                data += rows(
                    seed, "candidate", stratum, str(group), n, n if stratum == "composition" else 0
                )
    r = compare(
        data,
        seed=13,
        replicates=10000,
        comparisons=1,
        links=[("composition/0", "unseen_structure/0")],
    )
    assert r["mean_gain"] == 0.5
    assert r["seed_effects"] == {1: 0.5, 2: 0.5}
    assert r["nominal_interval"] == (0.5, 0.5)
    assert r["uncertainty_scope"] == "family_sampling_conditional_on_observed_seeds"


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

    base = dict(
        mean_gain=0.03,
        seed_effects=[0.03, 0.03],
        completion_by_arm_seed_stratum=[1.0] * 8,
        retention_losses=[0.0, 0.021],
        minimum_groups=16,
        lower=0.01,
        upper=0.05,
        integrity=True,
        claim="capability",
    )
    assert classify(**base) != "CONFIRMED_GAIN"
    assert classify(**(base | dict(retention_losses=[0.0, 0.0], minimum_groups=15))) == "INCOMPLETE"
    assert (
        classify(
            **(
                base
                | dict(
                    retention_losses=[0.0, 0.0],
                    mean_gain=0.0,
                    seed_effects=[0.0, 0.0],
                    lower=-0.005,
                    claim="efficiency",
                    cost_reduction=0.2,
                )
            )
        )
        == "CONFIRMED_EFFICIENCY"
    )


@pending
def test_quantile_rounding_and_multiplicity_are_prospective():
    from oeis_learn.generalization.progressive.metrics import decision_bounds

    # Sorted fixture 0/10000 .. 9999/10000, one-based floor/ceil indices.
    values = [i / 10000 for i in range(10000)]
    assert decision_bounds(values, comparisons=1) == (0.0499, 0.9499)
    assert decision_bounds(values, comparisons=4) == (0.0124, 0.9874)
    with pytest.raises(ValueError):
        decision_bounds(values, comparisons=0)
