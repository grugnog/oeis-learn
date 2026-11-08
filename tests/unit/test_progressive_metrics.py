"""Independent counts, pairing, censoring and family-resampling acceptance for T024."""

import pytest


def rows(seed, arm, stratum, group, n, success, unresolved=0):
    assert success + unresolved <= n
    return [
        {
            "seed": seed,
            "arm": arm,
            "stratum": stratum,
            "group": group,
            "task": f"{stratum}/{group}/{i}",
            "outcome": "success"
            if i < success
            else "unresolved"
            if i >= n - unresolved
            else "failure",
        }
        for i in range(n)
    ]


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


def test_quantile_rounding_and_multiplicity_are_prospective():
    from oeis_learn.generalization.progressive.metrics import decision_bounds

    # Sorted fixture 0/10000 .. 9999/10000, one-based floor/ceil indices.
    values = [i / 10000 for i in range(10000)]
    assert decision_bounds(values, comparisons=1) == (0.0499, 0.9499)
    assert decision_bounds(values, comparisons=4) == (0.0124, 0.9874)
    with pytest.raises(ValueError):
        decision_bounds(values, comparisons=0)


def paired_panel(
    *,
    groups=16,
    seeds=(1, 2),
    candidate_success=1,
    baseline_success=0,
    candidate_unresolved=0,
    baseline_unresolved=0,
):
    data = []
    for run in seeds:
        for stratum in ("composition", "unseen_structure"):
            for group in range(groups):
                data += rows(
                    run, "baseline", stratum, str(group), 4, baseline_success, baseline_unresolved
                )
                data += rows(
                    run,
                    "candidate",
                    stratum,
                    str(group),
                    4,
                    candidate_success,
                    candidate_unresolved,
                )
    return data


def test_upper_bound_uses_opposite_censoring_and_seed_effects_are_equal_weighted():
    from oeis_learn.generalization.progressive.metrics import compare

    # Every family has observed difference 1/4, lower -1/4, upper 1/2.
    r = compare(paired_panel(candidate_unresolved=1, baseline_unresolved=2), seed=17)
    assert r["mean_gain"] == 0.25
    assert r["gain_bounds"] == (-0.25, 0.5)
    assert (r["lower"], r["upper"]) == (-0.25, 0.5)
    assert r["nominal_interval"] == (0.25, 0.25)
    assert r["cells"]["baseline"][1]["composition"]["completion"] == 0.5
    assert r["cells"]["candidate"][1]["composition"]["completion"] == 0.75


def test_separate_retention_uses_pessimistic_all_task_aggregate():
    from oeis_learn.generalization.progressive.metrics import compare

    data = paired_panel()
    for run in (1, 2):
        data += rows(run, "baseline", "retention", "familiar", 100, 50, 3)
        data += rows(run, "candidate", "retention", "familiar", 100, 52, 2)
    r = compare(data, seed=11)
    assert r["retention_losses"] == {1: 0.01, 2: 0.01}
    assert r["mean_gain"] == 0.25


def test_equal_seed_weights_and_nonprimary_strata_do_not_change_endpoint():
    from oeis_learn.generalization.progressive.metrics import compare

    data = paired_panel()
    for row in data:
        if row["seed"] == 2 and row["arm"] == "candidate":
            row["outcome"] = "failure"
    for run in (1, 2):
        data += rows(run, "baseline", "oeis", "separate", 200, 0)
        data += rows(run, "candidate", "oeis", "separate", 200, 200)
    r = compare(data, seed=1)
    assert r["seed_effects"] == {1: 0.25, 2: 0.0}
    assert r["seed_range"] == (0.0, 0.25)
    assert r["mean_gain"] == 0.125
    assert r["nominal_interval"] == (0.125, 0.125)


@pytest.mark.parametrize("groups,seeds", [(15, (1, 2)), (16, (1,))])
def test_sparse_support_has_no_claim_interval(groups, seeds):
    from oeis_learn.generalization.progressive.metrics import compare

    r = compare(paired_panel(groups=groups, seeds=seeds), seed=13)
    assert r["support_status"] == "INCOMPLETE"
    assert r["lower"] is None and r["upper"] is None
    assert r["nominal_interval"] is None
    assert r["support_deficits"]


def test_missing_stratum_is_incomplete_not_reweighted():
    from oeis_learn.generalization.progressive.metrics import compare

    r = compare([r for r in paired_panel() if r["stratum"] == "composition"], seed=1)
    assert r["mean_gain"] is None
    assert r["support_status"] == "INCOMPLETE"
    assert r["primary_weights"] == {"composition": 0.5, "unseen_structure": 0.5}


def test_invalid_pairing_and_rows_fail_closed():
    from copy import deepcopy

    from oeis_learn.generalization.progressive.metrics import compare, summarize

    data = paired_panel()
    variants = [data[1:], data + [data[0]]]
    for key, value in (
        ("seed", True),
        ("arm", "surprise"),
        ("outcome", "timeout"),
        ("group", "wrong"),
        ("stratum", "unknown"),
    ):
        changed = deepcopy(data)
        changed[0][key] = value
        variants.append(changed)
    for bad in variants:
        with pytest.raises(ValueError):
            compare(bad, seed=1)
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        summarize(data)  # Never pool treatment/seed cells as independent tasks.


def test_explicit_frozen_roster_keeps_missing_results_in_denominator():
    from oeis_learn.generalization.progressive.metrics import compare

    data = paired_panel()
    roster = [
        {k: r[k] for k in ("task", "group", "stratum")}
        for r in data
        if r["seed"] == 1 and r["arm"] == "baseline"
    ]
    kept = [r for r in data if not (r["arm"] == "candidate" and r["seed"] == 2)]
    r = compare(kept, roster=roster, seeds=(1, 2), seed=1)
    assert r["cells"]["candidate"][2]["composition"]["tasks"] == 64
    assert r["cells"]["candidate"][2]["composition"]["unresolved"] == 64
    assert r["seed_effects"][2] == 0.0
    assert r["gain_bounds"] == (0.125, 0.625)


def test_exact_nominal_and_adjusted_indices_and_invalid_settings():
    from oeis_learn.generalization.progressive.metrics import (
        compare,
        decision_bounds,
        nominal_interval,
    )

    values = [i / 10000 for i in range(10000)]
    assert nominal_interval(values) == (0.0249, 0.9749)
    assert decision_bounds(list(reversed(values)), comparisons=3) == (0.0165, 0.9833)
    assert decision_bounds(values, comparisons=500) == (0.0, 0.9998)
    for k in (0, 501, True, 1.5):
        with pytest.raises(ValueError):
            decision_bounds(values, comparisons=k)
    for bad in (values[:-1], values[:-1] + [float("nan")], values[:-1] + [True]):
        with pytest.raises(ValueError):
            decision_bounds(bad)
    for kwargs in ({"replicates": 100}, {"seed": True}, {"seed": -1}):
        with pytest.raises(ValueError):
            compare(paired_panel(), **({"seed": 1} | kwargs))


def test_small_resampling_fixture_preserves_linked_weights_and_renormalizes():
    from oeis_learn.generalization.progressive.metrics import _resample

    # Family vectors are (lower, observed, upper) after equal seed averaging.
    cells = {
        "composition": {"a": (0, 0, 0), "b": (1, 1, 1)},
        "unseen_structure": {"a": (1, 1, 1), "b": (0, 0, 0)},
    }
    links = [("composition/a", "unseen_structure/a"), ("composition/b", "unseen_structure/b")]
    # Opposite family effects cancel in every common draw, not merely in expectation.
    assert _resample(cells, links, seed=7, replicates=12) == [(0.5, 0.5, 0.5)] * 12
    # Independent draws instead produce paired means 0, 1/4, 1/2, 3/4, 1.
    independent = _resample(cells, [], seed=7, replicates=100)
    assert set(independent) == {(v, v, v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)}
    assert independent == _resample(cells, [], seed=7, replicates=100)
    assert independent != _resample(cells, [], seed=8, replicates=100)


def test_linked_unequal_stratum_sizes_have_fixed_weights():
    from oeis_learn.generalization.progressive.metrics import _resample

    cells = {
        "composition": {"a": (1, 1, 1)},
        "unseen_structure": {"a": (0, 0, 0), "b": (0, 0, 0), "c": (0, 0, 0)},
    }
    result = _resample(cells, [("composition/a", "unseen_structure/a")], seed=13, replicates=20)
    assert result == [(0.5, 0.5, 0.5)] * 20  # Never pooled to 1/4.


def test_link_validation_and_group_dependence_count():
    from oeis_learn.generalization.progressive.metrics import compare

    for links in (
        [("composition/alien", "unseen_structure/0")],
        [("composition/0", "composition/1")],
        [("composition/0", "unseen_structure/0")] * 2,
    ):
        with pytest.raises(ValueError):
            compare(paired_panel(), seed=1, links=links)
    r = compare(
        paired_panel(),
        seed=1,
        links=[("composition/0", "unseen_structure/0"), ("composition/1", "unseen_structure/0")],
    )
    assert r["minimum_groups"] == 15
    assert r["support_status"] == "INCOMPLETE"


def test_policy_binds_rounding_censoring_and_precedence():
    from oeis_learn.generalization.progressive.targets import defaults, validate_policy
    from oeis_learn.pipeline_validation.identity import seal

    p = defaults()
    stat = p["payload"]["statistics"]
    assert stat["replicates"] == 10000
    assert stat["lower_order_statistic"] == "max(1,floor(B/(20*K)))"
    assert stat["upper_order_statistic"] == "min(B,ceil(B*(20*K-1)/(20*K)))"
    assert stat["no_material_gain"] == "conservative_adjusted_upper < 0.02"
    assert stat["confirmed_regression"] == "conservative_adjusted_upper < 0"
    assert p["payload"]["verdict_precedence"] == [
        "INVALID",
        "INCOMPLETE",
        "REGRESSION",
        "CONFIRMED_GAIN",
        "CONFIRMED_EFFICIENCY",
        "NO_MATERIAL_GAIN",
        "INCONCLUSIVE",
    ]
    stat["replicates"] = 100
    with pytest.raises(ValueError):
        validate_policy(seal(p))


def test_historical_seed_population_guard_is_unchanged():
    from oeis_learn.generalization.metrics import seed_uncertainty

    with pytest.raises(ValueError):
        seed_uncertainty({1: (0, 1), 2: (0, 1)})


def test_partial_link_fixture_matches_independent_integer_draw_counts():
    from fractions import Fraction as F

    from oeis_learn.generalization.progressive.metrics import _resample

    cells = {
        "composition": {"a": (1, 1, 1), "b": (0, 0, 0)},
        "unseen_structure": {"a": (0, 0, 0), "c": (1, 1, 1), "d": (0, 0, 0)},
    }
    # Seed13's union-cluster counts: (0,2,2,0), (2,1,1,0), (1,1,1,1),
    # (1,1,1,1), (0,1,2,1), (0,1,1,2), (1,0,2,1), (0,1,1,2).
    # Independently: 1/2 * [a/(a+b) + c/(a+c+d)], retaining full families.
    expected = [F(1, 2), F(1, 2), F(5, 12), F(5, 12), F(1, 3), F(1, 6), F(3, 4), F(1, 6)]
    assert _resample(cells, [("composition/a", "unseen_structure/a")], seed=13, replicates=8) == [
        (v, v, v) for v in expected
    ]


def test_success_does_not_imply_every_attempt_completed():
    from oeis_learn.generalization.progressive.metrics import summarize

    data = rows(1, "candidate", "composition", "a", 2, 1)
    data[0]["complete"] = False
    r = summarize(data)
    assert r["all_task_yield"] == 0.5
    assert r["completion"] == 0.5
    assert r["unresolved"] == 0
    data[1]["complete"] = False
    with pytest.raises(ValueError):
        summarize(data)  # An unfinished unsuccessful portfolio is unresolved.


def test_order_invariance_and_nontrivial_censored_intervals():
    from oeis_learn.generalization.progressive.metrics import compare

    data = paired_panel()
    for row in data:
        if row["arm"] == "candidate" and int(row["group"]) < 8:
            row["outcome"] = "success"
    a = compare(data, seed=321, comparisons=4)
    b = compare(list(reversed(data)), seed=321, comparisons=4)
    assert a == b
    assert a["lower"] < a["mean_gain"] < a["upper"]
    assert a["lower"] <= a["nominal_interval"][0]
    assert a["upper"] >= a["nominal_interval"][1]
    censored = [dict(r, outcome="unresolved") if r["arm"] == "candidate" else r for r in data]
    c = compare(censored, seed=321)
    assert c["mean_gain"] == 0.0
    assert (c["lower"], c["upper"]) == (0.0, 1.0)
    # Candidate censoring cannot establish no material gain or regression.
    assert c["upper"] >= 0.02


def test_oeis_endpoint_has_full_weight_and_independent_seed_support():
    from oeis_learn.generalization.progressive.metrics import compare

    values = []
    roster = []
    for g in range(16):
        for i in range(4):
            task = f"oeis-{g}-{i}"
            roster.append(dict(task=task, group=str(g), stratum="oeis"))
            for seed in (1, 2):
                for arm in ("baseline", "candidate"):
                    values.append(
                        dict(
                            task=task,
                            group=str(g),
                            stratum="oeis",
                            seed=seed,
                            arm=arm,
                            outcome="success" if arm == "candidate" or i < 3 else "failure",
                        )
                    )
    result = compare(values, roster=roster, seeds=[1, 2], seed=13, primary=("oeis",))
    assert result["primary_weights"] == {"oeis": 1.0}
    assert result["mean_gain"] == result["lower"] == result["upper"] == 0.25
    assert result["support"] == {"oeis": 16}
