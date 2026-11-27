"""Acceptance fixtures for a paired confirmation on one previously reserved roster."""

from copy import deepcopy

import pytest

from oeis_learn.generalization.progressive import decisions, metrics
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.graph import velocity_look
from oeis_learn.generalization.progressive.program_search_costing import occupied_time
from oeis_learn.generalization.progressive.training_rows import pair_seeds
from oeis_learn.generalization.progressive.velocity_metrics import case_outcome_receipt, contrast


@pytest.fixture
def replication():
    registration = digest("prospective replication fixture")
    origins = [pair_seeds(registration, i) for i in (0, 1)]
    seeds = [p["initialization"] for p in origins]
    roster = [
        dict(task=f"replication-{s}-{i}", stratum=s, group=f"{s}-{i}")
        for s in ("oeis", "composition", "unseen_structure", "retention")
        for i in range(16)
    ]
    panels = dict(
        development=[r for r in roster if r["stratum"] != "retention"],
        retention=[r for r in roster if r["stratum"] == "retention"],
    )

    def look(method, seed):
        return velocity_look(
            registration=registration,
            method=method,
            seed=seed,
            look=4_000_000,
            profile="full",
            **panels,
        )

    # The incumbent has a single inference coordinate on this roster.
    looks = {"D": look("D", 0)}
    for seed in seeds:
        for method in ("00", "01"):
            looks[method, seed] = look(method, seed)
    rows = []
    for case in roster:
        task, stratum = case["task"], case["stratum"]
        observations = {}
        for name, coords in looks.items():
            method = coords["method"]
            solved = (
                stratum == "retention" or method == "01" or (method == "D" and stratum == "oeis")
            )
            status = "solve" if solved else "miss"
            receipt = case_outcome_receipt(
                task,
                outcome_ref=digest([task, coords["identity"]]),
                choice_ref=digest([coords["identity"], task, "choice"]),
                scorer_ref=digest("scorer"),
                truth_ref=digest([task, "truth"]),
                status=status,
            )
            observations[name] = dict(status=status, closed=True, complete=True, receipt=receipt)
        rows.append(
            dict(case=task, stratum=stratum, family=case["group"], observations=observations)
        )
    return origins, roster, looks, rows


def test_both_seed_pairs_use_the_reserved_panel_and_one_shared_incumbent_charge(replication):
    origins, roster, looks, rows = replication
    assert all(
        origins[0][field] != origins[1][field] for field in ("initialization", "training_families")
    )
    assert all(v["members"] == roster for v in looks.values())
    # Keep string names in a contrast, while the fixture keys distinguish both learner seeds.
    for origin in origins:
        seed = origin["initialization"]
        names = {"finalist": ("01", seed), "ordinary": ("00", seed), "incumbent": "D"}
        named = {k: looks[v] for k, v in names.items()}
        projected = [
            dict(r, observations={k: r["observations"][v] for k, v in names.items()}) for r in rows
        ]
        assert (
            contrast(projected, {"finalist": 1, "ordinary": -1}, stratum="oeis", looks=named)[
                "observed_pp"
            ]
            == 100
        )
        assert (
            contrast(projected, {"finalist": 1, "incumbent": -1}, stratum="oeis", looks=named)[
                "observed_pp"
            ]
            == 0
        )
        assert (
            contrast(
                projected, {"finalist": 1, "incumbent": -1}, stratum="structural", looks=named
            )["observed_pp"]
            == 100
        )
        bad = deepcopy(named)
        del bad["ordinary"]["members"][0]
        with pytest.raises(ValueError, match="frozen look|roster"):
            contrast(projected, {"finalist": 1, "ordinary": -1}, stratum="oeis", looks=bad)
    d = looks["D"]
    debit = dict(
        receipt=digest(["paid", d["identity"]]),
        host="one",
        boot="one",
        start_us=3,
        end_us=14,
        beneficiaries=["seed-one contrast", "seed-two contrast"],
    )
    cost = occupied_time([debit, debit])
    assert cost["receipt_count"] == 1 and cost["union_us"] == 11


@pytest.mark.parametrize(
    ("comparator", "primary", "verdict"),
    [
        ("00", ("oeis",), "CONFIRMED_GAIN"),
        ("D", ("oeis",), "NO_MATERIAL_GAIN"),
        ("D", ("composition", "unseen_structure"), "CONFIRMED_GAIN"),
    ],
)
def test_registered_direct_and_structural_confirmation_use_formal_adjustment(
    replication, comparator, primary, verdict
):
    origins, roster, _, rows = replication
    seeds = [p["initialization"] for p in origins]
    projected = []
    for seed in seeds:
        control = "D" if comparator == "D" else (comparator, seed)
        for row in rows:
            for name, arm in ((control, "baseline"), (("01", seed), "candidate")):
                obs = row["observations"][name]
                projected.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        task=row["case"],
                        group=row["family"],
                        stratum=row["stratum"],
                        outcome={"solve": "success", "miss": "failure"}[obs["status"]],
                        complete=obs["complete"],
                    )
                )
    design = dict(
        claim="capability",
        primary=primary,
        costs=None,
        links=[],
        bootstrap_seed=91,
        comparisons=2 * 2,
    )  # one finalist and look, two comparators, two primary endpoints
    evidence, result = decisions.recompute(projected, roster, seeds, design)
    assert result == verdict and set(evidence["seed_effects"]) == set(seeds)
    assert evidence["minimum_groups"] == 16 and evidence["retention_losses"] == dict.fromkeys(
        seeds, 0
    )
    assert (evidence["replicates"], evidence["comparisons"]) == (10_000, 4)
    if comparator == "00":
        first = [r for r in projected if r["seed"] == seeds[0]]
        assert decisions.recompute(first, roster, seeds[:1], design)[1] == "INCOMPLETE"
        sparse = [r for r in roster if r["group"] != "oeis-15"]
        allowed = {r["task"] for r in sparse}
        assert (
            decisions.recompute(
                [r for r in projected if r["task"] in allowed], sparse, seeds, design
            )[1]
            == "INCOMPLETE"
        )


def test_completion_retention_and_multiplicity_can_each_block_a_formal_gain():
    args = dict(
        mean_gain=0.06,
        seed_effects=[0.04, 0.08],
        minimum_groups=16,
        lower=0.01,
        upper=0.09,
        integrity=True,
        claim="capability",
        primary=("oeis",),
        retention_losses=[0, 0.02],
        completion_by_arm_seed_stratum=[1, 1, 0.95, 1],
    )
    assert decisions.classify(**args) == "CONFIRMED_GAIN"
    assert (
        decisions.classify(**dict(args, completion_by_arm_seed_stratum=[1, 1, 0.949, 1]))
        == "INCOMPLETE"
    )
    assert decisions.classify(**dict(args, retention_losses=[0, 0.0201])) == "INCONCLUSIVE"
    assert decisions.classify(**dict(args, seed_effects=[0, 0.12])) == "INCONCLUSIVE"
    draws = [-0.01] * 200 + [0.01] * 9800
    assert metrics.decision_bounds(draws, comparisons=1)[0] > 0
    adjusted = metrics.decision_bounds(draws, comparisons=4)[0]
    assert adjusted < 0 and decisions.classify(**dict(args, lower=adjusted)) == "INCONCLUSIVE"
