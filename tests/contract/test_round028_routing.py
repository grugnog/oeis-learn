from collections import Counter
from itertools import product

import pytest

from oeis_learn.generalization.progressive.program_search_routing import (
    bounds, feasible, project, route,
)


def sample(masks, *, fault=False, replay=False):
    masks = list(masks) + ["N"] * (64 - len(masks))
    return [{"task": f"private-{i:02d}", "stratum": "composition" if i < 32 else "unseen_structure",
             "feasible": mask, "unfinished_fit": fault and i == 0, "fault": fault and i == 0,
             "band_witness": False,
             "replay": {"ordinals": [2, 4], "cost_us": 8, "resolves": mask}
                       if replay and len(mask) > 1 else None}
            for i, mask in enumerate(masks)]


@pytest.mark.parametrize(("masks", "support", "expected"), [
    (["P", "P", "NSP"], True, "MATCHED_GENERATOR"),
    (["P", "P", "NSP"], False, "RETAIN_REFERENCE"),
    (["S", "P"], True, "CPU_SELECTOR"),
    (["S", "NP"], True, "CPU_SELECTOR"),
    (["N"], True, "RETAIN_REFERENCE"),
    (["P", "NSP"], True, "RETAIN_REFERENCE"),
])
def test_one_vote_bounds_and_spending(masks, support, expected):
    projection = project(sample(masks), training_support=support)
    assert route(projection)["action"] == expected
    assert "private-" not in str(projection)


def test_fault_precedes_learning_and_overlapping_fit_has_one_vote():
    result = route(project(sample(["P"], fault=True), training_support=True))
    assert result["action"] == "ISOLATE_FAULT"
    assert (result["lower"], result["upper"]) == (1, 1)
    assert result["denominator"] == 64


def test_exhaustive_private_contributions_equal_histogram_bounds():
    for masks in product(("N", "S", "P", "NS", "NP", "SP", "NSP"), repeat=3):
        sums = [sum({"N": 0, "S": -1, "P": 1}[x] for x in assignment)
                for assignment in product(*masks)]
        assert bounds(Counter(masks)) == (min(sums), max(sums))


def test_ordered_replay_must_fit_change_the_action_and_run_once():
    cases = list(reversed(sample(["NSP", "NSP", "NSP"], replay=True)))
    def decision(budget, spent=False):
        return route(project(cases, training_support=True, replay_budget_us=budget, replay_used=spent))
    assert decision(8)["action"] == "RETAIN_REFERENCE"
    assert decision(16)["action"] == "REPLAY"
    assert decision(16)["replay_cases"] == 2
    assert decision(16)["replay_cost_us"] == 16
    assert decision(16, True)["action"] == "RETAIN_REFERENCE"
    assert decision(16) == decision(16)
    assert decision(16) == route(project(list(reversed(cases)), training_support=True, replay_budget_us=16))


def test_incomplete_banks_and_band_witness_cannot_prove_proposal():
    assert feasible(selected="success", bank=None, closed=False, exact_witness=False) == "N"
    assert feasible(selected="failure", bank=["miss", "solve"], closed=True, exact_witness=False) == "S"
    assert feasible(selected="failure", bank=None, closed=False, exact_witness=True) == "NSP"
    assert feasible(selected="failure", bank=[], closed=True, exact_witness=True) == "P"
    assert feasible(selected="failure", bank=["miss"], closed=True, exact_witness=False) == "NP"


def test_votes_require_exact_roster_and_counts():
    cases = sample(["S"])
    with pytest.raises(ValueError):
        project([cases[0], *cases[:-1]], training_support=True)
    p = project(cases, training_support=True)
    p["histogram"]["N"] -= 1
    with pytest.raises(ValueError):
        route(p)
