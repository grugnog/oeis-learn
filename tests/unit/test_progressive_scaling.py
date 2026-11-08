import pytest

from oeis_learn.generalization.progressive.contingencies import validate_branch
from oeis_learn.generalization.progressive.milestones import milestone, scaling_plan


def test_full_update_overshoot_and_frozen_horizon():
    assert (
        milestone(25000003, ancestor_tokens=0, ancestor_cost_us=4, new_cost_us=5)["overshoot"] == 3
    )
    assert scaling_plan()["schedule_horizon_tokens"] == 400000000
    with pytest.raises(ValueError):
        milestone(True, ancestor_tokens=0, ancestor_cost_us=0, new_cost_us=0)


def test_reserved_diagnostic_requires_nontraining_evidence_and_fresh_panel():
    branch = {
        "id": "b",
        "failure_class": "weak",
        "trigger": "heldout_gap",
        "design": "fixed",
        "panel": "panel",
        "task": "task",
        "reservation": "budget",
        "maximum_visits": 1,
        "terminal": "DEFER_BRANCH",
    }
    assert (
        validate_branch(
            branch,
            evidence={"heldout_gap": True},
            remaining={"budget": 1},
            visited=[],
            exposed_panels=[],
        )
        == branch
    )
    with pytest.raises(ValueError):
        validate_branch(
            branch,
            evidence={"training_loss": 1},
            remaining={"budget": 1},
            visited=[],
            exposed_panels=[],
        )
    with pytest.raises(ValueError):
        validate_branch(
            branch,
            evidence={"heldout_gap": True},
            remaining={"budget": 1},
            visited=["weak"],
            exposed_panels=[],
        )
    with pytest.raises(ValueError):
        validate_branch(
            branch,
            evidence={"heldout_gap": True},
            remaining={"budget": 1},
            visited=[],
            exposed_panels=["panel"],
        )


def test_marginal_gain_keeps_measured_ancestor_costs_and_has_no_gate_authority():
    from oeis_learn.generalization.progressive.decisions import publish
    from oeis_learn.generalization.progressive.milestones import gain_cost
    from tests.unit.test_progressive_decisions import fixture

    data, roster, design, policy = fixture()
    decision = publish(
        data,
        roster,
        [1, 2],
        design,
        policy=policy,
        target_id="gain",
        action="RETAIN_BASELINE",
        remaining={},
        created_at=policy["created_at"],
        purpose=policy["purpose"],
        track=policy["track"],
    )
    value = gain_cost(
        decision,
        data,
        roster,
        [1, 2],
        design,
        costs={"current": 3600000000},
        ancestor_costs={"initial": 7200000000, "preparation": 100},
    )
    assert value["marginal_gain_per_active_hour"] == 1
    assert value["total_active_us"] == 10800000100
    assert value["ancestor_costs"] == {"initial": 7200000000, "preparation": 100}
    assert "next_action" not in value
