"""T095: admission and reservations must describe the run actually executed."""

import pytest

from oeis_learn.fresh.cost_policy import screen_cost


def measured():
    return dict(training_p95=1, teacher_forcing_p95=0.01, generation_p95=1,
                generation_batch_p95=2, generation_batch_size=8, fit_p95=0.1,
                reference_p95=0.1, hidden_scoring_p95=0.2, checkpoint_p95=0.5)


def test_diverse32_and_transfer_screen_have_distinct_complete_workloads():
    tiny = screen_cost(**measured(), scope="diverse32", updates=200, cadence=100, budget=14400)
    screen = screen_cost(**measured(), scope="transfer-screen", updates=200, cadence=100)
    assert tiny["scope"] == "diverse32"
    assert tiny["workload"]["periodic_prompts"] == 32
    assert tiny["end_screen_prompts"] == 32
    assert screen["workload"]["periodic_prompts"] == 96
    assert screen["end_screen_prompts"] == 256
    assert tiny["projected_seconds"] < screen["projected_seconds"]
    for result in (tiny, screen):
        assert result["workload"]["generation_rounds"] == 24
        assert result["workload"]["fitting_rounds"] == 32
        assert result["workload"]["hidden_rounds"] == 6
        assert result["omitted_prompts"] == 0
        assert result["reserve_fraction"] == 0.2


def test_scoped_tail_validation_and_role_reservations_match_preflight():
    from oeis_learn.fresh.cost_policy import training_role_seconds, end_role_seconds

    costs = measured()
    result = screen_cost(**costs, scope="diverse32", updates=150, cadence=100, budget=14400)
    assert result["scheduled_validations"] == 2
    reserved = sum(training_role_seconds(costs, "diverse32", n) for n in (100, 50))
    reserved += end_role_seconds(costs, "diverse32", 32, "generation")
    reserved += end_role_seconds(costs, "diverse32", 32, "score")
    assert result["reserved_seconds"] == reserved
    assert reserved <= result["budget"]


@pytest.mark.parametrize("scope", ["tiny", "full", "", False])
def test_unknown_scope_cannot_authorize_training(scope):
    with pytest.raises(ValueError, match="scope"):
        screen_cost(**measured(), scope=scope)
