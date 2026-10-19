"""Complete scheduled screen cost; failure requires a newly registered recipe."""

import math


def screen_cost(
    *,
    training_p95,
    teacher_forcing_p95,
    generation_p95,
    fit_p95,
    reference_p95,
    hidden_scoring_p95,
    checkpoint_p95,
    updates=3000,
    cadence=500,
    budget=7200,
):
    costs = (
        training_p95,
        teacher_forcing_p95,
        generation_p95,
        fit_p95,
        reference_p95,
        hidden_scoring_p95,
        checkpoint_p95,
    )
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in costs):
        raise ValueError("measured finite nonnegative p95 costs required")
    if (
        type(updates) is not int
        or type(cadence) is not int
        or updates <= 0
        or cadence <= 0
        or type(budget) not in (int, float)
        or not math.isfinite(budget)
        or budget <= 0
    ):
        raise ValueError("invalid registered schedule")
    scheduled = updates // cadence
    phases = dict(
        training=updates * training_p95,
        periodic_synthesis=scheduled * 96 * (generation_p95 + fit_p95 + reference_p95),
        full_development_loss=(scheduled + 1) * 256 * teacher_forcing_p95,
        end_screen_generation=256 * 32 * generation_p95,
        end_screen_fitting=256 * 40 * (fit_p95 + reference_p95),
        # raw, same-candidate rescue, equal-budget, shuffled, absent, template
        end_screen_hidden=256 * 6 * hidden_scoring_p95,
        checkpoint=(scheduled + 1) * checkpoint_p95,
    )
    total = sum(phases.values())
    return dict(
        phases=phases,
        projected_seconds=total,
        budget=budget,
        reserve_fraction=0.2,
        within_budget=total <= 0.8 * budget,
        scheduled_validations=scheduled,
        end_screen_prompts=256,
        omitted_prompts=0,
    )
