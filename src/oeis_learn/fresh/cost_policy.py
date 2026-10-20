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
    generation_batch_p95=None,
    generation_batch_size=1,
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

    batch_costs = dict(
        generation_p95=generation_p95,
        generation_batch_p95=generation_batch_p95,
        generation_batch_size=generation_batch_size,
    )

    phases = dict(
        training=updates * training_p95,
        periodic_synthesis=scheduled
        * (generation_seconds(96, batch_costs) + 96 * (fit_p95 + reference_p95)),
        full_development_loss=(scheduled + 1) * 256 * teacher_forcing_p95,
        end_screen_generation=32 * generation_seconds(256, batch_costs),
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
        generation_batch_size=generation_batch_size,
    )


def generation_seconds(prompts, costs):
    """Charge one decode round, including singleton upper bounds for tail rows."""
    if type(prompts) is not int or prompts < 0:
        raise ValueError("nonnegative exact prompt count required")
    generation_p95 = costs["generation_p95"]
    generation_batch_p95 = costs.get("generation_batch_p95")
    generation_batch_size = costs.get("generation_batch_size", 1)
    if (
        type(generation_p95) not in (int, float)
        or not math.isfinite(generation_p95)
        or generation_p95 < 0
    ):
        raise ValueError("measured finite singleton cost required")
    if (
        type(generation_batch_size) is not int
        or generation_batch_size not in (1, 8)
        or (generation_batch_p95 is None) != (generation_batch_size == 1)
        or (
            generation_batch_p95 is not None
            and (
                type(generation_batch_p95) not in (int, float)
                or not math.isfinite(generation_batch_p95)
                or generation_batch_p95 < 0
            )
        )
    ):
        raise ValueError("complete measured batch8 latency or retained singleton costs required")
    if generation_batch_p95 is None:
        return prompts * generation_p95
    full, tail = divmod(prompts, generation_batch_size)
    return full * generation_batch_p95 + tail * generation_p95
