"""Prospective cumulative milestones; finite slots never restart a schedule."""

from oeis_learn.pipeline_validation.identity import seal

from .trials import integer

MILESTONES = (25000000, 100000000, 400000000)


def scaling_plan():
    return {
        "version": "progressive-scaling/v1",
        "executable": False,
        "milestones": list(MILESTONES),
        "schedule_horizon_tokens": 400000000,
        "warmup_tokens": 1250000,
        "final_multiplier": 0.1,
        "maximum_routine_slots": 8,
        "routine_seconds": 168 * 3600,
        "diagnostic_seconds": 24 * 3600,
        "maximum_task_seconds": 86400,
        "initializer": None,
        "qualification": None,
        "task_slots": None,
    }


def milestone(tokens, *, ancestor_tokens, ancestor_cost_us, new_cost_us):
    for x in (tokens, ancestor_tokens, ancestor_cost_us, new_cost_us):
        integer(x)
    if tokens < ancestor_tokens:
        raise ValueError("cumulative exposure cannot decrease")
    reached = [m for m in MILESTONES if tokens >= m]
    return seal(
        {
            "version": "progressive-milestone/v1",
            "tokens": tokens,
            "reached": reached,
            "overshoot": tokens - reached[-1] if reached else 0,
            "ancestor_tokens": ancestor_tokens,
            "new_tokens": tokens - ancestor_tokens,
            "total_cost_us": ancestor_cost_us + new_cost_us,
            "new_cost_us": new_cost_us,
            "schedule_horizon_tokens": 400000000,
        }
    )


def scaling_gate(decision, rows, roster, seeds, design, *, remaining, required, **kwargs):
    from .decisions import verify

    verify(decision, rows, roster, seeds, design, **kwargs)
    p = decision["payload"]
    if p["evidence_verdict"] != "CONFIRMED_GAIN" or p["next_action"] != "ADVANCE_REGISTERED":
        return "RETAIN_BASELINE"
    for k, v in required.items():
        integer(v)
        if v > remaining.get(k, 0):
            return "DEFER_BRANCH"
    return "ADVANCE_REGISTERED"
