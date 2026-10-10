"""Deterministic exhaustion-only fresh-call retries with one charged aggregate budget.

Streaming has its own rollback/replay controller; this service never mutates stream state.
"""

from oeis_learn.sandbox.pipeline import Limits
from oeis_learn.experiments.decisions import frozen_protocol
import time


def execute_escalating(evaluate, source, indices, *, deadline_ns=None):
    indices = list(indices)
    deadline = deadline_ns or time.monotonic_ns() + 2_000_000_000
    used, steps, records = 0, 0, []
    for tier in frozen_protocol()["fuel_tiers"]:
        remaining = deadline - time.monotonic_ns()
        if remaining <= 0 or used >= 50_000_000 or steps >= 50_000_000:
            break
        limits = Limits(
            fuel_per_term=tier,
            fuel_aggregate=50_000_000 - used,
            reference_aggregate=50_000_000 - steps,
            deadline_ns=min(2_000_000_000, remaining),
        )
        result = evaluate(source, indices, limits)
        if result.fuel is None or result.reference_steps is None:
            records.append(dict(tier=tier, evidence=result.to_dict()))
            break
        if type(result.fuel) is not int or not 0 <= result.fuel <= limits.fuel_aggregate:
            raise ValueError("invalid retry consumption")
        if (
            type(result.reference_steps) is not int
            or not 0 <= result.reference_steps <= limits.reference_aggregate
        ):
            raise ValueError("invalid retry reference consumption")
        used += result.fuel
        steps += result.reference_steps
        records.append(dict(tier=tier, evidence=result.to_dict()))
        if result.outcome != "execution_limit" or result.reason != "wasmtime_fuel":
            break
    return dict(
        attempts=records,
        aggregate_fuel=used,
        aggregate_reference_steps=steps,
        selected=False,
        budget_refunded=False,
        completed=bool(records) and records[-1]["evidence"]["outcome"] is None,
    )
