"""Deterministic exhaustion-only fresh-call retries with one charged aggregate budget.

Streaming has its own rollback/replay controller; this service never mutates stream state.
"""

from oeis_learn.sandbox.pipeline import Limits
from oeis_learn.experiments.decisions import frozen_protocol
import time


def execute_escalating(evaluate, source, indices, *, deadline_ns=None):
    indices = list(indices)
    start = time.monotonic_ns()
    if deadline_ns is not None and type(deadline_ns) is not int:
        raise ValueError("exact external deadline required")
    deadline = (
        min(deadline_ns, start + 2_000_000_000)
        if deadline_ns is not None
        else start + 2_000_000_000
    )
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
            used += limits.fuel_aggregate
            steps += limits.reference_aggregate
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


def run_escalating(runner, source, indices, *, request_id, deadline_ns=None):
    """Production adapter: durable tier requests, all costs attached before selection."""
    from copy import deepcopy
    from oeis_learn.sandbox.pipeline import ExecutionEvidence, digest_bytes

    results = []

    def evaluate(body, positions, limits):
        result = runner.run_single(
            body,
            positions,
            request_id=f"{request_id}-fuel{limits.fuel_per_term}",
            limits=limits,
            deadline_ns=deadline_ns,
        )
        results.append(result)
        return result

    positions = list(indices)
    report = execute_escalating(evaluate, source, positions, deadline_ns=deadline_ns)
    result = (
        deepcopy(results[-1])
        if results
        else ExecutionEvidence(
            digest_bytes(source.encode()),
            positions,
            outcome="execution_limit",
            reason="candidate_deadline",
        )
    )
    result.retry_attempts = report["attempts"]
    result.fuel = report["aggregate_fuel"]
    result.reference_steps = report["aggregate_reference_steps"]
    result.fuel_terms = [fuel for attempt in results for fuel in attempt.fuel_terms]
    result.elapsed_ns = sum(attempt.elapsed_ns for attempt in results)
    result.phase_timings_ns = {}
    for attempt in results:
        for name, duration in attempt.phase_timings_ns.items():
            result.phase_timings_ns[name] = result.phase_timings_ns.get(name, 0) + duration
    return result
