import pytest
from oeis_learn.tracking.foundation_metrics import fuel_summary, PhaseTimer
from oeis_learn.sandbox.pipeline import ExecutionEvidence
from oeis_learn.evaluation.fuel_escalation import execute_escalating


def test_observed_fuel_and_synchronized_phase_boundaries():
    result = fuel_summary([1, 75, 100], limit=100)
    assert result["aggregate"] == 176 and result["near_limit_count"] == 2
    assert result["p50"] == 75 and result["p95"] == 100
    for bad in ([True], [101], [-1]):
        with pytest.raises(ValueError):
            fuel_summary(bad, limit=100)
    ticks = iter([1, 11])
    timer = PhaseTimer(clock=lambda: next(ticks))
    calls = []
    timer.synchronize = lambda: calls.append("sync")
    with timer.phase("decode"):
        pass
    assert calls == ["sync", "sync"] and timer.values == {"decode": 10}


def test_escalation_charges_failures_before_selection_and_stops_on_other_limits():
    limits = []

    def evaluate(source, indices, limit):
        limits.append(limit)
        failure = len(limits) < 3
        return ExecutionEvidence(
            "sha256:" + "0" * 64,
            indices,
            fuel=limit.fuel_per_term,
            reference_steps=10,
            outcome="execution_limit" if failure else None,
            reason="wasmtime_fuel" if failure else None,
        )

    report = execute_escalating(evaluate, "i256.zero", [0])
    assert [l.fuel_per_term for l in limits] == [100000, 250000, 1000000]
    assert [l.fuel_aggregate for l in limits] == [50000000, 49900000, 49650000]
    assert report["aggregate_fuel"] == 1350000 and report["selected"] is False

    def other(source, indices, limit):
        return ExecutionEvidence(
            "sha256:" + "0" * 64,
            indices,
            fuel=1,
            reference_steps=1,
            outcome="execution_limit",
            reason="candidate_deadline",
        )

    assert len(execute_escalating(other, "i256.zero", [0])["attempts"]) == 1


def test_plan_defaults_match_frozen_config():
    import yaml
    from oeis_learn.experiments.decisions import frozen_protocol

    assert yaml.safe_load(open("configs/foundation/experiments.yaml")) == frozen_protocol()
