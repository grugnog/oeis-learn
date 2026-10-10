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


def test_production_reference_phase_timings_are_actual_and_separate():
    from oeis_learn.sandbox.pipeline import Runtime

    evidence = Runtime().evaluate("i256.const 7", range(20))
    assert evidence.outcome is None
    assert set(evidence.phase_timings_ns) == {
        "prepare",
        "production",
        "independent_reference",
        "controller_overhead",
    }
    assert all(type(n) is int and n >= 0 for n in evidence.phase_timings_ns.values())
    assert sum(evidence.phase_timings_ns.values()) == evidence.elapsed_ns


def test_release_aggregation_cannot_promote_lifecycle_or_unreplayed_status(tmp_path):
    from oeis_learn.evaluation.readiness import release_readiness

    report = release_readiness()
    assert report["missing_gates"] == [
        "G0",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
        "G8",
        "G9",
        "G10",
    ]
    assert not report["full_release_qualified"] and report["state"] == "baseline_pending"
    forged = release_readiness(g7=({"status": "PASS"}, tmp_path))
    assert forged["gates"]["G7"]["status"] == "FAIL"
    assert forged["gates"]["G10"]["status"] == "pending"


def test_expired_escalation_deadline_cannot_issue_work():
    def forbidden(*args):
        raise AssertionError("expired work must not execute")

    report = execute_escalating(forbidden, "i256.zero", [0], deadline_ns=0)
    assert report["attempts"] == [] and not report["completed"]
    with pytest.raises(ValueError, match="exact external deadline"):
        execute_escalating(forbidden, "i256.zero", [0], deadline_ns=True)


def test_replayed_cpu_learning_evidence_keeps_hardware_pending(tmp_path, monkeypatch):
    import oeis_learn.evaluation.readiness as readiness

    calls = []

    def validator(report, *, evidence_root, diagnostic=False):
        calls.append((evidence_root, diagnostic))
        assert diagnostic is True

    monkeypatch.setattr(readiness, "require_learning_readiness", validator)
    report = readiness.release_readiness(
        g8=(
            dict(hardware_status="PENDING", hardware=None, report_sha256="sha256:" + "0" * 64),
            tmp_path,
        )
    )
    assert calls == [(tmp_path, True)]
    assert report["gates"]["G8"]["status"] == "pending"
    assert report["gates"]["G8"]["software_status"] == "PASS"
    assert not report["full_release_qualified"]


def test_real_stream_retry_restores_state_and_charges_both_attempts():
    from oeis_learn.sandbox.streaming import next_block

    source = "(stream (init (set 0 (const 0))) (next (set 0 (add (get 0) (const 1))) (repeat 40 (repeat 40 (set 1 (add (get 1) (const 1))))) (return (get 0))))"
    result = next_block(source, 2)
    assert result["outcome"] is None and result["outputs"] == ["1", "2"]
    calls = [r for r in result["telemetry"]["calls"] if r["call"] == "next"]
    assert [r["tier"] for r in calls] == [100000, 250000, 100000, 250000]
    assert (
        calls[0]["production_outcome"] == "execution_limit"
        and calls[0]["reference_outcome"] is None
    )
    assert result["spent_fuel"] == sum(r["fuel"] for r in result["telemetry"]["calls"])
    assert result["spent_fuel"] > sum(r["fuel"] for r in calls if r["production_outcome"] is None)
