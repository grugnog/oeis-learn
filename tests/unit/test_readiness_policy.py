"""Unit tests for pure readiness policy evaluation and gate logic."""

from __future__ import annotations

import pytest
from oeis_learn.data.models import ReadinessPolicy, ReadinessThreshold
from oeis_learn.evaluation.readiness import evaluate_readiness_policy


def sample_policy() -> ReadinessPolicy:
    h = "sha256:" + "0" * 64
    return ReadinessPolicy(
        schema_version="1.0",
        policy_id=h,
        name="test_policy",
        thresholds=[
            ReadinessThreshold("exact_success", "exact_success_count", "GE", 1.0, "count", "spec"),
            ReadinessThreshold("assembly_validity", "assembly_validity_rate", "EQ", 1.0, "ratio", "const"),
            ReadinessThreshold("trap_rate", "runtime_trap_rate", "LE", 0.15, "ratio", "const"),
            ReadinessThreshold("coverage", "min_coverage", "GE", 0.50, "ratio", "const"),
        ],
        required_experiment_ids=["exp1"],
        required_artifacts=["art1"],
        native_evaluator_required=True,
    )


def test_readiness_policy_all_passed():
    policy = sample_policy()
    metrics = {
        "exact_success_count": 2.0,
        "assembly_validity_rate": 1.0,
        "runtime_trap_rate": 0.05,
        "min_coverage": 0.65,
    }
    report = evaluate_readiness_policy(policy, metrics, run_id="run_100")
    assert report.overall_passed is True
    assert report.qualification_state == "AUTHORIZED"
    assert report.override is None
    assert len(report.gate_results) == 4
    for gr in report.gate_results:
        assert gr.passed is True


def test_readiness_policy_fails_on_trap_rate():
    policy = sample_policy()
    metrics = {
        "exact_success_count": 2.0,
        "assembly_validity_rate": 1.0,
        "runtime_trap_rate": 0.16,  # > 0.15 threshold
        "min_coverage": 0.65,
    }
    report = evaluate_readiness_policy(policy, metrics, run_id="run_100")
    assert report.overall_passed is False
    assert report.qualification_state == "BLOCKED"
    failed_gates = [gr.gate_id for gr in report.gate_results if not gr.passed]
    assert failed_gates == ["trap_rate"]


def test_readiness_policy_fails_on_zero_success():
    policy = sample_policy()
    metrics = {
        "exact_success_count": 0.0,
        "assembly_validity_rate": 1.0,
        "runtime_trap_rate": 0.0,
        "min_coverage": 0.55,
    }
    report = evaluate_readiness_policy(policy, metrics, run_id="run_100")
    assert report.overall_passed is False
    assert report.qualification_state == "BLOCKED"


def test_readiness_policy_diagnostic_override():
    policy = sample_policy()
    metrics = {
        "exact_success_count": 0.0,
        "assembly_validity_rate": 1.0,
        "runtime_trap_rate": 0.20,
        "min_coverage": 0.40,
    }
    override_kwargs = {
        "operator": "researcher",
        "reason": "Debugging Stage 2 loop rotation",
        "diagnostic_intent": "Capture trace without blocking local dev",
    }
    report = evaluate_readiness_policy(
        policy, metrics, run_id="run_100", override_info=override_kwargs
    )
    # Overall passed remains False
    assert report.overall_passed is False
    # State changes to OVERRIDDEN_UNQUALIFIED
    assert report.qualification_state == "OVERRIDDEN_UNQUALIFIED"
    assert report.override is not None
    assert set(report.override.failed_gate_ids) == {"exact_success", "trap_rate", "coverage"}
