"""Contract tests for readiness report schema conformance."""

from __future__ import annotations

import copy
import pytest
from jsonschema import ValidationError


@pytest.fixture
def valid_readiness_report_data() -> dict:
    h = "sha256:" + "0" * 64
    return {
        "schema_version": "1.0",
        "report_id": "rep_readiness_001",
        "run_id": "run_008_test",
        "created_at": "2026-09-04T12:00:00Z",
        "policy": {
            "schema_version": "1.0",
            "policy_id": h,
            "name": "tier1_readiness_v1",
            "thresholds": [
                {
                    "gate_id": "runtime_trap_rate",
                    "metric": "runtime_trap_rate",
                    "comparator": "LE",
                    "threshold": 0.15,
                    "unit": "ratio",
                    "source": "constitution/III",
                    "non_relaxable": True,
                }
            ],
            "required_experiment_ids": ["trustworthy_inference_v1"],
            "required_artifacts": ["synthesis_evaluation"],
            "native_evaluator_required": True,
        },
        "gate_results": [
            {
                "gate_id": "runtime_trap_rate",
                "measured_value": 0.05,
                "threshold": {
                    "gate_id": "runtime_trap_rate",
                    "metric": "runtime_trap_rate",
                    "comparator": "LE",
                    "threshold": 0.15,
                    "unit": "ratio",
                    "source": "constitution/III",
                    "non_relaxable": True,
                },
                "passed": True,
                "evaluated_at": "2026-09-04T12:00:00Z",
                "evidence": [
                    {
                        "artifact_id": "eval_001",
                        "artifact_sha256": h,
                    }
                ],
                "diagnostics": [],
            }
        ],
        "overall_passed": True,
        "override": None,
        "qualification_state": "AUTHORIZED",
    }


def test_valid_readiness_report_conforms(valid_readiness_report_data, validate_contract):
    validate_contract(valid_readiness_report_data, "readiness-report")


def test_authorized_requires_overall_passed_true(valid_readiness_report_data, validate_contract):
    d = copy.deepcopy(valid_readiness_report_data)
    d["overall_passed"] = False
    with pytest.raises(ValidationError):
        validate_contract(d, "readiness-report")


def test_overridden_unqualified_requires_override_record(valid_readiness_report_data, validate_contract):
    d = copy.deepcopy(valid_readiness_report_data)
    d["qualification_state"] = "OVERRIDDEN_UNQUALIFIED"
    d["overall_passed"] = False
    d["override"] = None
    with pytest.raises(ValidationError):
        validate_contract(d, "readiness-report")


def test_valid_override_record(valid_readiness_report_data, validate_contract):
    h = "sha256:" + "0" * 64
    d = copy.deepcopy(valid_readiness_report_data)
    d["qualification_state"] = "OVERRIDDEN_UNQUALIFIED"
    d["overall_passed"] = False
    d["override"] = {
        "override_id": "ovr_001",
        "operator": "researcher",
        "created_at": "2026-09-04T12:05:00Z",
        "reason": "Diagnostic investigation of Stage 2 recurrence traps",
        "diagnostic_intent": "Capture memory traces only",
        "failed_gate_ids": ["runtime_trap_rate"],
        "policy_id": h,
    }
    validate_contract(d, "readiness-report")
