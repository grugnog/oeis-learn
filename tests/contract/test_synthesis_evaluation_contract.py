"""Contract tests for synthesis evaluation result schema."""

from __future__ import annotations

import copy
import pytest
from jsonschema import ValidationError


def make_valid_stage_records():
    stages = [
        "GENERATION",
        "CONSTANT_RESOLUTION",
        "CANONICALIZATION",
        "ASSEMBLY",
        "EXECUTION",
        "OBSERVED_MATCH",
        "EXTRAPOLATION",
        "COMPACTNESS",
    ]
    return [
        {
            "stage": s,
            "status": "PASSED",
            "duration_ms": 1.2,
            "reason_code": None,
            "message": None,
            "evidence": {},
        }
        for s in stages
    ]


@pytest.fixture
def valid_evaluation_data() -> dict:
    h = "sha256:" + "0" * 64
    proto = {
        "schema_version": "1.0",
        "protocol_id": h,
        "checkpoint_sha256": h,
        "benchmark_manifest_sha256": h,
        "observed_horizon": 20,
        "unseen_horizon": 100,
        "candidate_budget": 1,
        "base_seed": 42,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 128,
        "constant_resolution": True,
        "solver_timeout_ms": 250,
        "max_placeholders": 4,
        "fuel_per_invocation": 10000,
        "memory_limit_mib": 16,
        "mdl_ratio_max": 1.2,
        "native_evaluator_required": True,
        "code_revision": "test",
        "environment_fingerprint": h,
    }

    checkpoint = {
        "format_version": "2.0",
        "checkpoint_sha256": h,
        "producer_version": "oeis-learn-0.1.0",
        "epoch": 60,
        "precision": "fp32",
        "encoder_config": {"d_model": 64},
        "decoder_config": {"d_model": 64},
        "vocabulary_sha256": h,
        "source_checkpoint_sha256": None,
        "runtime_environment": {},
    }

    target = {
        "cohort_id": "trustworthy_synthesis_v1",
        "oeis_id": "A000290",
        "offset": 0,
        "terms_sha256": h,
        "result_profile": "i64_scalar_v1",
    }

    candidate = {
        "candidate_id": "cand_0",
        "candidate_index": 0,
        "candidate_seed": 123456789,
        "raw_token_ids": [1, 2, 3],
        "raw_wat": "(module ...)",
        "resolved_wat": "(module ...)",
        "resolved_constants": [],
        "canonical_wat": "(module ...)",
        "canonical_sha256": h,
        "duplicate_of": None,
        "stage_records": make_valid_stage_records(),
        "outputs": [str(i * i) for i in range(120)],
        "max_fuel": 260,
        "total_fuel": 31200,
        "peak_memory_mib": 0.5,
        "first_observed_divergence": None,
        "first_unseen_divergence": None,
        "byte_size": 48,
        "mdl_ratio": 0.25,
        "classification": "EXTRAPOLATING_SUCCESS",
        "primary_failure_stage": None,
        "secondary_diagnostics": [],
    }

    return {
        "schema_version": "1.0",
        "evaluation_id": "eval_A000290_b1",
        "created_at": "2026-09-04T12:00:00Z",
        "protocol": proto,
        "checkpoint": checkpoint,
        "target": target,
        "candidates": [candidate],
        "unique_candidate_count": 1,
        "qualified_candidate_ids": ["cand_0"],
        "status": "QUALIFIED_SUCCESS",
        "duration_ms": 15.4,
    }


def test_valid_evaluation_conforms(valid_evaluation_data, validate_contract):
    validate_contract(valid_evaluation_data, "synthesis-evaluation")


def test_failed_candidate_requires_primary_failure_stage(valid_evaluation_data, validate_contract):
    d = copy.deepcopy(valid_evaluation_data)
    cand = d["candidates"][0]
    cand["classification"] = "FAILED"
    cand["primary_failure_stage"] = None  # Invalid: must be stage_name
    with pytest.raises(ValidationError):
        validate_contract(d, "synthesis-evaluation")

    cand["primary_failure_stage"] = "OBSERVED_MATCH"
    validate_contract(d, "synthesis-evaluation")


def test_extrapolating_success_requires_120_outputs(valid_evaluation_data, validate_contract):
    d = copy.deepcopy(valid_evaluation_data)
    d["candidates"][0]["outputs"] = [str(i) for i in range(50)]
    with pytest.raises(ValidationError):
        validate_contract(d, "synthesis-evaluation")


def test_extrapolating_success_forbids_failure_stage(valid_evaluation_data, validate_contract):
    d = copy.deepcopy(valid_evaluation_data)
    d["candidates"][0]["primary_failure_stage"] = "OBSERVED_MATCH"
    with pytest.raises(ValidationError):
        validate_contract(d, "synthesis-evaluation")
