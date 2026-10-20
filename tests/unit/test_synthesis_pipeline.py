"""Unit tests for the 8-stage synthesis candidate evaluation state machine."""

from __future__ import annotations

import pytest
from dataclasses import replace
from oeis_learn.data.models import BenchmarkTarget
from oeis_learn.evaluation.protocol import EvaluationProtocol
from oeis_learn.evaluation.synthesis import (
    evaluate_candidate_stages,
)


@pytest.fixture
def sample_protocol():
    return EvaluationProtocol.from_dict(
        {
            "schema_version": "1.0",
            "checkpoint_sha256": "sha256:" + "0" * 64,
            "benchmark_manifest_sha256": "sha256:" + "1" * 64,
            "observed_horizon": 20,
            "unseen_horizon": 100,
            "candidate_budget": 1,
            "base_seed": 42,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 128,
            "constant_resolution": True,
            "solver_timeout_ms": 2000,
            "max_placeholders": 4,
            "fuel_per_invocation": 10000,
            "memory_limit_mib": 16,
            "mdl_ratio_max": 1.2,
            "native_evaluator_required": False,
            "code_revision": "test",
            "environment_fingerprint": "sha256:" + "2" * 64,
        }
    )


@pytest.fixture
def sample_target():
    return BenchmarkTarget(
        oeis_id="A000290",
        name="The squares: a(n) = n^2",
        offset=0,
        family="POLYNOMIAL_2",
        curriculum_stage=1,
        observed_terms=[str(i * i) for i in range(20)],
        unseen_terms=[str((i + 20) ** 2) for i in range(100)],
        result_profile="i64_scalar_v1",
        terms_sha256="sha256:" + "3" * 64,
        formula_definition_id=None,
        term_fingerprint="sha256:" + "4" * 64,
        program_fingerprints=[],
        tags=["polynomial"],
    )


def test_state_machine_exact_success(sample_protocol, sample_target):
    # Perfect squares program
    wat_code = """(module
  (func (export "compute") (param $n i32) (result i64)
    local.get $n
    i64.extend_i32_u
    local.get $n
    i64.extend_i32_u
    i64.mul
  )
)"""
    rec = evaluate_candidate_stages(
        candidate_index=0,
        candidate_seed=42,
        raw_token_ids=[1, 2, 3],
        raw_wat=wat_code,
        target=sample_target,
        protocol=sample_protocol,
        seen_canonical_hashes={},
    )

    assert rec.classification == "EXTRAPOLATING_SUCCESS"
    assert rec.primary_failure_stage is None
    assert len(rec.outputs) == 120
    assert len(rec.stage_records) == 8
    for sr in rec.stage_records:
        assert sr.status in ("PASSED", "NOT_REQUIRED")


def test_legacy_native_request_cannot_silently_use_python(sample_protocol, sample_target):
    from dataclasses import replace

    with pytest.raises(ValueError, match="G7-activated"):
        evaluate_candidate_stages(
            0,
            42,
            [1],
            "i256.const 7",
            sample_target,
            replace(sample_protocol, native_evaluator_required=True),
            {},
        )


def test_state_machine_assembly_failure(sample_protocol, sample_target):
    # Exercise assembly itself. The shared grounding service also parses and
    # verifies no-placeholder candidates when constant resolution is enabled.
    sample_protocol = replace(sample_protocol, constant_resolution=False)
    malformed_wat = "(module (func broken syntax"
    rec = evaluate_candidate_stages(
        candidate_index=0,
        candidate_seed=42,
        raw_token_ids=[1, 2],
        raw_wat=malformed_wat,
        target=sample_target,
        protocol=sample_protocol,
        seen_canonical_hashes={},
    )
    assert rec.classification == "FAILED"
    assert rec.primary_failure_stage == "ASSEMBLY"
    stages_dict = {sr.stage: sr.status for sr in rec.stage_records}
    assert stages_dict["ASSEMBLY"] == "FAILED"
    assert stages_dict["EXECUTION"] == "NOT_RUN"


def test_state_machine_observed_mismatch(sample_protocol, sample_target):
    sample_protocol = replace(sample_protocol, constant_resolution=False)
    # Returns n instead of n^2
    wrong_wat = """(module
  (func (export "compute") (param $n i32) (result i64)
    local.get $n
    i64.extend_i32_u
  )
)"""
    rec = evaluate_candidate_stages(
        candidate_index=0,
        candidate_seed=42,
        raw_token_ids=[1, 2],
        raw_wat=wrong_wat,
        target=sample_target,
        protocol=sample_protocol,
        seen_canonical_hashes={},
    )
    assert rec.classification == "FAILED"
    assert rec.primary_failure_stage == "OBSERVED_MATCH"
    assert rec.first_observed_divergence == 2  # n=2: returns 2, expected 4


def test_state_machine_duplicate_detection(sample_protocol, sample_target):
    wat_code = """(module
  (func (export "compute") (param $n i32) (result i64)
    local.get $n
    i64.extend_i32_u
    local.get $n
    i64.extend_i32_u
    i64.mul
  )
)"""
    seen = {}
    rec1 = evaluate_candidate_stages(
        candidate_index=0,
        candidate_seed=42,
        raw_token_ids=[1],
        raw_wat=wat_code,
        target=sample_target,
        protocol=sample_protocol,
        seen_canonical_hashes=seen,
    )
    assert rec1.classification != "DUPLICATE"

    # Second candidate with identical canonical form
    rec2 = evaluate_candidate_stages(
        candidate_index=1,
        candidate_seed=43,
        raw_token_ids=[1],
        raw_wat=wat_code,
        target=sample_target,
        protocol=sample_protocol,
        seen_canonical_hashes=seen,
    )
    assert rec2.classification == "DUPLICATE"
    assert rec2.duplicate_of == rec1.candidate_id


def test_concrete_legacy_mismatch_is_rejected_without_solver_claims(sample_protocol, sample_target):
    wrong = '(module (func (export "compute") (param $n i32) (result i64) i64.const 0))'
    record = evaluate_candidate_stages(0, 42, [1], wrong, sample_target, sample_protocol, {})
    assert record.classification == "FAILED"
    assert record.primary_failure_stage == 'OBSERVED_MATCH'
    assert record.first_observed_divergence == 1
    stage=next(s for s in record.stage_records if s.stage=='CONSTANT_RESOLUTION')
    assert stage.status=='NOT_REQUIRED'
