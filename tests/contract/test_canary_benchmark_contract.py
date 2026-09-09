"""Contract tests for canary benchmark schema conformance."""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator, ValidationError


def test_canary_benchmark_schema_valid(load_schema):
    schema = load_schema("canary-benchmark.schema.json")
    validator = Draft7Validator(schema)

    valid_report = {
        "evaluation_timestamp": "2026-09-06T12:00:00Z",
        "checkpoint_evaluated": "runs/011_multilimb_256bit_production/checkpoints/sft_bridge_ready.pt",
        "result_profile": "i256x4_v1",
        "canary_results": [
            {
                "sequence_id": "A000217",
                "name": "Triangular numbers",
                "scalar_64bit_overflow_term": 6074001000,
                "observed_matched": True,
                "unseen_matched": True,
                "overflow_prevented": True,
                "fuel_consumed": 250,
                "verdict": "EXTRAPOLATING_SUCCESS",
            },
            {
                "sequence_id": "A000290",
                "name": "The squares",
                "scalar_64bit_overflow_term": 3037000500,
                "observed_matched": True,
                "unseen_matched": True,
                "overflow_prevented": True,
                "fuel_consumed": 210,
                "verdict": "EXTRAPOLATING_SUCCESS",
            },
            {
                "sequence_id": "A000079",
                "name": "Powers of 2",
                "scalar_64bit_overflow_term": 63,
                "observed_matched": True,
                "unseen_matched": True,
                "overflow_prevented": True,
                "fuel_consumed": 4960,
                "verdict": "EXTRAPOLATING_SUCCESS",
            },
            {
                "sequence_id": "A000045",
                "name": "Fibonacci numbers",
                "scalar_64bit_overflow_term": 93,
                "observed_matched": True,
                "unseen_matched": True,
                "overflow_prevented": True,
                "fuel_consumed": 9331,
                "verdict": "EXTRAPOLATING_SUCCESS",
            },
            {
                "sequence_id": "A000032",
                "name": "Lucas numbers",
                "scalar_64bit_overflow_term": 91,
                "observed_matched": True,
                "unseen_matched": True,
                "overflow_prevented": True,
                "fuel_consumed": 9331,
                "verdict": "EXTRAPOLATING_SUCCESS",
            },
            {
                "sequence_id": "A000129",
                "name": "Pell numbers",
                "scalar_64bit_overflow_term": 51,
                "observed_matched": True,
                "unseen_matched": True,
                "overflow_prevented": True,
                "fuel_consumed": 9200,
                "verdict": "EXTRAPOLATING_SUCCESS",
            },
        ],
        "all_canaries_passed": True,
    }

    validator.validate(valid_report)


def test_canary_benchmark_schema_invalid_count(load_schema):
    schema = load_schema("canary-benchmark.schema.json")
    validator = Draft7Validator(schema)

    invalid_report = {
        "evaluation_timestamp": "2026-09-06T12:00:00Z",
        "checkpoint_evaluated": "test.pt",
        "result_profile": "i256x4_v1",
        "canary_results": [],  # Must have exactly 6
        "all_canaries_passed": False,
    }

    with pytest.raises(ValidationError):
        validator.validate(invalid_report)
