"""Contract tests for symbolic grounding pipeline schema conformance."""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator, Draft7Validator, ValidationError
from oeis_learn.data.models import GroundedCandidate, ModularFilterCertificate


def test_symbolic_grounding_contract_valid(load_schema):
    schema = load_schema("symbolic-grounding.schema.json")
    validator = Draft7Validator(schema)

    cert = ModularFilterCertificate(
        status="CONSISTENT",
        prime=2305843009213693951,
        augmented_rank=2,
        coefficient_rank=2,
        unknown_count=2,
        elapsed_microseconds=120.5,
        penalty_reward=0.0,
    )
    candidate = GroundedCandidate(
        skeleton_id="skel_fib_001",
        constants=[1, 1],
        solver_tier="TIER2_DIXON_LIFTING",
        is_sat=True,
        solve_duration_ms=0.45,
        grounded_wat="(module ...)",
        certificate=cert,
    )

    data = {
        "skeleton_id": candidate.skeleton_id,
        "raw_wat": "(module (func ... i64.const_?))",
        "placeholder_count": 2,
        "is_linear": True,
        "certificate": cert.to_dict(),
        "is_sat": candidate.is_sat,
        "solver_tier": candidate.solver_tier,
        "constants": candidate.constants,
        "solve_duration_ms": candidate.solve_duration_ms,
        "grounded_wat": candidate.grounded_wat,
    }

    validator.validate(data)


def test_symbolic_grounding_contract_invalid_tier(load_schema):
    schema = load_schema("symbolic-grounding.schema.json")
    validator = Draft7Validator(schema)

    cert = {
        "status": "CONSISTENT",
        "prime": 2305843009213693951,
        "augmented_rank": 2,
        "coefficient_rank": 2,
        "unknown_count": 2,
        "elapsed_microseconds": 120.5,
        "penalty_reward": 0.0,
    }
    data = {
        "skeleton_id": "skel_fib_001",
        "raw_wat": "(module ...)",
        "placeholder_count": 2,
        "is_linear": True,
        "certificate": cert,
        "is_sat": True,
        "solver_tier": "INVALID_TIER",
        "constants": [1, 1],
        "solve_duration_ms": 0.45,
        "grounded_wat": "(module ...)",
    }

    with pytest.raises(ValidationError):
        validator.validate(data)
