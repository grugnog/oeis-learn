"""Contract tests for symbolic grounding pipeline schema conformance."""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator, ValidationError

def test_symbolic_grounding_contract_valid(load_schema):
    # Validate the unchanged historical envelope, not a live fitting API.
    schema=load_schema('symbolic-grounding.schema.json')
    data=dict(skeleton_id='historical',raw_wat='(module ...)',placeholder_count=2,is_linear=True,
              certificate=dict(status='CONSISTENT',prime=2305843009213693951,augmented_rank=2,coefficient_rank=2,unknown_count=2,elapsed_microseconds=120.5,penalty_reward=0.0),
              is_sat=True,solver_tier='TIER2_DIXON_LIFTING',constants=[1,1],solve_duration_ms=0.45,grounded_wat='(module ...)')
    Draft7Validator(schema).validate(data)
    from oeis_learn.synthesis.historical import inspect_record
    assert inspect_record(data)['status']=='historical_unreplayed'


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
