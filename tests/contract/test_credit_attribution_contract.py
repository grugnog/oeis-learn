"""Contract test for Execution-Grounded Credit Assignment schema."""

import json
import os
import pytest
from oeis_learn.data.models import ExecutionResult
from oeis_learn.sandbox.tracer import build_fine_grained_attribution


def test_credit_attribution_schema_conformance():
    schema_path = "specs/003-algorithmic-generalization-and-credit-assignment/contracts/credit-attribution.schema.json"
    assert os.path.exists(schema_path), f"Schema file not found at {schema_path}"

    with open(schema_path, "r") as f:
        schema = json.load(f)

    assert schema["title"] == "Execution-Grounded Credit Assignment (EGCA) Schema"

    # Build attribution result and verify fields
    wat_code = '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_s i64.const 2 i64.mul))'
    res = ExecutionResult(status="SUCCESS", consumed_fuel=100, output=[0, 2, 4, 6, 8], divergence_step=3)
    target = [0, 2, 4, 7, 8]

    attr = build_fine_grained_attribution(
        wat_code=wat_code,
        exec_result=res,
        target_terms=target,
        total_advantage=-0.8,
        total_tokens=20,
    )

    d = attr.to_dict()
    assert d["failure_mode"] in schema["properties"]["failure_mode"]["enum"]
    assert d["divergence_step"] == 3
    assert len(d["token_advantage_mask"]) == 20
    assert len(d["executed_token_mask"]) == 20
