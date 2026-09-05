"""Unit tests for downstream token zero-masking and advantage conservation in EGCA."""

import pytest
from oeis_learn.data.models import ExecutionResult
from oeis_learn.sandbox.tracer import build_fine_grained_attribution, classify_failure_mode


def test_classify_failure_modes():
    target = [0, 1, 2, 3, 4, 5]

    res_syntax = ExecutionResult(status="PARSE_ERROR", consumed_fuel=0, output=[])
    assert classify_failure_mode(res_syntax, target) == "SYNTAX"

    res_trap = ExecutionResult(status="OUT_OF_FUEL", consumed_fuel=10000, output=[0, 1])
    assert classify_failure_mode(res_trap, target) == "CONSTRAINT"

    res_logic = ExecutionResult(status="SUCCESS", consumed_fuel=50, output=[0, 1, 2, 0, 0, 0])
    assert classify_failure_mode(res_logic, target) == "LOGIC"

    res_correct = ExecutionResult(status="SUCCESS", consumed_fuel=50, output=[0, 1, 2, 3, 4, 5])
    assert classify_failure_mode(res_correct, target) == "CORRECT"


def test_downstream_token_zero_masking_and_advantage_conservation():
    wat_code = '(module (func (export "compute") (param $n i32) (result i64) (local $a i64) local.get $n i64.extend_i32_s i64.const 2 i64.mul))'
    target = [0, 2, 4, 7, 8, 10]  # Diverges at index 3
    res = ExecutionResult(status="SUCCESS", consumed_fuel=100, output=[0, 2, 4, 6, 8, 10], divergence_step=3)

    total_adv = -1.2
    total_tokens = 30

    attr = build_fine_grained_attribution(
        wat_code=wat_code,
        exec_result=res,
        target_terms=target,
        total_advantage=total_adv,
        total_tokens=total_tokens,
    )

    assert attr.failure_mode == "LOGIC"
    adv_vec = attr.token_advantage_mask
    assert len(adv_vec) == total_tokens

    # Downstream tokens after causal span must be zero-masked
    for t in range(attr.causal_token_end, total_tokens):
        assert adv_vec[t] == 0.0

    # Total advantage sum must be conserved (sum a_{i,t} = A_i)
    assert abs(sum(adv_vec) - total_adv) < 1e-4
