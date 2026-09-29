"""Integration test for execution trace divergence localization and basic block coverage masking."""

import pytest
import torch
from oeis_learn.data.models import ExecutionResult
from oeis_learn.sandbox.tracer import build_fine_grained_attribution, extract_token_coverage


def test_token_coverage_identifies_unreachable_code():
    wat_with_unreachable = '(module (func (export "compute") (param $n i32) (result i64) return nop nop i64.const 5))'
    cov = extract_token_coverage(wat_with_unreachable)

    # Tokens after 'return' before closing ')' must be flagged unreachable (False)
    assert not all(cov)
    assert cov[0] is True  # module header is executed


def test_egca_concentrates_gradient_mass_on_causal_window():
    wat_code = '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_s i64.const 3 i64.mul))'
    target = [0, 2, 4, 6, 8]  # candidate multiplies by 3 instead of 2 -> diverges at step 1
    res = ExecutionResult(status="SUCCESS", consumed_fuel=100, output=[0, 3, 6, 9, 12], divergence_step=1)

    attr = build_fine_grained_attribution(
        wat_code=wat_code,
        exec_result=res,
        target_terms=target,
        total_advantage=-1.0,
        total_tokens=25,
    )

    adv = attr.token_advantage_mask
    causal_mass = sum(abs(adv[t]) for t in range(attr.causal_token_start, attr.causal_token_end))
    total_mass = sum(abs(a) for a in adv)

    assert total_mass > 0.0
    # >= 90% of gradient mass must be concentrated on causal error span
    assert (causal_mass / total_mass) >= 0.90
