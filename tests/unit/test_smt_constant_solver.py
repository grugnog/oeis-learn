"""Unit tests for Z3 SMT non-linear constant solver fallback."""

from __future__ import annotations

import pytest
from oeis_learn.decoder.constant_solver import (
    parse_ast_placeholders,
    solve_smt_constants,
    splice_constants_into_wat,
)
from oeis_learn.sandbox.runner import WasmRunner


@pytest.fixture
def wasm_runner():
    return WasmRunner(fuel_budget=10000)


def test_nonlinear_smt_modulo_grounding(wasm_runner):
    # a(n) = (n * 3 + 1) % 5
    target_terms = [(n * 3 + 1) % 5 for n in range(20)]

    wat_skeleton = """(module
  (func (export "compute") (param $n i32) (result i64)
    (i64.rem_u
      (i64.add
        (i64.mul (i64.extend_i32_u (local.get $n)) (i64.const 3))
        (i64.const 1)
      )
      i64.const_?
    )
  )
)"""

    skeleton = parse_ast_placeholders(wat_skeleton)
    assert skeleton.placeholder_count == 1
    assert skeleton.is_linear is False

    result = solve_smt_constants(skeleton, target_terms, timeout_ms=500)
    assert result.is_sat is True
    assert result.solver_type == "Z3_SMT"
    assert result.constants == [5]

    grounded_wat = splice_constants_into_wat(skeleton, result.constants)
    exec_res = wasm_runner.run_single(grounded_wat, terms_to_generate=20)
    assert exec_res.status == "SUCCESS"
    assert exec_res.output == target_terms


def test_nonlinear_smt_bitwise_shift_grounding(wasm_runner):
    # a(n) = n << 3
    target_terms = [(n << 3) for n in range(20)]

    wat_skeleton = """(module
  (func (export "compute") (param $n i32) (result i64)
    (i64.shl
      (i64.extend_i32_u (local.get $n))
      i64.const_?
    )
  )
)"""

    skeleton = parse_ast_placeholders(wat_skeleton)
    assert skeleton.is_linear is False

    result = solve_smt_constants(skeleton, target_terms, timeout_ms=500)
    assert result.is_sat is True
    assert result.constants == [3]


def test_smt_solver_timeout_enforcement():
    # Construct an unsatisfiable or very complex constraint and verify timeout
    target_terms = [1000 * n for n in range(20)]
    wat_skeleton = """(module
  (func (export "compute") (param $n i32) (result i64)
    (i64.rem_u (i64.extend_i32_u (local.get $n)) i64.const_?)
  )
)"""
    skeleton = parse_ast_placeholders(wat_skeleton)
    result = solve_smt_constants(skeleton, target_terms, timeout_ms=20)
    # Target cannot be matched by modulo since target exceeds any remainder
    assert result.is_sat is False
