"""Unit tests for exact Hermite Normal Form Diophantine linear integer solver."""

from __future__ import annotations

import time
import pytest
from oeis_learn.decoder.constant_solver import (
    parse_ast_placeholders,
    solve_linear_diophantine,
    splice_constants_into_wat,
)
from oeis_learn.sandbox.runner import WasmRunner


@pytest.fixture
def wasm_runner():
    return WasmRunner(fuel_budget=10000)


def test_linear_affine_diophantine_grounding(wasm_runner):
    # a(n) = 5n + 2 for n=0..19
    # a(n) = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97]
    target_terms = [5 * n + 2 for n in range(20)]

    wat_skeleton = """(module
  (func (export "compute") (param $n i32) (result i64)
    (i64.add
      (i64.mul (i64.extend_i32_u (local.get $n)) i64.const_?)
      i64.const_?
    )
  )
)"""

    skeleton = parse_ast_placeholders(wat_skeleton)
    assert skeleton.placeholder_count == 2
    assert skeleton.is_linear is True

    start_t = time.perf_counter()
    result = solve_linear_diophantine(skeleton, target_terms, wasm_runner)
    duration_ms = (time.perf_counter() - start_t) * 1000.0

    assert result.is_sat is True
    assert result.solver_type == "DIOPHANTINE_HNF"
    assert result.constants == [5, 2]
    assert duration_ms < 100.0  # Well under safety budget, targeting sub-millisecond in batch

    # Verify spliced WAT execution
    grounded_wat = splice_constants_into_wat(skeleton, result.constants)
    exec_res = wasm_runner.run_single(grounded_wat, terms_to_generate=20)
    assert exec_res.status == "SUCCESS"
    assert exec_res.output == target_terms


def test_quadratic_polynomial_diophantine_grounding(wasm_runner):
    # a(n) = 3n^2 - 4n + 7
    target_terms = [3 * n * n - 4 * n + 7 for n in range(20)]

    wat_skeleton = """(module
  (func (export "compute") (param $n i32) (result i64)
    (local $n64 i64)
    (local.set $n64 (i64.extend_i32_u (local.get $n)))
    (i64.add
      (i64.add
        (i64.mul (i64.mul (local.get $n64) (local.get $n64)) i64.const_?)
        (i64.mul (local.get $n64) i64.const_?)
      )
      i64.const_?
    )
  )
)"""

    skeleton = parse_ast_placeholders(wat_skeleton)
    assert skeleton.placeholder_count == 3
    assert skeleton.is_linear is True

    result = solve_linear_diophantine(skeleton, target_terms, wasm_runner)
    assert result.is_sat is True
    assert result.solver_type == "DIOPHANTINE_HNF"
    assert result.constants == [3, -4, 7]

    grounded_wat = splice_constants_into_wat(skeleton, result.constants)
    exec_res = wasm_runner.run_single(grounded_wat, terms_to_generate=20)
    assert exec_res.status == "SUCCESS"
    assert exec_res.output == target_terms


def test_underdetermined_minimal_l1_norm(wasm_runner):
    # a(n) = 4n: skeleton has c_1*n + c_2*n + c_0
    target_terms = [4 * n for n in range(20)]

    wat_skeleton = """(module
  (func (export "compute") (param $n i32) (result i64)
    (i64.add
      (i64.add
        (i64.mul (i64.extend_i32_u (local.get $n)) i64.const_?)
        (i64.mul (i64.extend_i32_u (local.get $n)) i64.const_?)
      )
      i64.const_?
    )
  )
)"""

    skeleton = parse_ast_placeholders(wat_skeleton)
    result = solve_linear_diophantine(skeleton, target_terms, wasm_runner)
    assert result.is_sat is True
    # Should find integers summing to 4 with minimal L1 norm
    assert sum(result.constants[:2]) == 4
    assert result.constants[2] == 0


def test_inconsistent_linear_diophantine(wasm_runner):
    # Target is non-linear (powers of 2), but skeleton is linear
    target_terms = [2**n for n in range(20)]

    wat_skeleton = """(module
  (func (export "compute") (param $n i32) (result i64)
    (i64.add
      (i64.mul (i64.extend_i32_u (local.get $n)) i64.const_?)
      i64.const_?
    )
  )
)"""

    skeleton = parse_ast_placeholders(wat_skeleton)
    result = solve_linear_diophantine(skeleton, target_terms, wasm_runner)
    assert result.is_sat is False
    assert result.constants is None
