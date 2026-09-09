"""Integration tests comparing decoupled three-tier symbolic grounding against SMT bit-blasting."""

from __future__ import annotations

import time
import pytest
from oeis_learn.decoder.constant_solver import parse_ast_placeholders, solve_constants
from oeis_learn.sandbox.runner import WasmRunner


def test_three_tier_grounding_linear_affine():
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
    runner = WasmRunner(fuel_budget=10000)

    start_t = time.perf_counter()
    cand = solve_constants(skeleton, target_terms, runner=runner)
    dur_ms = (time.perf_counter() - start_t) * 1000.0

    assert cand.is_sat is True
    assert cand.solver_tier == "TIER2_DIXON_LIFTING"
    assert cand.constants == [5, 2]
    assert cand.certificate is not None
    assert cand.certificate.status == "CONSISTENT"
    assert cand.certificate.penalty_reward == 0.0

    # Verify execution of grounded program
    assert cand.grounded_wat is not None
    exec_res = runner.run_single(cand.grounded_wat, terms_to_generate=20)
    assert exec_res.status == "SUCCESS"
    assert exec_res.output == target_terms


def test_three_tier_grounding_fast_inconsistency_rejection():
    # Target terms don't match linear affine model (e.g. exponential 2^n)
    target_terms = [1 << n for n in range(20)]
    wat_skeleton = """(module
  (func (export "compute") (param $n i32) (result i64)
    (i64.add
      (i64.mul (i64.extend_i32_u (local.get $n)) i64.const_?)
      i64.const_?
    )
  )
)"""
    skeleton = parse_ast_placeholders(wat_skeleton)
    runner = WasmRunner(fuel_budget=10000)

    start_t = time.perf_counter()
    cand = solve_constants(skeleton, target_terms, runner=runner)
    dur_ms = (time.perf_counter() - start_t) * 1000.0

    assert cand.is_sat is False
    assert cand.solver_tier == "TIER1_MODULAR_M61"
    assert cand.certificate is not None
    assert cand.certificate.status in ("INCONSISTENT", "UNDERDETERMINED")
    assert cand.certificate.penalty_reward == -0.50
    # Fast modular rejection should take only a fraction of a millisecond for filter logic
    assert cand.certificate.elapsed_microseconds < 1000.0
