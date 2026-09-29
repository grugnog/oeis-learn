"""Contract tests for static arithmetic preamble loading, verification, and limits."""

from __future__ import annotations

import pytest
from oeis_learn.sandbox.preamble import (
    load_preamble_wat,
    get_preamble_fuel_costs,
    verify_zero_memory,
    get_static_preamble,
)
from oeis_learn.sandbox.runner import WasmRunner


def test_preamble_wat_contents():
    wat = load_preamble_wat()
    assert "(func $mul64_wide" in wat
    assert "(func $i256_add" in wat
    assert "(func $i256_sub" in wat
    assert "(func $i256_mul_scalar" in wat


def test_preamble_zero_memory_enforced():
    wat = load_preamble_wat()
    assert verify_zero_memory(wat) is True
    # Test that a module with memory is rejected
    assert verify_zero_memory("(module (memory 1))") is False


def test_preamble_fuel_costs_contract():
    costs = get_preamble_fuel_costs()
    assert costs["i256_add"] == 55
    assert costs["i256_sub"] == 56
    assert costs["mul64_wide"] == 60
    assert costs["i256_mul_scalar"] == 271


def test_static_preamble_entity():
    preamble = get_static_preamble()
    assert preamble.version == "1.0.0"
    assert preamble.memory_limit_bytes == 0
    assert len(preamble.wat_code) > 0


def test_preamble_execution_addition():
    runner = WasmRunner(fuel_budget=10000)
    preamble_wat = load_preamble_wat()
    # Strip module wrap
    inner = preamble_wat[preamble_wat.find("(func"):preamble_wat.rfind(")")]
    full_module = f"""(module
      {inner}
      (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
        i64.const 10 i64.const 0 i64.const 0 i64.const 0
        i64.const 32 i64.const 0 i64.const 0 i64.const 0
        call $i256_add
      )
    )"""
    res = runner.run_single(full_module, terms_to_generate=1, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    assert res.output == [42]
