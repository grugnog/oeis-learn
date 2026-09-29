"""Contract tests for multi-limb macro instruction lowering and invariants."""

from __future__ import annotations

import pytest
from oeis_learn.sandbox.lowering import lower_macro_wat, get_macro_fuel_cost


def test_macro_instruction_lowering_add():
    wat = "(module (func (export \"compute\") (param $n i32) (result i64 i64 i64 i64) i256.add))"
    lowered = lower_macro_wat(wat)
    assert "call $i256_add" in lowered
    assert "$i256_add" in lowered
    assert "i256.add" not in lowered


def test_macro_instruction_lowering_sub():
    wat = "(module (func (export \"compute\") (param $n i32) (result i64 i64 i64 i64) i256.sub))"
    lowered = lower_macro_wat(wat)
    assert "call $i256_sub" in lowered
    assert "$i256_sub" in lowered
    assert "i256.sub" not in lowered


def test_macro_instruction_lowering_mul_scalar():
    wat = "(module (func (export \"compute\") (param $n i32) (result i64 i64 i64 i64) i256.mul_scalar))"
    lowered = lower_macro_wat(wat)
    assert "call $i256_mul_scalar" in lowered
    assert "$i256_mul_scalar" in lowered
    assert "i256.mul_scalar" not in lowered


def test_macro_instruction_lowering_const_positive():
    wat = "(module (func (export \"compute\") (param $n i32) (result i64 i64 i64 i64) (i256.const 42)))"
    lowered = lower_macro_wat(wat)
    assert "i64.const 42 i64.const 0 i64.const 0 i64.const 0" in lowered


def test_macro_instruction_lowering_const_negative():
    wat = "(module (func (export \"compute\") (param $n i32) (result i64 i64 i64 i64) (i256.const -5)))"
    lowered = lower_macro_wat(wat)
    assert "i64.const -5 i64.const -1 i64.const -1 i64.const -1" in lowered


def test_macro_instruction_lowering_zero():
    wat = "(module (func (export \"compute\") (param $n i32) (result i64 i64 i64 i64) i256.zero))"
    lowered = lower_macro_wat(wat)
    assert "i64.const 0 i64.const 0 i64.const 0 i64.const 0" in lowered


def test_macro_fuel_costs():
    assert get_macro_fuel_cost("i256.add") == 55
    assert get_macro_fuel_cost("i256.sub") == 56
    assert get_macro_fuel_cost("i256.mul_scalar") == 271
    assert get_macro_fuel_cost("i256.const") == 4
    assert get_macro_fuel_cost("i256.zero") == 4


def test_fibonacci_macro_token_budget():
    fib_macro = """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.zero local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i256.const 1 local.set $b3 local.set $b2 local.set $b1 local.set $b0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        i256.add
        local.set $t3 local.set $t2 local.set $t1 local.set $t0
        local.get $b0 local.set $a0 local.get $b1 local.set $a1
        local.get $b2 local.set $a2 local.get $b3 local.set $a3
        local.get $t0 local.set $b0 local.get $t1 local.set $b1
        local.get $t2 local.set $b2 local.get $t3 local.set $b3
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)"""
    lowered = lower_macro_wat(fib_macro)
    assert "$i256_add" in lowered
    assert "call $i256_add" in lowered
    assert "(result i64 i64 i64 i64)" in lowered
