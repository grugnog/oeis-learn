"""Integration tests for exact 120-term recurrence qualification and state rotation."""

from __future__ import annotations

import pytest
from oeis_learn.curriculum.extrapolation import ExtrapolationVerifier
from oeis_learn.sandbox.runner import WasmRunner


def test_fibonacci_canary_recurrence_qualification():
    runner = WasmRunner(fuel_budget=10000)
    # Correct Fibonacci loop computing first 30 terms
    fib_wat = """(module
  (func (export "compute") (param $n i32) (result i64)
    (local $a i64) (local $b i64) (local $temp i64) (local $i i32)
    (local.set $a (i64.const 0))
    (local.set $b (i64.const 1))
    (local.set $i (i32.const 0))
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
        (local.set $temp (i64.add (local.get $a) (local.get $b)))
        (local.set $a (local.get $b))
        (local.set $b (local.get $temp))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $a)
  )
)"""
    res = runner.run_single(fib_wat, terms_to_generate=30)
    assert res.status == "SUCCESS"

    fib_truth = [0, 1]
    for _ in range(2, 30):
        fib_truth.append(fib_truth[-1] + fib_truth[-2])

    assert res.output == fib_truth


def test_incomplete_rotation_fails_extrapolation():
    runner = WasmRunner(fuel_budget=10000)
    # Incorrect loop: fails to rotate $a = $b properly (drops $temp or sets $a to $a)
    bad_wat = """(module
  (func (export "compute") (param $n i32) (result i64)
    (local $a i64) (local $b i64) (local $temp i64) (local $i i32)
    (local.set $a (i64.const 0))
    (local.set $b (i64.const 1))
    (local.set $i (i32.const 0))
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
        (local.set $temp (i64.add (local.get $a) (local.get $b)))
        (local.set $b (local.get $temp))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $a)
  )
)"""
    res = runner.run_single(bad_wat, terms_to_generate=10)
    assert res.status == "SUCCESS"
    assert res.output != [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_fibonacci_multilimb_100_terms_overflow_free():
    runner = WasmRunner(fuel_budget=10000)
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
    res = runner.run_single(fib_macro, terms_to_generate=100, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    assert len(res.output) == 100

    # Compute ground truth in Python arbitrary precision
    truth = [0, 1]
    for _ in range(2, 100):
        truth.append(truth[-1] + truth[-2])

    assert res.output == truth
    # Specifically check term 93 and term 99 > 2^63 - 1
    assert res.output[93] == 12200160415121876738
    assert res.output[93] > (1 << 63) - 1


def test_lucas_multilimb_100_terms_overflow_free():
    runner = WasmRunner(fuel_budget=10000)
    lucas_macro = """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.const 2 local.set $a3 local.set $a2 local.set $a1 local.set $a0
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
    res = runner.run_single(lucas_macro, terms_to_generate=100, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    assert len(res.output) == 100

    truth = [2, 1]
    for _ in range(2, 100):
        truth.append(truth[-1] + truth[-2])

    assert res.output == truth
    assert res.output[91] > (1 << 63) - 1


def test_powers_of_two_multilimb_100_terms_overflow_free():
    runner = WasmRunner(fuel_budget=10000)
    pow2_macro = """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.const 1 local.set $a3 local.set $a2 local.set $a1 local.set $a0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        i256.add
        local.set $t3 local.set $t2 local.set $t1 local.set $t0
        local.get $t0 local.set $a0 local.get $t1 local.set $a1
        local.get $t2 local.set $a2 local.get $t3 local.set $a3
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)"""
    res = runner.run_single(pow2_macro, terms_to_generate=100, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    assert len(res.output) == 100

    truth = [1 << n for n in range(100)]
    assert res.output == truth
    assert res.output[63] == 1 << 63
    assert res.output[99] == 1 << 99


def test_pell_multilimb_100_terms_overflow_free():
    runner = WasmRunner(fuel_budget=10000)
    pell_macro = """(module
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
        ;; 2*b + a: b + b + a
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        i256.add
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
    res = runner.run_single(pell_macro, fuel_budget=16000, terms_to_generate=100, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    assert len(res.output) == 100

    truth = [0, 1]
    for _ in range(2, 100):
        truth.append(2 * truth[-1] + truth[-2])

    assert res.output == truth
    assert res.output[51] > (1 << 63) - 1
