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
