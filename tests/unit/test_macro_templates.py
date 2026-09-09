"""Unit tests for curriculum macro-scaffolding templates execution."""

from __future__ import annotations

import math
import pytest
from oeis_learn.curriculum.macro_templates import (
    get_stage1_polynomial_template,
    get_stage2_linear_recurrence_template,
    get_stage3_holonomic_template,
    get_stage4_divisor_loop_template,
)
from oeis_learn.sandbox.runner import WasmRunner


@pytest.fixture
def runner():
    return WasmRunner(fuel_budget=10000)


def test_stage1_polynomial_template(runner):
    wat = get_stage1_polynomial_template(degree=2)
    res = runner.run_single(wat, terms_to_generate=10, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    assert res.output == [n * n for n in range(10)]


def test_stage2_order1_geometric(runner):
    wat = get_stage2_linear_recurrence_template(order=1)
    res = runner.run_single(wat, terms_to_generate=10, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    assert res.output == [1 << n for n in range(10)]


def test_stage2_order2_fibonacci(runner):
    wat = get_stage2_linear_recurrence_template(order=2)
    res = runner.run_single(wat, terms_to_generate=10, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    assert res.output == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_stage2_order3_tribonacci(runner):
    wat = get_stage2_linear_recurrence_template(order=3)
    res = runner.run_single(wat, terms_to_generate=10, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    assert res.output == [0, 0, 1, 1, 2, 4, 7, 13, 24, 44]


def test_stage3_holonomic_factorial(runner):
    wat = get_stage3_holonomic_template()
    res = runner.run_single(wat, terms_to_generate=8, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    expected = [math.factorial(n) for n in range(8)]
    assert res.output == expected


def test_stage4_divisor_count(runner):
    wat = get_stage4_divisor_loop_template()
    res = runner.run_single(wat, terms_to_generate=11, result_profile="i256x4_v1")
    assert res.status == "SUCCESS"
    # Number of divisors for n=1..10
    # For n=0, returns 0
    def num_divisors(n: int) -> int:
        if n <= 0:
            return 0
        return sum(1 for d in range(1, n + 1) if n % d == 0)

    expected = [num_divisors(n) for n in range(11)]
    assert res.output == expected
