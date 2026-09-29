"""Unit tests for WASM sandbox runner, fuel limits, and error trapping."""

import pytest
from oeis_learn.sandbox.fallback_runner import evaluate_wat_single_fallback
from oeis_learn.sandbox.runner import WasmRunner


@pytest.mark.parametrize("use_fallback", [False, True])
def test_wasm_runner_valid_program(use_fallback):
    runner = WasmRunner(fuel_budget=10000, use_fallback=use_fallback)
    wat = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (i64.mul (i64.extend_i32_s (local.get $n)) (i64.const 3))
        )
    )
    """
    res = runner.run_single(wat, terms_to_generate=5)
    assert res.status == "SUCCESS"
    assert res.output == [0, 3, 6, 9, 12]
    assert 0 < res.consumed_fuel < 10000
    assert res.error is None


@pytest.mark.parametrize("use_fallback", [False, True])
def test_wasm_runner_infinite_loop_trap(use_fallback):
    runner = WasmRunner(fuel_budget=10000, use_fallback=use_fallback)
    wat = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (loop $l (br $l))
            (i64.const 0)
        )
    )
    """
    res = runner.run_single(wat, terms_to_generate=5)
    assert res.status == "OUT_OF_FUEL"
    assert res.consumed_fuel == 10000


@pytest.mark.parametrize("use_fallback", [False, True])
def test_wasm_runner_division_by_zero(use_fallback):
    runner = WasmRunner(fuel_budget=10000, use_fallback=use_fallback)
    wat = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (i64.div_s (i64.const 42) (i64.const 0))
        )
    )
    """
    res = runner.run_single(wat, terms_to_generate=5)
    assert res.status == "EXECUTION_TRAP"
    assert res.error is not None


@pytest.mark.parametrize("use_fallback", [False, True])
def test_wasm_runner_syntax_error(use_fallback):
    runner = WasmRunner(fuel_budget=10000, use_fallback=use_fallback)
    wat = "(module (func (unclosed paren"
    res = runner.run_single(wat)
    assert res.status == "PARSE_ERROR"
