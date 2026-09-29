"""Contract test for native PyO3 WASM evaluator FFI bindings."""

import oeis_wasm_evaluator
import pytest


def test_ffi_single_evaluation_contract():
    # Triangular numbers: a(n) = n*(n+1)/2
    wat = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (local $n64 i64)
            (local.set $n64 (i64.extend_i32_u (local.get $n)))
            (i64.div_u
                (i64.mul (local.get $n64) (i64.add (local.get $n64) (i64.const 1)))
                (i64.const 2)
            )
        )
    )
    """
    res = oeis_wasm_evaluator.evaluate_wat_single(wat, fuel_budget=10000, terms_to_generate=6)

    assert hasattr(res, "status")
    assert hasattr(res, "consumed_fuel")
    assert hasattr(res, "output")
    assert hasattr(res, "error")

    assert res.status == "SUCCESS"
    assert res.consumed_fuel > 0 and res.consumed_fuel <= 10000
    assert res.output == [0, 1, 3, 6, 10, 15]
    assert res.error is None


def test_ffi_batch_evaluation_contract():
    wat_triangular = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (local $n64 i64)
            (local.set $n64 (i64.extend_i32_u (local.get $n)))
            (i64.div_u
                (i64.mul (local.get $n64) (i64.add (local.get $n64) (i64.const 1)))
                (i64.const 2)
            )
        )
    )
    """
    wat_loop = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (loop $l (br $l))
            (i64.const 0)
        )
    )
    """
    batch = [wat_triangular, wat_loop, "(invalid wat"]
    results = oeis_wasm_evaluator.evaluate_wat_batch(batch, fuel_budget=10000, terms_to_generate=5)

    assert len(results) == 3
    assert results[0].status == "SUCCESS"
    assert results[0].output == [0, 1, 3, 6, 10]
    assert results[1].status == "OUT_OF_FUEL"
    assert results[1].consumed_fuel == 10000
    assert results[2].status == "PARSE_ERROR"
