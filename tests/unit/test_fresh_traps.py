from oeis_learn.sandbox.pipeline import Limits, Runtime


def test_wasmtime20_fuel_trap_is_a_result_not_enum_exception():
    result = Runtime().evaluate("i256.zero", [0], Limits(fuel_per_term=1))
    assert result.outcome == "execution_limit"
    assert result.reason == "wasmtime_fuel"
    assert result.fuel == 1


def test_wasmtime20_divide_by_zero_is_distinct_from_fuel_exhaustion():
    result = Runtime().evaluate(
        "i64.const 1 i64.const 0 i64.div_s i64.const 0 i64.const 0 i64.const 0", [0]
    )
    assert result.outcome == "runtime_failure"
    assert "INTEGER_DIVISION_BY_ZERO" in result.reason
