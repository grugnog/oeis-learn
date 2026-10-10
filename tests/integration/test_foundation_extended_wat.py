"""Mandatory native logical profile parity in the pinned CPU Docker build."""

import importlib.util
import pytest
from oeis_learn.sandbox.streaming import execute_logical, next_block
from tests.unit.test_foundation_extended_wat import STREAM, MIN, MAX

pytestmark = pytest.mark.docker


@pytest.fixture(autouse=True)
def native_available():
    if importlib.util.find_spec("oeis_wasm_evaluator") is None:
        pytest.skip("native extension requires the CPU Docker qualification image")


@pytest.mark.parametrize("count", [20, 50, 100, 120])
def test_native_stream_pure_reset_resume_parity(count):
    python = next_block(STREAM, count)
    native = next_block(STREAM, count, backend="rust")
    assert native["outcome"] is None, native
    assert native["outputs"] == python["outputs"]
    first = next_block(STREAM, 10, backend="rust")
    rest = next_block(STREAM, count - 10, backend="rust", checkpoint=first["checkpoint"])
    assert first["outputs"] + rest["outputs"] == native["outputs"]


@pytest.mark.parametrize(
    "source",
    [
        f"(pure (return (mul (const {MIN}) (const -1))))",
        f"(pure (return (div (const {MIN}) (const -1))))",
        "(pure (return (rem (const -17) (const 5))))",
        "(pure (return (mul (const 123456789123456789) (const -777777777777777777))))",
        "(pure (return (load (const 1024))))",
        "(pure (store (const 4) (index)) (return (load (const 4))))",
        "(pure (return (shr_u (const -1) 255)))",
        "(pure (return (from_i32 (to_i32 (index)))))",
    ],
)
def test_native_logical_operator_outcome_parity(source):
    a = execute_logical(source, list(range(5)))
    b = execute_logical(source, list(range(5)), backend="rust")
    assert (a["outcome"], a["outputs"]) == (b["outcome"], b["outputs"]), b


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "rem", "and", "or", "xor"])
def test_native_all_binary_operators_and_boundaries(op):
    for a, b in [(MIN, -1), (MAX, 2), (MIN, MIN), (0, 0), (-17, 5), (1 << 200, 1 << 54)]:
        source = f"(pure (return ({op} (const {a}) (const {b}))))"
        result = execute_logical(source, [0], backend="rust")
        python = execute_logical(source, [0])
        assert (result["outcome"], result["outputs"]) == (python["outcome"], python["outputs"]), (
            result
        )


@pytest.mark.parametrize("op", ["eq", "ne", "lt", "le", "gt", "ge"])
def test_native_all_comparisons(op):
    for a, b in [(MIN, MAX), (MAX, MIN), (-1, 0), (1 << 64, 1 << 64)]:
        source = f"(pure (return (select ({op} (const {a}) (const {b})) (const 1) (const 0))))"
        result = execute_logical(source, [0], backend="rust")
        assert (
            result["outcome"] is None
            and result["outputs"] == execute_logical(source, [0])["outputs"]
        ), result


@pytest.mark.parametrize("op", ["shl", "shr_s", "shr_u"])
def test_native_all_shifts(op):
    for count in [0, 1, 63, 64, 127, 128, 255]:
        source = f"(pure (return ({op} (const -17) {count})))"
        result = execute_logical(source, [0], backend="rust")
        python = execute_logical(source, [0])
        assert (result["outcome"], result["outputs"]) == (python["outcome"], python["outputs"]), (
            result
        )


def test_native_arrays_alias_zero_reset_and_failed_transition():
    source = "(pure (if (eq (index) (const 0)) (store (const 1023) (const 17)) (set 0 (const 0))) (store (const 4) (const 3)) (store (const 4) (add (load (const 4)) (const 2))) (return (add (load (const 1023)) (load (const 4)))))"
    result = execute_logical(source, [0, 1, 0, 1], backend="rust")
    assert result["outcome"] is None and result["outputs"] == ["22", "5", "22", "5"], result
    failure = "(stream (init) (next (store (const 1023) (const 17)) (set 0 (const 3)) (return (div (const 1) (const 0)))))"
    first = next_block(failure, 1, backend="rust")
    assert (
        first["outcome"] == "runtime_failure" and first["checkpoint"]["state"]["array"][1023] == 0
    )
    assert (
        first["checkpoint"]["state"]["slots"][0] == 0 and first["checkpoint"]["state"]["index"] == 0
    )
    retry = next_block(failure, 1, checkpoint=first["checkpoint"], backend="rust")
    assert retry["spent_fuel"] > first["spent_fuel"] and retry["outputs"] == []


def test_native_generated_arithmetic_and_structured_control():
    import random

    rng = random.Random(9003)
    for _ in range(12):
        a, b = rng.randrange(MIN, MAX), rng.randrange(MIN, MAX)
        for op in ["div", "rem", "mul"]:
            x, y = (a >> 130, b >> 130) if op == "mul" else (a, b)
            source = f"(pure (return ({op} (const {x}) (const {y}))))"
            result = execute_logical(source, [0], backend="rust")
            quotient = abs(x) // abs(y) * (-1 if (x < 0) != (y < 0) else 1)
            expected = x * y if op == "mul" else quotient if op == "div" else x - quotient * y
            assert result["outcome"] is None and result["outputs"] == [str(expected)], result
    source = "(pure (repeat 3 (set 0 (add (get 0) (const 2)))) (if (gt (index) (const 0)) (set 0 (add (get 0) (index))) (set 0 (get 0))) (return (get 0)))"
    assert execute_logical(source, [0, 1, 2], backend="rust")["outputs"] == ["6", "7", "8"]
    from tests.unit.test_foundation_extended_wat import PURE_FIB

    assert (
        execute_logical(PURE_FIB, list(range(120)), backend="rust")["outputs"]
        == next_block(STREAM, 120, backend="rust")["outputs"]
    )


def test_native_checked_casts_and_resource_limits():
    from oeis_learn.sandbox.streaming import LogicalLimits

    for n in [-(1 << 31), (1 << 31) - 1, -(1 << 31) - 1, 1 << 31]:
        result = execute_logical(
            f"(pure (return (from_i32 (to_i32 (const {n})))))", [0], backend="rust"
        )
        assert result["outcome"] == (None if -(1 << 31) <= n < (1 << 31) else "numeric_limit"), (
            result
        )
    from tests.unit.test_foundation_extended_wat import PURE_FIB

    result = execute_logical(
        PURE_FIB, [119], backend="rust", limits=LogicalLimits(per_call_fuel=1000)
    )
    assert result["outcome"] == "execution_limit" and result["fuel"] > 0, result
