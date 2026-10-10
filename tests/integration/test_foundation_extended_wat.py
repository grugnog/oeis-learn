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
