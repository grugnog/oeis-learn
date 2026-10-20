"""Retired wrapping/scalar fallback must fail explicitly without calling a solver."""

import pytest
from oeis_learn.synthesis.fitting import fit_program


@pytest.mark.parametrize(
    "body",
    [
        '(module (func (export "compute") (param $n i32) (result i64) (i64.rem_u (i64.const 17) i64.const_?)))',
        '(module (func (export "compute") (param $n i32) (result i64) (i64.shl (i64.const 1) i64.const_?)))',
        "i256.const_? i64.const_? i256.mul_scalar",
        "(loop $x i64.const_? drop br $x) i64.const 1",
    ],
)
def test_no_wrapping_bitwise_or_control_fallback(body):
    result = fit_program(body, [4] * 20)
    assert result.outcome == "unsupported" and not result.constants
    assert "retired" in result.reason


def test_nonpositive_deadline_rejected():
    with pytest.raises(ValueError):
        fit_program("i256.const 0", [1], timeout_ms=0)
