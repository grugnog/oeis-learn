"""Independent expectations: never obtain answers from native lowering."""

import math
from dataclasses import replace

import pytest

from oeis_learn.synthesis.contracts import ResourceProfile
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.synthesis.reference_parser import parse, SourceError

FACTORIAL = """(program (local $b0 bigint) (local $c0 control)
 (set $b0 (int 1))
 (loop $c0 (to_control $index)
   (set $b0 (mul (get $b0) (to_bigint (increment (get $c0))))))
 (return (get $b0)))"""


def test_factorial_actual_indices_and_fresh_state():
    result = evaluate(FACTORIAL, [str(n) for n in range(100)])
    assert result.outcome == "ok"
    assert result.values == tuple(str(math.factorial(n)) for n in range(100))
    assert evaluate(FACTORIAL, ["5", "0", "3"]).values == ("120", "1", "6")


def test_exact_negative_and_offset_indices():
    assert evaluate("(program (return (mul $index $index)))", ["-7", "17"]).values == ("49", "289")
    assert evaluate(FACTORIAL, ["-1"]).reason == "control_range"


def test_cancellation_does_not_hide_intermediate_limit():
    n = str(1 << 100)
    source = f"(program (return (sub (mul (int {n}) (int {n})) (mul (int {n}) (int {n})))))"
    limits = replace(
        ResourceProfile(), input_bits=128, literal_bits=128, intermediate_bits=128, output_bytes=16
    )
    assert evaluate(source, ["0"], limits).reason == "integer_limit"


def test_steps_aggregate_deadline_and_recovery():
    source = "(program (return (add $index (int 1))))"
    assert (
        evaluate(source, ["0"], replace(ResourceProfile(), reference_steps=1)).reason
        == "reference_limit"
    )
    assert (
        evaluate(source, ["0", "1"], replace(ResourceProfile(), aggregate_reference_steps=4)).reason
        == "reference_limit"
    )
    assert evaluate(source, ["0"], deadline=0).reason == "deadline"
    assert evaluate(source, ["0"]).values == ("1",)


@pytest.mark.parametrize(
    "source",
    [
        "(program (return (int -0)))",
        "(program (return (int 01)))",
        "(program (return (div (int 1) (int 2))))",
        "(program (return (get $b0)))",
        "(program (local $b0 bigint) (set $b0 (control 1)) (return (get $b0)))",
        "(program (return (param.get p0)))",
        "(program (return $index)) garbage",
        "(program (return $index)",
        "(program (return (control 1)))",
    ],
)
def test_reference_rejects_invalid_final_sources(source):
    with pytest.raises(SourceError):
        parse(source)


def test_eager_select_checks_unselected_operand():
    source = "(program (return (select (true) (int 1) (to_bigint (to_control (int -1))))))"
    assert evaluate(source, ["0"]).reason == "control_range"


def test_lazy_if_and_bounded_loop():
    source = """(program (local $b0 bigint) (local $c0 control)
    (if (lt $index (int 0)) (then (set $b0 (int 1)))
     (else (loop $c0 (to_control $index) (set $b0 (add (get $b0) (int 1))))))
    (return (get $b0)))"""
    assert evaluate(source, ["-1", "4"]).values == ("1", "4")
    assert evaluate(source, ["1025"]).reason == "loop_limit"


def test_wrong_production_and_reference_exhaustion_never_agree():
    from oeis_learn.synthesis.reference import compare_final

    source = "(program (return (mul (int 4) (int 4))))"
    assert compare_final(source, ["0"], ["16"], ["4"])[0] == "verification_failure"
    assert (
        compare_final("(program (return (int 4)))", ["0"], ["16"], ["4"])[0]
        == "verification_failure"
    )
    assert (
        compare_final("(program (return (int 4)))", ["0"], ["4"], ["4"], deadline=0)[0]
        == "resource_exhausted"
    )
    assert (
        compare_final("(program (return (int 4)))", ["0"], ["4"], ["4"])[0]
        == "independent_agreement"
    )
