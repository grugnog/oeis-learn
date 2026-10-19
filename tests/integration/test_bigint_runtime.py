"""The real native compiler and two-instance WASM path, never a Python substitute."""

import json
import math
import os
from pathlib import Path
import subprocess

import pytest

PROBE = Path(os.environ.get("OEIS_BIGINT_PROBE", "/opt/oeis-build-info/runtime-probe"))


def native(source, indices, **limits):
    if not PROBE.is_file():
        if os.environ.get("OEIS_REQUIRE_BIGINT_BUILD") == "1":
            pytest.fail("required native runtime probe missing")
        pytest.skip("native runtime probe requires qualification image")
    request = {"source": source, "indices": [str(n) for n in indices], "limits": limits}
    run = subprocess.run(
        [str(PROBE)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        timeout=40,
        check=True,
    )
    return json.loads(run.stdout)


def test_factorial_100_and_fresh_term_state():
    source = """(program (local $b0 bigint) (local $c0 control)
      (set $b0 (int 1)) (loop $c0 (to_control $index)
      (set $b0 (mul (get $b0) (to_bigint (increment (get $c0))))))
      (return (get $b0)))"""
    result = native(source, range(100))
    assert result["outcome"] == "ok", result
    assert result["values"] == [str(math.factorial(n)) for n in range(100)]


@pytest.mark.parametrize("bits", [256, 512, 1024, 4096])
def test_full_product_and_signed_offset_indices(bits):
    a, b = (1 << bits) - 1, -(1 << (bits - 1)) + 1
    source = f"(program (return (add (mul (int {a}) (int {b})) $index)))"
    indices = [-19, 0, 37]
    result = native(source, indices)
    assert result["values"] == [str(a * b + n) for n in indices], result


@pytest.mark.parametrize(
    "limits,reason",
    [
        ({"term_fuel": 1}, "term_fuel"),
        ({"aggregate_fuel": 1}, "aggregate_fuel"),
        ({"arena_entries": 1}, "arena_limit"),
        ({"output_bytes": 1}, "output_limit"),
    ],
)
def test_failure_and_healthy_recovery(limits, reason):
    result = native("(program (return (int 65536)))", [0], **limits)
    assert result["reason"] == reason, result
    assert native("(program (return $index))", [-7])["values"] == ["-7"]


def test_exact_borderline_product_not_rejected_by_loose_estimate():
    source = "(program (return (mul (int 128) (int 1))))"
    assert native(source, [0], input_bits=8, literal_bits=8, intermediate_bits=8)["values"] == [
        "128"
    ]


@pytest.mark.parametrize(
    "source",
    [
        '(module (memory 1) (func (export "compute") (param i32) (result i32) i32.const 1))',
        "(program (return (host_status)))",
        "(program (return (int -0)))",
        "(program (return (param.get p0)))",
        "(program (return (get $b0)))",
        "(program (return (control 1)))",
    ],
)
def test_arbitrary_wat_privileged_and_forged_handles_never_lower(source):
    assert native(source, [0])["reason"] == "invalid_source"


def test_memory_limiter_is_classified_not_guessed_from_trap():
    result = native("(program (return $index))", [0], memory_bytes=65536, arena_bytes=65536)
    assert result["reason"] == "memory_limit", result


def test_reference_agrees_on_deterministic_differential_corpus():
    from oeis_learn.synthesis.reference import evaluate
    import random

    rng = random.Random(20261018)
    for _ in range(12):
        a, b = rng.randrange(-(10**40), 10**40), rng.randrange(-(10**40), 10**40)
        source = f"(program (return (sub (mul (add $index (int {a})) (int {b})) $index)))"
        indices = ["-19", "0", "37"]
        expected = evaluate(source, indices)
        actual = native(source, indices)
        assert expected.outcome == actual["outcome"] == "ok"
        assert actual["values"] == list(expected.values)


@pytest.mark.parametrize(
    "source,limits,reason",
    [
        ("(program (return (to_bigint (to_control (int -1)))))", {}, "control_range"),
        (
            "(program (local $c0 control) (loop $c0 (control 1025) (if (true) (then) (else))) (return $index))",
            {},
            "loop_limit",
        ),
        ("(program (return $index))", {"evaluation_seconds": 1e-9}, "deadline"),
        ("(program (return $index))", {"term_fuel": 0}, "unsupported_profile"),
        (f"(program (return (int {1 << 4096})))", {}, "literal_limit"),
        (
            "(program (return (mul (int 128) (int 128))))",
            {"input_bits": 8, "literal_bits": 8, "intermediate_bits": 8},
            "integer_limit",
        ),
    ],
)
def test_classified_limits_and_no_partial_numeric_results(source, limits, reason):
    result = native(source, [0], **limits)
    assert result["reason"] == reason and result["values"] == [], result


def test_every_frozen_arithmetic_fixture():
    fixtures = json.loads(
        (Path(__file__).parents[1] / "fixtures/bigint/arithmetic.json").read_text()
    )["cases"]
    for case in fixtures:
        op = case["operation"]
        if op == "factorial":
            continue  # covered through all 100 indices above
        if op == "compare":
            for a, b, expected in zip(
                case["values"], case["values"][1:], case["adjacent_comparisons"]
            ):
                source = f"(program (return (select (lt (int {a}) (int {b})) (int -1) (int 0))))"
                assert native(source, [0])["values"] == [str(expected)]
            continue
        operands = case.get("operands", [])
        if op == "multiply":
            expr = f"(mul (int {operands[0]}) (int {operands[1]}))"
        elif op == "subtract":
            expr = f"(sub (int {operands[0]}) (int {operands[1]}))"
        elif op == "multiply_then_subtract":
            expr = f"(sub (mul (int {operands[0]}) (int {operands[1]})) (int {operands[2]}))"
        elif op == "affine":
            expr = f"(add (mul (int {case['coefficient']}) $index) (int {case['offset']}))"
        else:
            raise AssertionError(f"uncovered fixture operation {op}")
        expected = case["expected"] if isinstance(case["expected"], list) else [case["expected"]]
        assert (
            native(f"(program (return {expr}))", case.get("indices", [0]))["values"] == expected
        ), case["id"]
