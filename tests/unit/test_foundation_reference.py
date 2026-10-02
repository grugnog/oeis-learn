"""Hand-calculated native/control examples; expected values never use lowering."""

import time
import pytest
from oeis_learn.sandbox.reference import execute_term
from oeis_learn.sandbox.pipeline import Runtime
from oeis_learn.sandbox.wat_ast import parse, Instruction, Program

# For negative native results we intentionally zero-extend the single limb.
BINARY = [
    ("add", 7, -3, 4),
    ("sub", 7, -3, 10),
    ("mul", -7, 3, -21),
    ("div_s", -7, 3, -2),
    ("rem_s", -7, 3, -1),
    ("div_u", 17, 3, 5),
    ("rem_u", 17, 3, 2),
    ("and", 6, 3, 2),
    ("or", 6, 3, 7),
    ("xor", 6, 3, 5),
    ("shl", 3, 2, 12),
    ("shr_s", -7, 1, -4),
    ("shr_u", 8, 2, 2),
    ("eq", 3, 3, 1),
    ("ne", 3, 4, 1),
    ("lt_s", -3, 2, 1),
    ("gt_s", -3, 2, 0),
    ("le_s", 3, 3, 1),
    ("ge_s", 3, 4, 0),
]
COMPARISONS = {"eq", "ne", "lt_s", "gt_s", "le_s", "ge_s"}


def finish(source, typ="i64"):
    return (
        source
        + (" i64.extend_i32_s" if typ == "i32" else "")
        + " i64.const 0 i64.const 0 i64.const 0"
    )


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("op,a,b,expected", BINARY)
def test_native_binary_hand_vectors(bits, op, a, b, expected):
    typ = f"i{bits}"
    source = f"{typ}.const {a} {typ}.const {b} {typ}.{op}"
    result_type = "i32" if bits == 32 or op in COMPARISONS else "i64"
    body = finish(source, result_type)
    assert execute_term(parse(body), 0).value == expected % 2**64
    result = Runtime().evaluate(body, [0])
    assert result.outcome is None and result.outputs == [str(expected % 2**64)]


@pytest.mark.parametrize(
    "source,expected",
    [
        ("i32.const 0 i32.eqz i64.extend_i32_s", 1),
        ("i64.const -1 i64.eqz i64.extend_i32_s", 0),
        ("i64.const 4294967297 i32.wrap_i64 i64.extend_i32_s", 1),
        ("i32.const -1 i64.extend_i32_u", 4294967295),
        ("i32.const -1 i64.extend_i32_s", 2**64 - 1),
        ("i64.const 9223372036854775807 i64.const 1 i64.add", 2**63),
        ("i32.const 2147483647 i32.const 1 i32.add i64.extend_i32_u", 2**31),
        ("i64.const 1 i64.const 65 i64.shl", 2),
        ("i32.const 1 i32.const -1 i32.shl i64.extend_i32_u", 2**31),
        ("i64.const -1 i64.const 63 i64.shr_u", 1),
        ("i32.const -1 i32.const 31 i32.shr_u i64.extend_i32_s", 1),
        ("i64.const -1 i64.const 2 i64.div_u", 2**63 - 1),
        ("i32.const -1 i32.const 2 i32.rem_u i64.extend_i32_s", 1),
    ],
)
def test_unary_wrap_shifts_unsigned(source, expected):
    assert execute_term(parse(finish(source)), 0).value == expected
    result = Runtime().evaluate(finish(source), [0])
    assert result.outcome is None and result.outputs == [str(expected)]


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("op", ["div_s", "div_u", "rem_s", "rem_u"])
def test_zero_divisor_traps(bits, op):
    source = finish(f"i{bits}.const 1 i{bits}.const 0 i{bits}.{op}", f"i{bits}")
    assert execute_term(parse(source), 0).reason == "integer_divide_by_zero"


@pytest.mark.parametrize("bits", [32, 64])
def test_signed_division_overflow_and_defined_remainder(bits):
    start = f"i{bits}.const {-(2 ** (bits - 1))} i{bits}.const -1"
    assert (
        execute_term(parse(finish(start + f" i{bits}.div_s", f"i{bits}")), 0).outcome
        == "runtime_failure"
    )
    assert execute_term(parse(finish(start + f" i{bits}.rem_s", f"i{bits}")), 0).value == 0


@pytest.mark.parametrize(
    "source,expected",
    [
        ("i256.zero", 0),
        ("i256.const -1", -1),
        ("i256.const 18446744073709551615 i256.const 1 i256.add", 2**64),
        ("i256.const 18446744073709551616 i256.const 1 i256.sub", 2**64 - 1),
        ("i256.const 1 i64.const -9223372036854775808 i256.mul_scalar", -(2**63)),
        ("nop i32.const 9 drop i256.const 7 return unreachable", 7),
        (
            "( block ( result i64 ) i64.const 7 br 0 i64.const 99 ) i64.const 0 i64.const 0 i64.const 0",
            7,
        ),
        (
            "( block ( result i64 ) ( block i64.const 8 br 1 ) i64.const 99 ) i64.const 0 i64.const 0 i64.const 0",
            8,
        ),
        (
            "i32.const 1 ( if ( result i64 ) ( then i64.const 2 ) ( else i64.const 3 ) ) i64.const 0 i64.const 0 i64.const 0",
            2,
        ),
        (
            "i32.const 0 ( if ( result i64 ) ( then i64.const 2 ) ( else i64.const 3 ) ) i64.const 0 i64.const 0 i64.const 0",
            3,
        ),
        (
            "i64.const 7 local.tee $v31 local.get $v31 i64.add i64.const 0 i64.const 0 i64.const 0",
            14,
        ),
        (
            "( loop local.get $c7 i32.const 1 i32.add local.tee $c7 i32.const 3 i32.lt_s br_if 0 ) local.get $c7 i64.extend_i32_s i64.const 0 i64.const 0 i64.const 0",
            3,
        ),
    ],
)
def test_macros_control_and_locals(source, expected):
    assert execute_term(parse(source), 0).value == expected


def test_step_deadline_and_fresh_state():
    loop = parse("( loop br 0 ) i256.zero")
    result = execute_term(loop, 0, step_limit=11)
    assert result.steps == 11 and result.reason == "reference_steps"
    assert execute_term(loop, 0, deadline_ns=time.monotonic_ns() - 1).reason == "reference_deadline"
    body = parse(finish("local.get $v0 i64.const 1 i64.add local.tee $v0"))
    assert [execute_term(body, n).value for n in range(5)] == [1] * 5
    assert execute_term(parse("unreachable"), 0).reason == "unreachable"


def test_hand_constructed_ast_does_not_depend_on_parser_or_production(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("production or parser called by reference")

    import oeis_learn.sandbox.lowering as lowering
    import oeis_learn.sandbox.wat_ast as ast
    import wasmtime

    monkeypatch.setattr(lowering, "lower_body", forbidden)
    monkeypatch.setattr(lowering, "lower_constant", forbidden)
    monkeypatch.setattr(ast, "parse", forbidden)
    monkeypatch.setattr(wasmtime, "Module", forbidden)
    nodes = (
        Instruction("i256.const", 1),
        Instruction("i64.const", -1),
        Instruction("i256.mul_scalar"),
    )
    assert execute_term(Program(nodes, "", ()), 0).value == -1
    for op, arg in [("i256.add", 1), ("i256.sub", -1)]:
        nodes = (
            Instruction("i256.const", 2**255 - 1),
            Instruction("i256.const", arg),
            Instruction(op),
        )
        assert execute_term(Program(nodes, "", ()), 0).outcome == "numeric_limit"
