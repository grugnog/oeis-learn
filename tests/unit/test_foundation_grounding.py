import pytest
from oeis_learn.decoder.constant_solver import (
    parse_ast_placeholders, solve_constants, resolve_program_constants,
    abstract_wat_constants,
)
from oeis_learn.decoder.dixon_solver import solve_exact_bounded
from oeis_learn.decoder.mersenne61_filter import M61_PRIME, evaluate_modular_filter
from oeis_learn.decoder.grounding import ground
from oeis_learn.decoder.recurrence import order2_template, recognize, lag_system


def module(body, locals_=""):
    return f'(module (func (export "compute") (param $n i32) (result i64) {locals_} {body}))'


def test_nonlinear_parameter_local_square():
    source = module("i64.const_? local.tee $c local.get $c i64.mul", "(local $c i64)")
    skeleton = parse_ast_placeholders(source)
    assert not skeleton.is_linear
    result = solve_constants(skeleton, [4] * 20, timeout_ms=1000)
    assert result.is_sat and result.constants[0] in (-2, 2), result
    assert result.solver_tier == "z3_qfbv"


def test_large_exact_signed_terms():
    target = -(2**100 + 13)
    result = ground(f"i256.const {target-7} i256.const_? i256.add", [target] * 20)
    assert result.outcome == "verified_solution" and result.constants == (7,), result


def test_real_integer_nonlinear_dispatch():
    result = ground("i256.const_? i64.const_? i256.mul_scalar", [4] * 20, timeout_ms=1000)
    assert result.outcome == "verified_solution" and result.method == "z3_qfnia", result
    assert result.constants[0] * result.constants[1] == 4


@pytest.mark.parametrize("matrix,rhs,k,expected", [
    ([[1, 1]], [3], 2, "verified_solution"),
    ([[0, 1], [0, 2]], [2, 4], 2, "verified_solution"),
    ([[M61_PRIME]], [M61_PRIME * 7], 1, "verified_solution"),
    ([[2]], [1], 1, "proved_unsat_in_scope"),
    ([[2**80 + 1]], [-9 * (2**80 + 1)], 1, "verified_solution"),
])
def test_exact_algebra(matrix, rhs, k, expected):
    result = solve_exact_bounded(matrix, rhs, k)
    assert result.outcome == expected
    if result.constants:
        assert all(sum(a*c for a, c in zip(row, result.constants)) == b for row, b in zip(matrix, rhs))


def test_augmented_rank_is_not_inconsistent_row_count():
    assert evaluate_modular_filter([[0], [0]], [1, 2], 1).augmented_rank == 1
    assert evaluate_modular_filter([[1, 1]], [2], 2).penalty_reward == 0


def test_empty_parameter_domain_is_verified():
    assert resolve_program_constants(module("i64.const 1"), [2]*20)[2] == "PROVED_UNSAT_IN_SCOPE"
    assert ground("i256.const 1", [2]*20).outcome == "proved_unsat_in_scope"
    assert ground("i256.const 2", [2]*20).outcome == "verified_solution"


def test_control_and_invalid_signatures_are_not_unsat():
    source = module("(loop $x i64.const_? drop br $x) i64.const 1")
    assert ground(source, [1]*20).outcome == "unsupported"
    invalid = module("i64.const_?").replace("(result i64)", "(result i32)")
    assert ground(invalid, [1]*20).outcome != "verified_solution"


def test_tail_is_not_fitting_data():
    a = ground("i256.const_?", [5]*20 + [7]*100)
    b = ground("i256.const_?", [5]*20 + [9]*100)
    assert a.outcome == b.outcome == "verified_solution"
    assert a.constants == b.constants and a.visible_prompt_sha256 == b.visible_prompt_sha256
    assert a.fit_indices == tuple(range(20))


def test_narrow_recurrence_and_cache():
    source = order2_template()
    terms = [0, 1]
    for _ in range(18):
        terms.append(sum(terms[-2:]))
    assert recognize(source) == (0, 1)
    assert recognize(source.replace("i32.ge_s", "i32.gt_s")) is None
    result = ground(source, terms)
    assert result.outcome == "verified_solution" and result.constants == (1, 1), result
    assert lag_system(tuple(terms)) is lag_system(tuple(terms))


def test_abstraction_preserves_helper_literals():
    assert abstract_wat_constants("i64.const 17 i256.const 22 i32.const 1") == "i64.const 17 i256.const_? i32.const 1"
