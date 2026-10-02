"""Exact bounded affine solving. The historical API name is retained, not its false Dixon claim."""
from fractions import Fraction
import time
from oeis_learn.decoder.mersenne61_filter import evaluate_modular_filter
from oeis_learn.decoder.qfnia_solver import IntegerSolution, solve_integer_constraints


def solve_exact_bounded(matrix_a, vector_b, unknown_count, bound=1000, timeout_ms=240):
    started = time.monotonic()
    evaluate_modular_filter(matrix_a, vector_b, unknown_count)  # diagnostic only
    if type(bound) is not int or not 0 <= bound <= 1000:
        raise ValueError("coefficient bound must be in 0..1000")
    k = unknown_count
    rows = [[Fraction(v) for v in (*row, b)] for row, b in zip(matrix_a, vector_b)]
    pivots = []
    for column in range(k):
        position = len(pivots)
        pivot = next((r for r in range(position, len(rows)) if rows[r][column]), None)
        if pivot is None:
            continue
        rows[position], rows[pivot] = rows[pivot], rows[position]
        scale = rows[position][column]
        rows[position] = [v / scale for v in rows[position]]
        for r in range(len(rows)):
            if r != position:
                scale = rows[r][column]
                rows[r] = [x - scale * y for x, y in zip(rows[r], rows[position])]
        pivots.append(column)
    def result(outcome, values=()):
        return IntegerSolution(outcome, tuple(values), (time.monotonic() - started) * 1000, method="exact_rational_elimination")
    if any(not any(row[:k]) and row[k] for row in rows):
        return result("proved_unsat_in_scope")
    if len(pivots) == k:
        values = [0] * k
        for r, column in enumerate(pivots):
            value = rows[r][k]
            if value.denominator != 1 or abs(value) > bound:
                return result("proved_unsat_in_scope")
            values[column] = int(value)
        assert all(sum(a * c for a, c in zip(row, values)) == b for row, b in zip(matrix_a, vector_b))
        return result("verified_solution", values)
    return solve_integer_constraints(k, lambda cs: [sum(a * c for a, c in zip(row, cs)) == b for row, b in zip(matrix_a, vector_b)], bound, timeout_ms)


def solve_dixon_bounded(matrix_a, vector_b, unknown_count, bound=1000):
    certificate = evaluate_modular_filter(matrix_a, vector_b, unknown_count)
    result = solve_exact_bounded(matrix_a, vector_b, unknown_count, bound)
    return result.outcome == "verified_solution", list(result.constants), result.elapsed_ms, certificate
