"""Unit tests for Dixon 1-step bounded Diophantine solver."""

from __future__ import annotations

import pytest
from oeis_learn.decoder.dixon_solver import solve_dixon_bounded


def test_dixon_fibonacci_solving():
    # Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89...
    # c1 = 1, c2 = 1
    fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # Rows: a(n) = c1 * a(n-1) + c2 * a(n-2)
    A = []
    b = []
    for n in range(2, len(fib)):
        A.append([fib[n - 1], fib[n - 2]])
        b.append(fib[n])

    is_sat, constants, dur_ms, cert = solve_dixon_bounded(A, b, unknown_count=2)
    assert is_sat is True
    assert constants == [1, 1]
    assert cert.status == "CONSISTENT"
    assert dur_ms < 1.0


def test_dixon_pell_solving():
    # Pell: 0, 1, 2, 5, 12, 29, 70, 169...
    # c1 = 2, c2 = 1
    pell = [0, 1, 2, 5, 12, 29, 70, 169]
    A = []
    b = []
    for n in range(2, len(pell)):
        A.append([pell[n - 1], pell[n - 2]])
        b.append(pell[n])

    is_sat, constants, dur_ms, cert = solve_dixon_bounded(A, b, unknown_count=2)
    assert is_sat is True
    assert constants == [2, 1]
    assert cert.status == "CONSISTENT"
    assert dur_ms < 1.0


def test_dixon_negative_coefficients():
    # a(n) = 3*a(n-1) - 2*a(n-2) (powers of 2 offset: 2^n - 1: 0, 1, 3, 7, 15, 31, 63...)
    seq = [0, 1, 3, 7, 15, 31, 63, 127]
    A = []
    b = []
    for n in range(2, len(seq)):
        A.append([seq[n - 1], seq[n - 2]])
        b.append(seq[n])

    is_sat, constants, dur_ms, cert = solve_dixon_bounded(A, b, unknown_count=2)
    assert is_sat is True
    assert constants == [3, -2]
    assert cert.status == "CONSISTENT"
    assert dur_ms < 1.0


def test_dixon_order3_tribonacci():
    # Tribonacci: 0, 0, 1, 1, 2, 4, 7, 13, 24, 44...
    trib = [0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81]
    A = []
    b = []
    for n in range(3, len(trib)):
        A.append([trib[n - 1], trib[n - 2], trib[n - 3]])
        b.append(trib[n])

    is_sat, constants, dur_ms, cert = solve_dixon_bounded(A, b, unknown_count=3)
    assert is_sat is True
    assert constants == [1, 1, 1]
    assert cert.status == "CONSISTENT"
    assert dur_ms < 1.0


def test_dixon_inconsistent_fails():
    A = [[1, 0], [2, 1], [3, 2]]
    b = [10, 20, 999]  # Inconsistent
    is_sat, constants, dur_ms, cert = solve_dixon_bounded(A, b, unknown_count=2)
    assert is_sat is False
    assert constants == []
    assert cert.status == "INCONSISTENT"
