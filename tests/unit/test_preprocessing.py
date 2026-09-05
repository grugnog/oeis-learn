"""Unit tests for sequence preprocessing, magnitude features, and growth rate analysis."""

import pytest
from oeis_learn.data.preprocessing import (
    analyze_log_linearity,
    check_finite_difference_polynomial_degree,
    compute_magnitude_4d_features,
)


def test_compute_magnitude_4d_features():
    # Test zero
    log_v, s_pos, s_neg, s_zero = compute_magnitude_4d_features(0)
    assert log_v == 0.0
    assert (s_pos, s_neg, s_zero) == (0.0, 0.0, 1.0)

    # Test positive 1
    log_v, s_pos, s_neg, s_zero = compute_magnitude_4d_features(1)
    assert abs(log_v - 1.0) < 1e-6
    assert (s_pos, s_neg, s_zero) == (1.0, 0.0, 0.0)

    # Test negative 100
    log_v, s_pos, s_neg, s_zero = compute_magnitude_4d_features(-100)
    assert abs(log_v - 3.0) < 1e-6  # 1.0 + log10(100) = 3.0
    assert (s_pos, s_neg, s_zero) == (0.0, 1.0, 0.0)

    # Test astronomical number (10^500)
    huge = 10**500
    log_v, s_pos, s_neg, s_zero = compute_magnitude_4d_features(huge)
    assert log_v >= 500.0
    assert (s_pos, s_neg, s_zero) == (1.0, 0.0, 0.0)


def test_analyze_log_linearity_exponential():
    # Powers of 2: 2^n -> strictly log-linear
    powers_of_2 = [2**n for n in range(1, 20)]
    is_log_lin, r2 = analyze_log_linearity(powers_of_2)
    assert is_log_lin is True
    assert r2 > 0.99

    # Fibonacci numbers: F(n) -> asymptotically phi^n (log-linear)
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    is_log_lin, r2 = analyze_log_linearity(fib)
    assert is_log_lin is True
    assert r2 > 0.98


def test_analyze_log_linearity_polynomial():
    # Squares: n^2 -> log(n^2) = 2*log(n), not linear in n!
    squares = [n**2 for n in range(1, 25)]
    is_log_lin, r2 = analyze_log_linearity(squares)
    assert is_log_lin is False


def test_check_finite_difference_polynomial_degree():
    # Linear: 3n + 5 -> degree 1
    linear = [3 * n + 5 for n in range(10)]
    assert check_finite_difference_polynomial_degree(linear) == 1

    # Quadratic: 2n^2 + n + 1 -> degree 2
    quad = [2 * n * n + n + 1 for n in range(10)]
    assert check_finite_difference_polynomial_degree(quad) == 2

    # Cubic: n^3 - 2n -> degree 3
    cubic = [n**3 - 2 * n for n in range(12)]
    assert check_finite_difference_polynomial_degree(cubic) == 3

    # Exponential: 2^n -> not low degree polynomial
    exp_seq = [2**n for n in range(10)]
    assert check_finite_difference_polynomial_degree(exp_seq) is None
