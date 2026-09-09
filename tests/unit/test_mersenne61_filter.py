"""Unit tests for Mersenne-61 fast modular rank filter."""

from __future__ import annotations

import time
import pytest
from oeis_learn.decoder.mersenne61_filter import (
    M61_PRIME,
    evaluate_modular_filter,
    fold61_u64,
    int_to_m61,
    project_i256_to_m61,
)


def test_fold61_u64():
    assert fold61_u64(0) == 0
    assert fold61_u64(42) == 42
    # 2^61: (2^61 & mask) + (2^61 >> 61) = 0 + 1 = 1
    assert fold61_u64(1 << 61) == 1
    # 2^62: (2^62 & mask) + (2^62 >> 61) = 0 + 2 = 2
    assert fold61_u64(1 << 62) == 2


def test_project_i256_to_m61_positive_and_negative():
    # 42
    assert project_i256_to_m61((42, 0, 0, 0)) == 42
    # -1: 2's complement is (mask64, mask64, mask64, mask64)
    mask64 = (1 << 64) - 1
    res_neg1 = project_i256_to_m61((mask64, mask64, mask64, mask64))
    # In F_p, -1 is p - 1
    assert res_neg1 == M61_PRIME - 1


def test_modular_filter_consistent_system():
    # Linear recurrence: a(n) = 2*a(n-1) + a(n-2) (Pell: 0, 1, 2, 5, 12, 29, 70...)
    # System A * [c1, c2] = b
    # For n=2: 2*1 + 0 = 2 -> A[0] = [1, 0], b[0] = 2
    # For n=3: 2*2 + 1 = 5 -> A[1] = [2, 1], b[1] = 5
    # For n=4: 2*5 + 2 = 12 -> A[2] = [5, 2], b[2] = 12
    A = [
        [1, 0],
        [2, 1],
        [5, 2],
        [12, 5],
    ]
    b = [2, 5, 12, 29]

    cert = evaluate_modular_filter(A, b, unknown_count=2)
    assert cert.status == "CONSISTENT"
    assert cert.coefficient_rank == 2
    assert cert.augmented_rank == 2
    assert cert.penalty_reward == 0.0
    assert cert.elapsed_microseconds < 500.0


def test_modular_filter_inconsistent_system():
    # Inconsistent equations:
    # x1 + x2 = 5
    # x1 + x2 = 6
    A = [
        [1, 1],
        [1, 1],
        [2, 2],
    ]
    b = [5, 6, 11]

    cert = evaluate_modular_filter(A, b, unknown_count=2)
    assert cert.status == "INCONSISTENT"
    assert cert.penalty_reward == -0.50
    assert cert.elapsed_microseconds < 500.0


def test_modular_filter_underdetermined_system():
    # Only 1 independent equation for 2 unknowns:
    A = [
        [1, 2],
        [2, 4],
        [3, 6],
    ]
    b = [5, 10, 15]

    cert = evaluate_modular_filter(A, b, unknown_count=2)
    assert cert.status == "UNDERDETERMINED"
    assert cert.penalty_reward == -0.50
    assert cert.coefficient_rank == 1
    assert cert.unknown_count == 2
    assert cert.elapsed_microseconds < 500.0
