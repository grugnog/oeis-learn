"""Unit tests for algebraic sequence transformations."""

from oeis_learn.data.transforms import (
    alternating_sign_transform,
    binomial_transform,
    first_difference_transform,
    partial_sum_transform,
    shift_transform,
)


def test_partial_sums():
    seq = [1, 2, 3, 4, 5]
    sums = partial_sum_transform(seq)
    assert sums == [1, 3, 6, 10, 15]


def test_first_differences():
    seq = [1, 4, 9, 16, 25]
    diffs = first_difference_transform(seq)
    assert diffs == [3, 5, 7, 9]


def test_binomial_transform():
    # Binomial transform of all 1s is powers of 2 (2^n)
    seq = [1] * 10
    bin_trans = binomial_transform(seq)
    assert bin_trans == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def test_shift_and_alternating():
    seq = [1, 2, 3, 4]
    shifted = shift_transform(seq, k=2)
    assert shifted == [3, 4]

    alt = alternating_sign_transform(seq)
    assert alt == [1, -2, 3, -4]
