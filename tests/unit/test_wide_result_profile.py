"""Unit tests for exact 256-bit four-limb result profile decoding and arithmetic."""

from __future__ import annotations

import pytest
from oeis_learn.sandbox.runner import decode_i256_limbs


def test_decode_i256_limbs_zero():
    assert decode_i256_limbs([0, 0, 0, 0]) == 0


def test_decode_i256_limbs_small_positive():
    assert decode_i256_limbs([42, 0, 0, 0]) == 42


def test_decode_i256_limbs_carry_into_second_limb():
    # 2^64
    assert decode_i256_limbs([0, 1, 0, 0]) == 1 << 64


def test_decode_i256_limbs_carry_into_third_limb():
    # 2^128
    assert decode_i256_limbs([0, 0, 1, 0]) == 1 << 128


def test_decode_i256_limbs_negative_one():
    # In 64-bit signed representation, -1 is 0xFFFFFFFFFFFFFFFF
    assert decode_i256_limbs([-1, -1, -1, -1]) == -1


def test_decode_i256_powers_of_two_term_119():
    # 2^119: 119 = 64 + 55, so limb 0 = 0, limb 1 = 1 << 55
    expected = 1 << 119
    limbs = [0, 1 << 55, 0, 0]
    assert decode_i256_limbs(limbs) == expected
