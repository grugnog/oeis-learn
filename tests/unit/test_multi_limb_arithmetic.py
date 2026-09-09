"""Unit tests for MultiLimbRegisterState and 255-bit two's complement arithmetic."""

from __future__ import annotations

import pytest
from oeis_learn.data.models import MultiLimbRegisterState


def test_multilimb_zero():
    reg = MultiLimbRegisterState(limbs=(0, 0, 0, 0))
    assert reg.signed_value == 0
    assert reg.is_negative is False
    assert reg.bit_length == 0


def test_multilimb_small_positive():
    reg = MultiLimbRegisterState(limbs=(42, 0, 0, 0))
    assert reg.signed_value == 42
    assert reg.is_negative is False
    assert reg.bit_length == (42).bit_length()


def test_multilimb_limb_boundaries():
    # 2^64
    reg64 = MultiLimbRegisterState(limbs=(0, 1, 0, 0))
    assert reg64.signed_value == 1 << 64
    assert reg64.is_negative is False
    assert reg64.bit_length == 65

    # 2^128
    reg128 = MultiLimbRegisterState(limbs=(0, 0, 1, 0))
    assert reg128.signed_value == 1 << 128
    assert reg128.is_negative is False
    assert reg128.bit_length == 129

    # 2^192
    reg192 = MultiLimbRegisterState(limbs=(0, 0, 0, 1))
    assert reg192.signed_value == 1 << 192
    assert reg192.is_negative is False
    assert reg192.bit_length == 193


def test_multilimb_negative_one():
    # In 64-bit unsigned limbs, -1 in 2's complement is (2^64-1, 2^64-1, 2^64-1, 2^64-1)
    mask64 = (1 << 64) - 1
    reg_neg1 = MultiLimbRegisterState(limbs=(mask64, mask64, mask64, mask64))
    assert reg_neg1.signed_value == -1
    assert reg_neg1.is_negative is True
    assert reg_neg1.bit_length == 1


def test_multilimb_negative_values():
    # -42
    reg = MultiLimbRegisterState.from_int(-42)
    assert reg.signed_value == -42
    assert reg.is_negative is True
    assert reg.bit_length == (42).bit_length()


def test_multilimb_max_positive():
    max_pos = (1 << 255) - 1
    reg = MultiLimbRegisterState.from_int(max_pos)
    assert reg.signed_value == max_pos
    assert reg.is_negative is False
    assert reg.bit_length == 255


def test_multilimb_min_negative():
    min_neg = -(1 << 255)
    reg = MultiLimbRegisterState.from_int(min_neg)
    assert reg.signed_value == min_neg
    assert reg.is_negative is True
    assert reg.bit_length == 256


def test_multilimb_roundtrip_range():
    test_values = [
        0, 1, -1, 42, -42, 2**63 - 1, -(2**63), 2**64, -(2**64),
        2**120, -(2**120), 2**200, -(2**200), (1 << 255) - 1, -(1 << 255)
    ]
    for v in test_values:
        reg = MultiLimbRegisterState.from_int(v)
        assert reg.signed_value == v, f"Failed roundtrip for value {v}"
        reconstructed = MultiLimbRegisterState(limbs=reg.limbs)
        assert reconstructed.signed_value == v
