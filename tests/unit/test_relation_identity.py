"""Unit tests for canonical relation identity, GCD reduction, and triviality detection."""

from __future__ import annotations

import pytest
from oeis_learn.data.models import SequenceRef
from oeis_learn.discovery.relation_identity import (
    canonicalize_relation,
    is_trivial_relation,
)


def test_canonicalize_relation_permutation_and_gcd():
    # 2*A000290 - 4*A000027 = 0  =>  -4*A000027 + 2*A000290 = 0
    # Normalized by GCD 2 and first positive sign: 2*A000027 - 1*A000290 = 0
    op1 = SequenceRef("A000290", 1, 0)
    op2 = SequenceRef("A000027", 1, 0)

    rel1 = canonicalize_relation(operands=[op1, op2], coefficients=[2, -4])
    rel2 = canonicalize_relation(operands=[op2, op1], coefficients=[-4, 2])

    assert rel1.claim_id == rel2.claim_id
    assert rel1.canonical_expression == rel2.canonical_expression
    assert rel1.coefficients == ["2", "-1"]
    assert [op.oeis_id for op in rel1.operands] == ["A000027", "A000290"]


def test_triviality_detection_zero_coefficient():
    op1 = SequenceRef("A000005", 1, 0)
    op2 = SequenceRef("A000290", 1, 0)
    op3 = SequenceRef("A100000", 1, 0)

    # Relation with a zero coefficient: (0)*A000005 + (1)*A000290 + (-1)*A100000 = 0
    is_triv, reason = is_trivial_relation(operands=[op1, op2, op3], coefficients=[0, 1, -1])
    assert is_triv is True
    assert "zero coefficient" in reason.lower()


def test_triviality_detection_duplicate_sequence():
    op1 = SequenceRef("A000045", 1, 0)
    op2 = SequenceRef("A000045", 1, 0)
    is_triv, reason = is_trivial_relation(operands=[op1, op2], coefficients=[1, -1])
    assert is_triv is True
    assert "duplicate sequence" in reason.lower()
