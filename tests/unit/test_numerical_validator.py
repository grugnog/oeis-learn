"""Unit tests for exact partitioned numerical validation of candidate relations."""

from __future__ import annotations

from oeis_learn.data.models import SequenceRef
from oeis_learn.discovery.numerical_validator import validate_numerical_relation


def test_validate_numerical_relation_exact_true():
    # A000290 (n^2) and A100000 (1*n^2) are identical: 1*A000290 - 1*A100000 = 0
    seq_terms = {
        "A000290": [n * n for n in range(120)],
        "A100000": [n * n for n in range(120)],
    }
    ops = [SequenceRef("A000290", 1, 0), SequenceRef("A100000", 1, 0)]
    coeffs = [1, -1]

    res = validate_numerical_relation(
        operands=ops,
        coefficients=coeffs,
        sequence_terms_dict=seq_terms,
        search_indices=list(range(10, 20)),
        validation_indices=list(range(10)),
        unseen_indices=list(range(20, 120)),
    )
    assert res.outcome == "VERIFIED"
    assert res.first_counterexample is None


def test_validate_numerical_relation_counterexample_detected():
    # Relation holds for n=0 and n=1, but diverges at n=2: 2n vs n^2
    seq_terms = {
        "seq1": [2 * n for n in range(120)],  # 0, 2, 4, 6...
        "seq2": [0, 2, 5, 8] + [n * 3 for n in range(4, 120)],
    }
    ops = [SequenceRef("seq1", 1, 0), SequenceRef("seq2", 1, 0)]
    coeffs = [1, -1]

    res = validate_numerical_relation(
        operands=ops,
        coefficients=coeffs,
        sequence_terms_dict=seq_terms,
        search_indices=list(range(10, 20)),
        validation_indices=list(range(10)),
        unseen_indices=list(range(20, 120)),
    )
    assert res.outcome == "COUNTEREXAMPLE"
    assert res.first_counterexample is not None
    assert res.first_counterexample["index"] == 2


def test_missing_evidence_is_unknown_without_fabricated_metadata():
    result = validate_numerical_relation(
        [SequenceRef("a"), SequenceRef("b")],
        [1, -1],
        {"a": [1]},
        search_indices=[0],
        validation_indices=[1],
        unseen_indices=[2],
    )
    assert result.outcome == "INSUFFICIENT_EVIDENCE"
    assert (
        result.source_sha256 is None
        and result.matrix_rank is None
        and result.matrix_nullity is None
    )
    assert result.pslq_precision_digits is None and result.pslq_max_coefficient is None


def test_duplicate_operands_use_the_combined_exact_matrix():
    a, b = SequenceRef("a"), SequenceRef("b")
    result = validate_numerical_relation(
        [a, a, b],
        [1, 1, -2],
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        search_indices=[0],
        validation_indices=[1],
        unseen_indices=[2],
    )
    assert result.outcome == "VERIFIED" and result.coefficients == ["1", "-1"]
    assert result.matrix_rank == 1 and result.matrix_nullity == 1
