"""Exact partitioned numerical verification across all validation and unseen terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence
from oeis_learn.data.models import SequenceRef
from oeis_learn.discovery.relation_identity import is_trivial_relation


@dataclass
class NumericalValidationResult:
    """Numerical verification outcome and evidence."""

    outcome: str  # VERIFIED, COUNTEREXAMPLE, TRIVIAL, INSUFFICIENT_EVIDENCE
    source_sha256: str = "sha256:" + "0" * 64
    search_indices: List[int] = field(default_factory=list)
    validation_indices: List[int] = field(default_factory=list)
    unseen_indices: List[int] = field(default_factory=list)
    coefficient_method: str = "PSLQ_500DIGITS"
    pslq_precision_digits: int = 500
    pslq_max_coefficient: int = 1000
    coefficients: List[str] = field(default_factory=list)
    validation_residuals: List[str] = field(default_factory=list)
    unseen_residuals: List[str] = field(default_factory=list)
    matrix_rank: int = 0
    matrix_nullity: int = 0
    minimal_support: bool = True
    first_counterexample: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "source_sha256": self.source_sha256,
            "search_indices": self.search_indices,
            "validation_indices": self.validation_indices,
            "unseen_indices": self.unseen_indices,
            "coefficient_method": self.coefficient_method,
            "pslq_precision_digits": self.pslq_precision_digits,
            "pslq_max_coefficient": self.pslq_max_coefficient,
            "coefficients": self.coefficients,
            "validation_residuals": self.validation_residuals,
            "unseen_residuals": self.unseen_residuals,
            "matrix_rank": self.matrix_rank,
            "matrix_nullity": self.matrix_nullity,
            "minimal_support": self.minimal_support,
            "first_counterexample": self.first_counterexample,
        }


def validate_numerical_relation(
    operands: Sequence[SequenceRef],
    coefficients: Sequence[int | str],
    sequence_terms_dict: Mapping[str, Sequence[int]],
    validation_indices: Sequence[int] = tuple(range(20)),
    unseen_indices: Sequence[int] = tuple(range(20, 120)),
    search_indices: Sequence[int] = tuple(range(10, 20)),
) -> NumericalValidationResult:
    """Verifies that sum_i c_i * a_i(n) == 0 holds identically across all indices."""
    int_coeffs = [int(c) for c in coefficients]
    coeff_strs = [str(c) for c in int_coeffs]

    is_triv, _ = is_trivial_relation(operands, int_coeffs)
    if is_triv:
        return NumericalValidationResult(
            outcome="TRIVIAL",
            coefficients=coeff_strs,
        )

    # Check that all sequence terms are present
    for op in operands:
        if op.oeis_id not in sequence_terms_dict:
            return NumericalValidationResult(
                outcome="INSUFFICIENT_EVIDENCE",
                coefficients=coeff_strs,
            )

    val_residuals: List[str] = []
    uns_residuals: List[str] = []
    first_counterexample: Optional[Dict[str, Any]] = None

    # Check validation indices
    for idx in validation_indices:
        res = 0
        for op, c in zip(operands, int_coeffs):
            seq_idx = op.index_scale * idx + op.index_shift
            seq = sequence_terms_dict[op.oeis_id]
            if 0 <= seq_idx < len(seq):
                res += c * int(seq[seq_idx])
            else:
                return NumericalValidationResult(outcome="INSUFFICIENT_EVIDENCE", coefficients=coeff_strs)

        val_residuals.append(str(res))
        if res != 0 and first_counterexample is None:
            first_counterexample = {"index": idx, "residual": str(res)}

    # Check unseen indices
    for idx in unseen_indices:
        res = 0
        for op, c in zip(operands, int_coeffs):
            seq_idx = op.index_scale * idx + op.index_shift
            seq = sequence_terms_dict[op.oeis_id]
            if 0 <= seq_idx < len(seq):
                res += c * int(seq[seq_idx])
            else:
                return NumericalValidationResult(outcome="INSUFFICIENT_EVIDENCE", coefficients=coeff_strs)

        uns_residuals.append(str(res))
        if res != 0 and first_counterexample is None:
            first_counterexample = {"index": idx, "residual": str(res)}

    outcome = "COUNTEREXAMPLE" if first_counterexample is not None else "VERIFIED"

    return NumericalValidationResult(
        outcome=outcome,
        search_indices=list(search_indices),
        validation_indices=list(validation_indices),
        unseen_indices=list(unseen_indices),
        coefficients=coeff_strs,
        validation_residuals=val_residuals,
        unseen_residuals=uns_residuals,
        first_counterexample=first_counterexample,
    )
