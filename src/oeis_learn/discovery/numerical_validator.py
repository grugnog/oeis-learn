"""Exact partitioned numerical verification across all validation and unseen terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
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
    minimal_support: Optional[bool] = None
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
    operands,
    coefficients,
    sequence_terms_dict,
    validation_indices=tuple(range(10, 20)),
    unseen_indices=tuple(range(20, 120)),
    search_indices=tuple(range(10)),
):
    """Finite exact evidence over three disjoint, nonempty partitions."""
    import sympy as sp
    from oeis_learn.discovery.pslq_solver import exact_integer
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    parts = [tuple(p) for p in (search_indices, validation_indices, unseen_indices)]
    if (
        sum(map(len, parts)) > 120
        or any(
            not p or len(set(p)) != len(p) or any(type(i) is not int or i < 0 for i in p)
            for p in parts
        )
        or any(set(parts[i]) & set(parts[j]) for i in range(3) for j in range(i))
    ):
        raise ValueError("nonempty disjoint exact index partitions required")
    if len(operands) != len(coefficients) or not 2 <= len(operands) <= 8:
        raise ValueError("bounded equal-length operands/coefficients required")
    ints = [exact_integer(c) for c in coefficients]
    coeffs = [str(c) for c in ints]
    if is_trivial_relation(operands, ints)[0]:
        return NumericalValidationResult(outcome="TRIVIAL", coefficients=coeffs)
    rows = []
    for index in sum(parts, ()):
        row = []
        for op in operands:
            if type(op.index_scale) is not int or type(op.index_shift) is not int:
                raise ValueError("exact source index transforms required")
            seq = sequence_terms_dict.get(op.oeis_id)
            idx = op.index_scale * index + op.index_shift
            if seq is None or not 0 <= idx < len(seq):
                return NumericalValidationResult(
                    outcome="INSUFFICIENT_EVIDENCE", coefficients=coeffs
                )
            row.append(exact_integer(seq[idx]))
        rows.append(row)
    rank = int(sp.Matrix(rows[: len(parts[0])]).rank())
    residuals = [sum(c * v for c, v in zip(ints, row)) for row in rows]
    first = next(
        (
            {"index": idx, "residual": str(r)}
            for idx, r in sorted(zip(sum(parts, ()), residuals))
            if r
        ),
        None,
    )
    return NumericalValidationResult(
        outcome="COUNTEREXAMPLE" if first else "VERIFIED",
        source_sha256=digest(
            {
                "operands": [op.to_dict() for op in operands],
                "partitions": parts,
                "indexed_values": [[str(v) for v in row] for row in rows],
            }
        ),
        search_indices=list(parts[0]),
        validation_indices=list(parts[1]),
        unseen_indices=list(parts[2]),
        coefficients=coeffs,
        coefficient_method="EXACT_INTEGER_NULLSPACE",
        validation_residuals=[
            str(r) for r in residuals[len(parts[0]) : len(parts[0]) + len(parts[1])]
        ],
        unseen_residuals=[str(r) for r in residuals[len(parts[0]) + len(parts[1]) :]],
        matrix_rank=rank,
        matrix_nullity=len(operands) - rank,
        minimal_support=None,
        first_counterexample=first,
    )
