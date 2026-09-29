"""Primitive canonical relation identity, GCD reduction, and triviality checks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from oeis_learn.data.models import CanonicalRelation, SequenceRef
from oeis_learn.evaluation.protocol import canonical_json_dumps, canonical_json_hash


def is_trivial_relation(
    operands: Sequence[SequenceRef],
    coefficients: Sequence[int | str],
) -> Tuple[bool, Optional[str]]:
    """Checks whether a proposed linear relation is trivial or degenerate."""
    int_coeffs = [int(c) for c in coefficients]
    if len(operands) != len(int_coeffs):
        return True, "Operands and coefficients length mismatch"

    if any(c == 0 for c in int_coeffs):
        return True, "Contains zero coefficient"

    # Check for duplicate sequence operands
    seen_ops = set()
    for op in operands:
        key = (op.oeis_id, op.index_scale, op.index_shift)
        if key in seen_ops:
            return True, "Contains duplicate sequence operands"
        seen_ops.add(key)

    if len(operands) < 2:
        return True, "Relation must have at least 2 distinct operands"

    return False, None


def canonicalize_relation(
    operands: Sequence[SequenceRef],
    coefficients: Sequence[int | str],
) -> CanonicalRelation:
    """Reduces and deterministically canonicalizes a linear sequence relation."""
    int_coeffs = [int(c) for c in coefficients]

    # 1. Pair operands and coefficients, then sort lexicographically by operand identity
    paired = list(zip(operands, int_coeffs))
    paired.sort(key=lambda item: (item[0].oeis_id, item[0].index_scale, item[0].index_shift))

    sorted_ops = [p[0] for p in paired]
    sorted_coeffs = [p[1] for p in paired]

    # 2. Divide by greatest common divisor
    g = 0
    for c in sorted_coeffs:
        g = math.gcd(g, abs(c))
    if g > 1:
        sorted_coeffs = [c // g for c in sorted_coeffs]

    # 3. Global sign normalization: first non-zero coefficient must be positive
    first_nonzero = next((c for c in sorted_coeffs if c != 0), 0)
    if first_nonzero < 0:
        sorted_coeffs = [-c for c in sorted_coeffs]

    # 4. Deterministic string expression
    terms_strs = []
    for op, c in zip(sorted_ops, sorted_coeffs):
        term_label = op.oeis_id
        if op.index_scale != 1 or op.index_shift != 0:
            term_label = f"{op.oeis_id}({op.index_scale}n+{op.index_shift})"
        terms_strs.append(f"({c})*{term_label}")

    expr = " + ".join(terms_strs) + " = 0"
    coeff_strings = [str(c) for c in sorted_coeffs]

    payload_for_hash = {
        "relation_type": "POINTWISE_INTEGER_LINEAR_V1",
        "operands": [op.to_dict() for op in sorted_ops],
        "coefficients": coeff_strings,
    }
    claim_id = canonical_json_hash(payload_for_hash)

    return CanonicalRelation(
        relation_type="POINTWISE_INTEGER_LINEAR_V1",
        operands=sorted_ops,
        coefficients=coeff_strings,
        canonical_expression=expr,
        claim_id=claim_id,
    )
