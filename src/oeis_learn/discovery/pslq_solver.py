"""High-precision numerical sampling and PSLQ integer relation solver."""

from __future__ import annotations

import math
from typing import List, Mapping, Optional, Sequence, Tuple
import mpmath
from oeis_learn.data.models import LatentDiscoveryCandidate


class PslqRelationSolver:
    """Solves integer relations using the PSLQ algorithm at arbitrary precision (>500 digits)."""

    def __init__(self, precision_digits: int = 500, max_coeff: int = 1000):
        self.precision_digits = precision_digits
        self.max_coeff = max_coeff

    def find_relation(
        self,
        sequence_vectors: Sequence[Sequence[int]],
        term_index: int = 10,
    ) -> Tuple[Optional[List[int]], float]:
        """Finds integer vector a in Z^k such that sum_i a_i * x_i = 0.

        Args:
            sequence_vectors: List of sequences [seq_1, seq_2, ..., seq_k]
            term_index: Term index n to evaluate for integer relation

        Returns:
            Tuple of (integer relation vector or None, confidence ratio drop)
        """
        k = len(sequence_vectors)
        if k < 2:
            return None, 1.0

        min_len = min((len(seq) for seq in sequence_vectors), default=0)
        if min_len <= 0:
            return None, 1.0

        # Ensure term_index is within valid bounds
        eval_index = min(max(0, term_index), min_len - 1)

        with mpmath.workdps(self.precision_digits):
            # Form vector of terms at index n
            terms = [mpmath.mpf(seq[eval_index]) for seq in sequence_vectors]

            try:
                # Run PSLQ algorithm
                relation = mpmath.pslq(terms, maxcoeff=self.max_coeff)
            except Exception:
                relation = None

            if relation is None:
                return None, 1.0

            # Compute error residual sum_i a_i * x_i
            residual = abs(sum(r * t for r, t in zip(relation, terms)))
            max_val = max(abs(t) for t in terms)
            confidence_ratio = float(residual / (max_val + 1e-100))

            return list(relation), confidence_ratio

    def verify_candidate(
        self,
        candidate: LatentDiscoveryCandidate,
        sequence_terms_dict: Mapping[str, Sequence[int]],
    ) -> LatentDiscoveryCandidate:
        """Verifies candidate sequences using PSLQ across multiple term indices."""
        seq_vectors = [sequence_terms_dict[sid] for sid in candidate.sequences if sid in sequence_terms_dict]
        if len(seq_vectors) != len(candidate.sequences):
            candidate.status = "REJECTED"
            return candidate

        min_len = min((len(s) for s in seq_vectors), default=0)
        if min_len < 3:
            candidate.status = "REJECTED"
            return candidate

        # Run PSLQ on terms at target evaluation index
        eval_idx = min(15, min_len - 1)
        relation, conf = self.find_relation(seq_vectors, term_index=eval_idx)
        if relation is not None and conf < 1e-30:
            # Reject trivial zero coefficients
            if any(c == 0 for c in relation):
                candidate.status = "REJECTED"
                return candidate

            # Multi-term consistency check
            consistent = True
            for c_idx in [0, min(5, min_len - 1), min(10, min_len - 1)]:
                res_val = sum(r * s[c_idx] for r, s in zip(relation, seq_vectors))
                if res_val != 0:
                    consistent = False
                    break

            if consistent:
                candidate.pslq_vector = relation
                candidate.pslq_confidence_ratio = conf
                candidate.status = "PSLQ_VERIFIED"
            else:
                candidate.status = "REJECTED"
        else:
            candidate.status = "REJECTED"

        return candidate
