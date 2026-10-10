"""Exact sequence relations; PSLQ is restricted to approximate real constants."""

from dataclasses import dataclass
from itertools import product
from math import gcd, lcm
import time
from typing import Sequence
import mpmath
import sympy as sp


def exact_integer(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and str(int(value)) == value:
        return int(value)
    raise ValueError("canonical exact integer required")


def primitive(values):
    denominators = lcm(*(int(v.q) for v in values))
    integers = [int(v * denominators) for v in values]
    divisor = gcd(*integers)
    if not divisor:
        raise ValueError("zero relation")
    integers = [v // divisor for v in integers]
    if next(v for v in integers if v) < 0:
        integers = [-v for v in integers]
    return integers


@dataclass(frozen=True)
class RelationFit:
    status: str
    coefficients: tuple[int, ...] | None
    rank: int
    nullity: int
    support: int | None
    attempts: int
    minimal_support: None = None


class PslqRelationSolver:
    """Compatibility entry point, now using exact multi-index integer fitting."""

    def __init__(self, precision_digits=500, max_coeff=1000, *, max_attempts=4096):
        if not 1 <= max_coeff <= 10**9 or not 1 <= max_attempts <= 4096:
            raise ValueError("relation search bounds exceeded")
        self.precision_digits, self.max_coeff = precision_digits, max_coeff
        self.max_attempts = max_attempts

    def fit(self, sequence_vectors: Sequence[Sequence[int]], indices, *, seconds=2):
        k = len(sequence_vectors)
        indices = tuple(indices)
        if not 2 <= k <= 8 or not indices or len(set(indices)) != len(indices):
            raise ValueError("bounded arity and nonempty unique fitting indices required")
        if any(
            type(i) is not int or i < 0 or any(i >= len(s) for s in sequence_vectors)
            for i in indices
        ):
            raise ValueError("missing fitting index")
        if not 0 < seconds <= 10:
            raise ValueError("bounded fitting deadline required")
        deadline = time.monotonic() + seconds
        matrix = sp.Matrix([[exact_integer(s[i]) for s in sequence_vectors] for i in indices])
        rank = matrix.rank()
        basis = matrix.nullspace()
        if not basis:
            return RelationFit("NO_RELATION", None, rank, k - rank, None, 0)
        # A basis need not contain a bounded full-support vector. Search combinations;
        # budget exhaustion never establishes UNSAT or minimality.
        combinations = ([int(i == j) for i in range(len(basis))] for j in range(len(basis)))
        from itertools import chain

        attempts = 0
        for weights in chain(combinations, product(range(-2, 3), repeat=len(basis))):
            if attempts >= self.max_attempts or time.monotonic() >= deadline:
                break
            if not any(weights):
                continue
            attempts += 1
            vector = sum((w * b for w, b in zip(weights, basis)), sp.zeros(k, 1))
            if not any(vector):
                continue
            candidate = primitive(vector)
            if all(candidate) and max(map(abs, candidate)) <= self.max_coeff:
                return RelationFit("FIT", tuple(candidate), rank, k - rank, k, attempts)
        return RelationFit("UNKNOWN", None, rank, k - rank, None, attempts)

    def find_relation(self, sequence_vectors, term_index=10, *, fitting_indices=None):
        if len(sequence_vectors) < 2 or not all(sequence_vectors):
            return None, 1.0
        indices = (
            fitting_indices
            if fitting_indices is not None
            else range(min(20, min(map(len, sequence_vectors))))
        )
        result = self.fit(sequence_vectors, indices)
        return (list(result.coefficients), 0.0) if result.coefficients else (None, 1.0)

    def approximate_real_relation(self, values, *, contract):
        if contract != {
            "kind": "approximate_real_constants",
            "precision_digits": self.precision_digits,
            "max_coefficient": self.max_coeff,
        }:
            raise ValueError("explicit approximate-real contract required")
        with mpmath.workdps(self.precision_digits):
            relation = mpmath.pslq(
                mpmath.matrix([mpmath.mpf(v) for v in values]), maxcoeff=self.max_coeff
            )
        return {
            "kind": "approximate_real_proposal",
            "coefficients": relation,
            "proof_status": "not_claimed",
        }

    def verify_candidate(self, candidate, sequence_terms_dict):
        if any(s not in sequence_terms_dict for s in candidate.sequences):
            candidate.status = "INSUFFICIENT_EVIDENCE"
            return candidate
        vectors = [sequence_terms_dict[s] for s in candidate.sequences]
        relation, conf = self.find_relation(vectors)
        if relation and all(
            sum(c * s[i] for c, s in zip(relation, vectors)) == 0
            for i in range(min(map(len, vectors)))
        ):
            candidate.pslq_vector = relation
            candidate.pslq_confidence_ratio = conf
            candidate.status = "EXACT_FINITE_VERIFIED"
        else:
            candidate.status = "UNKNOWN"
        return candidate
