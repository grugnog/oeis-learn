"""Unit tests for arbitrary precision sampling and PSLQ integer relation solver."""

from oeis_learn.data.models import LatentDiscoveryCandidate
from oeis_learn.discovery.pslq_solver import PslqRelationSolver


def test_pslq_finds_exact_linear_sum():
    solver = PslqRelationSolver(precision_digits=100)

    # Relation: seq_A + seq_B = seq_C -> 1*A + 1*B - 1*C = 0
    seq_a = [n for n in range(30)]
    seq_b = [2 * n for n in range(30)]
    seq_c = [3 * n for n in range(30)]

    relation, conf = solver.find_relation([seq_a, seq_b, seq_c], term_index=10)

    assert relation is not None
    # 1, 1, -1 (or scaled multiple)
    assert sum(r * t for r, t in zip(relation, [10, 20, 30])) == 0
    assert conf < 1e-50


def test_pslq_verifier_candidate():
    solver = PslqRelationSolver(precision_digits=100)
    candidate = LatentDiscoveryCandidate(
        candidate_id="test_cand_1",
        relation_type="LINEAR_SUM",
        sequences=("A000001", "A000002", "A000003"),
        vector_distance=0.01,
    )

    terms_dict = {
        "A000001": [n for n in range(30)],
        "A000002": [n**2 for n in range(30)],
        "A000003": [n**2 + n for n in range(30)],
    }

    verified = solver.verify_candidate(candidate, terms_dict)
    assert verified.status == "PSLQ_VERIFIED"
    assert verified.pslq_vector is not None
