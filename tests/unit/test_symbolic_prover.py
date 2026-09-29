"""Unit tests for SymPy symbolic recurrence and identity prover."""

from oeis_learn.data.models import LatentDiscoveryCandidate
from oeis_learn.discovery.symbolic_prover import SymbolicProver


def test_symbolic_prover_verifies_polynomial_sum():
    prover = SymbolicProver()
    candidate = LatentDiscoveryCandidate(
        candidate_id="test_cand_2",
        relation_type="LINEAR_SUM",
        sequences=("A000001", "A000002", "A000003"),
        vector_distance=0.01,
        pslq_vector=[1, 1, -1],
        status="PSLQ_VERIFIED",
    )

    formulas = ["a(n) = n", "a(n) = n**2", "a(n) = n**2 + n"]
    proven = prover.prove_relation(candidate, formulas=formulas)

    assert proven.status == "PROVEN"
    assert proven.symbolic_proof is not None
    assert "Q.E.D." in proven.symbolic_proof


def test_symbolic_prover_missing_definitions_remains_conjecture():
    prover = SymbolicProver()
    candidate = LatentDiscoveryCandidate(
        candidate_id="test_cand_3",
        relation_type="LINEAR_SUM",
        sequences=("A000045", "A000032", "A000213"),
        vector_distance=0.05,
        pslq_vector=[1, 1, -1],
        status="PSLQ_VERIFIED",
    )

    # No explicit formulas provided
    result = prover.prove_relation(candidate, formulas=None)
    assert result.status != "PROVEN"
    assert result.status in ("NUMERICALLY_VERIFIED_CONJECTURE", "CONJECTURED", "PSLQ_VERIFIED")

