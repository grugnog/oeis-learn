"""Integration test for end-to-end PSLQ theorem discovery and SymPy proof pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from oeis_learn.discovery.pslq_solver import PslqRelationSolver
from oeis_learn.discovery.symbolic_prover import SymbolicProver
from oeis_learn.discovery.vector_search import VectorRelationSearcher


def test_pslq_discovery_e2e_pipeline():
    # Sequence A: n, Sequence B: 2n, Sequence C: 3n -> A + B - C = 0
    terms_dict = {
        "A000027": [n for n in range(100)],
        "A005843": [2 * n for n in range(100)],
        "A008585": [3 * n for n in range(100)],
    }

    # Normalized directional embeddings
    v_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v_c = np.array([0.7071, 0.7071, 0.0], dtype=np.float32)

    embeddings = {
        "A000027": v_a,
        "A005843": v_b,
        "A008585": v_c,
    }

    searcher = VectorRelationSearcher(eps_distance=0.8, normalize_l2=True)
    candidates = searcher.search_additive_triples(embeddings, max_candidates=5)
    assert len(candidates) >= 1

    pslq_solver = PslqRelationSolver(precision_digits=100)
    verified_candidate = None
    for cand in candidates:
        res = pslq_solver.verify_candidate(cand, terms_dict)
        if res.status == "PSLQ_VERIFIED":
            verified_candidate = res
            break

    assert verified_candidate is not None
    assert verified_candidate.pslq_vector is not None

    prover = SymbolicProver()
    proven = prover.prove_relation(
        verified_candidate,
        formulas=["a(n) = n", "a(n) = 2*n", "a(n) = 3*n"],
    )
    assert proven.status == "PROVEN"
    assert proven.symbolic_proof is not None
