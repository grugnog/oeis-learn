"""Unit tests for L2-normalized nearest-neighbor vector relation search."""

from __future__ import annotations

import numpy as np
import pytest
from oeis_learn.discovery.vector_search import VectorRelationSearcher


def test_l2_normalized_vector_relation_search():
    # Construct embeddings with large norms (~10.0) but aligned direction
    # z_A + z_B = z_C
    v_a = np.array([8.0, 0.0, 6.0], dtype=np.float32)   # norm 10.0
    v_b = np.array([0.0, 10.0, 0.0], dtype=np.float32)  # norm 10.0
    # In unnormalized space, norm is ~14.14
    # In normalized space: u_a = [0.8, 0, 0.6], u_b = [0, 1, 0], u_c = (u_a + u_b) / sqrt(2)
    u_a = v_a / np.linalg.norm(v_a)
    u_b = v_b / np.linalg.norm(v_b)
    u_c = (u_a + u_b) / np.linalg.norm(u_a + u_b)
    v_c = u_c * 12.0  # arbitrary large norm

    embeddings = {
        "A000001": v_a,
        "A000002": v_b,
        "A000003": v_c,
    }

    # With normalize_l2=True and distance threshold eps=0.8, the triple should be detected
    searcher = VectorRelationSearcher(eps_distance=0.8, normalize_l2=True)
    candidates = searcher.search_additive_triples(embeddings, max_candidates=10)

    assert len(candidates) >= 1
    cand = candidates[0]
    assert set(cand.sequences) == {"A000001", "A000002", "A000003"}
