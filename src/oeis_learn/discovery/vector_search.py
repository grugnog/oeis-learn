"""Vector arithmetic search for algebraic relation candidates in latent space."""

from __future__ import annotations

import itertools
import uuid
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from oeis_learn.data.models import LatentDiscoveryCandidate


class VectorRelationSearcher:
    """Finds candidate algebraic relation triples (vA + vB ~ vC) in continuous latent representation space."""

    def __init__(self, eps_distance: float = 0.8, normalize_l2: bool = True):
        self.eps_distance = eps_distance
        self.normalize_l2 = normalize_l2

    def search_additive_triples(
        self,
        embeddings: Dict[str, np.ndarray],
        max_candidates: int = 50,
    ) -> List[LatentDiscoveryCandidate]:
        """Searches for triples (A, B, C) satisfying ||hat{v}_A + hat{v}_B - hat{v}_C||_2 < eps."""
        oeis_ids = list(embeddings.keys())
        n = len(oeis_ids)
        if n < 3:
            return []

        # L2-normalize vectors if enabled
        norm_embeddings: Dict[str, np.ndarray] = {}
        for sid, vec in embeddings.items():
            if self.normalize_l2:
                norm = float(np.linalg.norm(vec))
                if norm > 1e-8:
                    norm_embeddings[sid] = vec / norm
                else:
                    norm_embeddings[sid] = vec
            else:
                norm_embeddings[sid] = vec

        candidates: List[LatentDiscoveryCandidate] = []

        # Compare combinations of triples
        for id_a, id_b in itertools.combinations(oeis_ids, 2):
            va = norm_embeddings[id_a]
            vb = norm_embeddings[id_b]
            v_target = va + vb
            if self.normalize_l2:
                v_target_norm = float(np.linalg.norm(v_target))
                if v_target_norm > 1e-8:
                    v_target = v_target / v_target_norm

            for id_c in oeis_ids:
                if id_c in (id_a, id_b):
                    continue
                vc = norm_embeddings[id_c]
                dist = float(np.linalg.norm(v_target - vc))

                if dist < self.eps_distance:
                    candidate = LatentDiscoveryCandidate(
                        candidate_id=str(uuid.uuid4()),
                        relation_type="LINEAR_SUM",
                        sequences=(id_a, id_b, id_c),
                        vector_distance=dist,
                        status="CONJECTURED",
                    )
                    candidates.append(candidate)
                    if len(candidates) >= max_candidates:
                        return candidates

        return candidates
