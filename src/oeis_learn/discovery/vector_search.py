"""Bounded neighborhood/pair proposals, ranked globally with stable tie breaks.

Normalized latent addition is a retrieval heuristic, never algebraic evidence.
"""

import heapq
import time
import numpy as np
from oeis_learn.data.models import LatentDiscoveryCandidate
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest


class VectorRelationSearcher:
    def __init__(
        self,
        eps_distance=0.8,
        normalize_l2=True,
        *,
        neighbors=8,
        block_size=128,
        max_pairs=4096,
        seconds=10,
    ):
        if (
            not 1 <= neighbors <= 32
            or not 1 <= block_size <= 512
            or not 1 <= max_pairs <= 16384
            or not 0 < seconds <= 10
            or not np.isfinite(eps_distance)
            or eps_distance <= 0
        ):
            raise ValueError("bounded retrieval settings required")
        self.eps_distance, self.normalize_l2 = eps_distance, normalize_l2
        self.neighbors, self.block_size, self.max_pairs, self.seconds = (
            neighbors,
            block_size,
            max_pairs,
            seconds,
        )
        self.last_metrics = {}

    def search_additive_triples(self, embeddings, max_candidates=50):
        if type(max_candidates) is not int or not 1 <= max_candidates <= 4096:
            raise ValueError("bounded proposal count required")
        ids = sorted(embeddings)
        if len(ids) < 3:
            return []
        matrix = np.asarray([embeddings[k] for k in ids], dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] > 4096
            or not np.isfinite(matrix).all()
            or len(ids) > 10000
        ):
            raise ValueError("finite bounded feature matrix required")
        if self.normalize_l2:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        start = time.monotonic()
        deadline = start + self.seconds
        pairs = set()
        comparisons = 0
        incomplete = False
        # O(N^2) neighborhood scan with bounded blocks, retaining only k per row.
        for a in range(len(ids)):
            heap = []
            for base in range(0, len(ids), self.block_size):
                distances = np.linalg.norm(
                    matrix[base : base + self.block_size] - matrix[a], axis=1
                )
                comparisons += len(distances)
                for offset, d in enumerate(distances):
                    b = base + offset
                    if a != b:
                        item = (-float(d), -b)
                        if len(heap) < self.neighbors:
                            heapq.heappush(heap, item)
                        elif item > heap[0]:
                            heapq.heapreplace(heap, item)
            pairs.update(tuple(sorted((a, -b))) for _, b in heap)
            if len(pairs) >= self.max_pairs or time.monotonic() >= deadline:
                incomplete = True
                break
        # Stable selected budget; results ranked over every considered hypothesis.
        retained = []
        pair_count = 0
        for a, b in sorted(pairs)[: self.max_pairs]:
            if time.monotonic() >= deadline:
                incomplete = True
                break
            pair_count += 1
            target = matrix[a] + matrix[b]
            if self.normalize_l2:
                target = target / max(np.linalg.norm(target), 1e-12)
            for base in range(0, len(ids), self.block_size):
                distances = np.linalg.norm(matrix[base : base + self.block_size] - target, axis=1)
                comparisons += len(distances)
                for offset, d in enumerate(distances):
                    c = base + offset
                    if c in (a, b) or d >= self.eps_distance:
                        continue
                    triple = (ids[a], ids[b], ids[c])
                    key = (float(d), triple)
                    retained.append(key)
                retained = sorted(retained)[:max_candidates]
        self.last_metrics = {
            "pairs": pair_count,
            "distance_evaluations": comparisons,
            "block_bytes": self.block_size * matrix.shape[1] * 8,
            "elapsed_seconds": time.monotonic() - start,
            "status": "UNKNOWN" if incomplete else "COMPLETE",
            "features": "heuristic_only",
        }
        return [
            LatentDiscoveryCandidate(
                candidate_id=digest(["vector-proposal/v2", triple]),
                relation_type="LINEAR_SUM",
                sequences=triple,
                vector_distance=d,
                status="CONJECTURED",
            )
            for d, triple in retained
        ]
