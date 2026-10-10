"""Dynamic Mixture Prompt Sampler across curriculum stages."""

from __future__ import annotations

import random
import math

from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from typing import Dict, List, Sequence
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.data.models import SequenceRecord


class DynamicMixtureSampler:
    """Samples prompts according to a 70/20/10 dynamic stage mixture:

    - 70% Active stage k
    - 20% Previous stage k - 1
    - 10% Earlier stages <= k - 2
    """

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        scheduler: CurriculumScheduler,
        p_active: float = 0.70,
        p_prev: float = 0.20,
        p_earlier: float = 0.10,
        seed: int = 20260913,
    ):
        probabilities = (p_active, p_prev, p_earlier)
        if any(not math.isfinite(p) or p < 0 for p in probabilities) or not math.isclose(
            sum(probabilities), 1.0, abs_tol=1e-12
        ):
            raise ValueError("mixture probabilities must be finite, nonnegative and sum to one")
        if len({r.oeis_id for r in records}) != len(records):
            raise ValueError("duplicate mixture prompt IDs")
        self.pool_sha256 = digest(
            sorted(
                [(r.oeis_id, r.curriculum_stage, r.terms, r.tags) for r in records],
                key=lambda value: value[0],
            )
        )
        self.scheduler = scheduler
        self.p_active = p_active
        self.p_prev = p_prev
        self.p_earlier = p_earlier
        self.rng = random.Random(seed)

        # Group records by stage
        self.stage_pools: Dict[int, List[SequenceRecord]] = {stage: [] for stage in range(1, 6)}
        for r in records:
            if 1 <= r.curriculum_stage <= 5:
                self.stage_pools[r.curriculum_stage].append(r)
                # Register in scheduler
                self.scheduler.register_prompt(r.oeis_id, r.curriculum_stage, r.tags, r.term_count)

    def sample_stage(self) -> int:
        """Determines target stage for the next sample."""
        active = self.scheduler.active_stage
        if active == 1:
            return 1
        elif active == 2:
            return 2 if self.rng.random() < self.p_active + self.p_earlier else 1
        else:
            rand = self.rng.random()
            if rand < self.p_active:
                return active
            elif rand < (self.p_active + self.p_prev):
                return active - 1
            else:
                earlier = list(range(1, active - 1))
                return self.rng.choice(earlier) if earlier else active

    def sample_batch(self, batch_size: int = 16) -> List[SequenceRecord]:
        """Samples a batch of sequence records matching the mixture distribution."""
        batch: List[SequenceRecord] = []
        for _ in range(batch_size):
            stage = self.sample_stage()
            pool = self.stage_pools.get(stage)
            if not pool:
                # Fallback to active stage pool
                pool = self.stage_pools.get(self.scheduler.active_stage, [])
            if pool:
                batch.append(self.rng.choice(pool))
        return batch

    def state_dict(self):
        return {
            "version": "mixture-sampler/v2",
            "pool_sha256": self.pool_sha256,
            "rng": self.rng.getstate(),
            "mixture": [self.p_active, self.p_prev, self.p_earlier],
        }

    def load_state_dict(self, state):
        if (
            state["version"] != "mixture-sampler/v2"
            or state.get("pool_sha256") != self.pool_sha256
            or state["mixture"]
            != [
                self.p_active,
                self.p_prev,
                self.p_earlier,
            ]
        ):
            raise ValueError("incompatible mixture state")
        self.rng.setstate(state["rng"])
