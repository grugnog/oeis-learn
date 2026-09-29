"""Dynamic Mixture Prompt Sampler across curriculum stages."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence
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
    ):
        self.scheduler = scheduler
        self.p_active = p_active
        self.p_prev = p_prev
        self.p_earlier = p_earlier

        # Group records by stage
        self.stage_pools: Dict[int, List[SequenceRecord]] = {stage: [] for stage in range(1, 6)}
        for r in records:
            if 1 <= r.curriculum_stage <= 5:
                self.stage_pools[r.curriculum_stage].append(r)
                # Register in scheduler
                self.scheduler.register_prompt(
                    r.oeis_id, r.curriculum_stage, r.tags, r.term_count
                )

    def sample_stage(self) -> int:
        """Determines target stage for the next sample."""
        active = self.scheduler.active_stage
        if active == 1:
            return 1
        elif active == 2:
            return 2 if random.random() < 0.80 else 1
        else:
            rand = random.random()
            if rand < self.p_active:
                return active
            elif rand < (self.p_active + self.p_prev):
                return active - 1
            else:
                earlier = list(range(1, active - 1))
                return random.choice(earlier) if earlier else active

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
                batch.append(random.choice(pool))
        return batch
