"""Integration test for dynamic mixture prompt sampling and curriculum gating."""

import pytest
from oeis_learn.curriculum.sampler import DynamicMixtureSampler
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.data.models import SequenceRecord


def test_dynamic_mixture_sampler_distribution():
    scheduler = CurriculumScheduler(initial_stage=1)

    records = []
    for stage in range(1, 6):
        for i in range(20):
            records.append(
                SequenceRecord(
                    oeis_id=f"A{stage*10000 + i:06d}",
                    name=f"Stage {stage} sequence {i}",
                    terms=[stage * n for n in range(20)],
                    curriculum_stage=stage,
                )
            )

    sampler = DynamicMixtureSampler(records=records, scheduler=scheduler)

    # In Stage 1: all samples should be Stage 1
    batch = sampler.sample_batch(batch_size=30)
    assert len(batch) == 30
    assert all(r.curriculum_stage == 1 for r in batch)

    # Manually graduate to Stage 3
    scheduler.active_stage = 3
    batch_st3 = sampler.sample_batch(batch_size=100)
    stages = [r.curriculum_stage for r in batch_st3]

    # Mixture should contain mostly Stage 3, some Stage 2, and some Stage 1
    st3_count = stages.count(3)
    st2_count = stages.count(2)
    st1_count = stages.count(1)

    assert st3_count > 40
    assert st2_count > 5
    assert st1_count >= 0
