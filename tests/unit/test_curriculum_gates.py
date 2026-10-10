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
                    oeis_id=f"A{stage * 10000 + i:06d}",
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


def test_mixture_custom_stage_two_weights_and_bound_resume():
    records = [
        SequenceRecord(oeis_id=f"A{stage:06d}", name="", terms=[stage], curriculum_stage=stage)
        for stage in (1, 2)
    ]
    scheduler = CurriculumScheduler(initial_stage=2)
    sampler = DynamicMixtureSampler(records, scheduler, p_active=0, p_prev=1, p_earlier=0, seed=7)
    assert all(sampler.sample_stage() == 1 for _ in range(30))
    state = sampler.state_dict()
    expected = [r.oeis_id for r in sampler.sample_batch(12)]
    sampler.load_state_dict(state)
    assert [r.oeis_id for r in sampler.sample_batch(12)] == expected
    changed = DynamicMixtureSampler(records[:1], scheduler, p_active=0, p_prev=1, p_earlier=0)
    with pytest.raises(ValueError, match="incompatible"):
        changed.load_state_dict(state)
    for probabilities in [(0.5, 0.5, 0.5), (-0.1, 0.6, 0.5), (float("nan"), 0.2, 0.1)]:
        with pytest.raises(ValueError, match="probabilities"):
            DynamicMixtureSampler(records, scheduler, *probabilities)
