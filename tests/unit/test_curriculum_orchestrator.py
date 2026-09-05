"""Unit tests for the adaptive curriculum orchestrator and replay management."""

from __future__ import annotations

import pytest
from oeis_learn.curriculum.orchestrator import CurriculumOrchestrator
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.curriculum.symple_bandit import AdaGGroupAllocator, Exp3SBanditScheduler
from oeis_learn.data.models import SequenceRecord
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer


def test_orchestrator_initialization_and_event_recording():
    records = [
        SequenceRecord("A000045", "Fibonacci", [0, 1, 1, 2, 3, 5, 8, 13], curriculum_stage=2),
        SequenceRecord("A000290", "Squares", [0, 1, 4, 9, 16, 25], curriculum_stage=1),
    ]
    bandit = Exp3SBanditScheduler(sequence_ids=[r.oeis_id for r in records])
    allocator = AdaGGroupAllocator(total_budget=32, min_g=8, max_g=16)
    elite_buffer = EliteSeedDemonstrationBuffer()
    scheduler = CurriculumScheduler(initial_stage=1)

    orchestrator = CurriculumOrchestrator(
        records=records,
        bandit=bandit,
        allocator=allocator,
        elite_buffer=elite_buffer,
        scheduler=scheduler,
    )

    prompts = orchestrator.select_active_prompts(batch_size=2)
    assert len(prompts) == 2

    sizes = orchestrator.allocate_rollouts(prompts, total_budget=32)
    assert sum(sizes.values()) <= 32
    for p in prompts:
        assert 8 <= sizes[p] <= 16

    # Verify event recorded
    assert len(orchestrator.events) >= 2
    event_types = {e["event_type"] for e in orchestrator.events}
    assert "TASK_SELECTED" in event_types
    assert "ROLLOUT_ALLOCATED" in event_types
