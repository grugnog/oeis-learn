"""Unit tests for EXP3.S bandit scheduler and Ada-G dynamic group allocator."""

from __future__ import annotations

import pytest
from oeis_learn.curriculum.symple_bandit import AdaGGroupAllocator, Exp3SBanditScheduler


def test_exp3s_cold_start_probabilities():
    tasks = ["A000045", "A000032", "A000129", "A000290"]
    bandit = Exp3SBanditScheduler(sequence_ids=tasks, gamma=0.15, alpha=0.05)
    probs = bandit.get_selection_probabilities()

    assert len(probs) == 4
    # Equal uniform distribution at cold start
    for sid in tasks:
        assert pytest.approx(probs[sid], abs=1e-4) == 0.25
    assert pytest.approx(sum(probs.values()), abs=1e-4) == 1.0


def test_adag_exact_budget_fill():
    tasks = ["A000045", "A000032"]
    bandit = Exp3SBanditScheduler(sequence_ids=tasks)
    allocator = AdaGGroupAllocator(total_budget=32, min_g=8, max_g=16)

    sizes = allocator.compute_group_sizes(prompts=tasks, bandit=bandit, total_budget=32)
    assert len(sizes) == 2
    # Sum must be <= total_budget and fill it properly
    assert sum(sizes.values()) == 32
    for s in sizes.values():
        assert 8 <= s <= 16


def test_adag_allocates_deeper_for_harder_tasks():
    tasks = ["hard_task", "easy_task"]
    bandit = Exp3SBanditScheduler(sequence_ids=tasks)
    # Simulate high success on easy task, low success on hard task
    bandit.update_feedback("easy_task", success_count=15, group_size=16)
    bandit.update_feedback("hard_task", success_count=0, group_size=16)

    allocator = AdaGGroupAllocator(total_budget=24, min_g=8, max_g=16)
    sizes = allocator.compute_group_sizes(prompts=tasks, bandit=bandit, total_budget=24)

    assert sizes["hard_task"] >= sizes["easy_task"]
    assert sum(sizes.values()) <= 24


def test_adag_handles_perfect_and_zero_competence():
    """Verify that p_hat = 1.0 (100% success) does not trigger math domain error."""
    tasks = ["perfect_task", "zero_task"]
    bandit = Exp3SBanditScheduler(sequence_ids=tasks)
    bandit.update_feedback("perfect_task", success_count=20, group_size=20)
    bandit.update_feedback("zero_task", success_count=0, group_size=20)

    assert bandit.get_competence("perfect_task") == 1.0
    assert bandit.get_competence("zero_task") == 0.0

    allocator = AdaGGroupAllocator(total_budget=32, min_g=8, max_g=16)
    sizes = allocator.compute_group_sizes(prompts=tasks, bandit=bandit, total_budget=32)

    assert len(sizes) == 2
    assert sum(sizes.values()) <= 32
    for s in sizes.values():
        assert 8 <= s <= 16
    assert sizes["zero_task"] >= sizes["perfect_task"]
