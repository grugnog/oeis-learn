"""Unit tests for Ada-G dynamic group sizing and Virtual Sample Injection."""

from __future__ import annotations

import pytest
from oeis_learn.curriculum.symple_bandit import AdaGGroupAllocator, Exp3SBanditScheduler
from oeis_learn.rl.egca_grpo import inject_virtual_sample_if_needed


def test_adag_group_allocation():
    seq_ids = ["A000001", "A000002"]
    bandit = Exp3SBanditScheduler(sequence_ids=seq_ids)

    # Simulate A000001 as hard frontier (p=0.05), A000002 as easier (p=0.50)
    for _ in range(19):
        bandit.update_feedback("A000001", success_count=0, group_size=1)
    bandit.update_feedback("A000001", success_count=1, group_size=1)

    for _ in range(10):
        bandit.update_feedback("A000002", success_count=1, group_size=2)
        bandit.update_feedback("A000002", success_count=0, group_size=2)

    allocator = AdaGGroupAllocator(total_budget=32, min_g=8, max_g=16)
    group_sizes = allocator.compute_group_sizes(
        prompts=["A000001", "A000002"],
        bandit=bandit,
    )

    assert "A000001" in group_sizes and "A000002" in group_sizes
    # Harder prompt should get larger group size (up to max_g=16)
    assert group_sizes["A000001"] >= group_sizes["A000002"]
    assert group_sizes["A000001"] <= 16
    assert group_sizes["A000002"] >= 8
    assert sum(group_sizes.values()) <= 32


def test_virtual_sample_injection_all_failure():
    # When all rollouts fail and sequence has EDB solution
    group_rewards = [0.0, 0.0, 0.0, 0.0]
    injected = inject_virtual_sample_if_needed(group_rewards, has_edb_solution=True)
    assert len(injected) == 5
    assert injected[-1] == 1.0


def test_virtual_sample_injection_no_edb():
    group_rewards = [0.0, 0.0, 0.0, 0.0]
    injected = inject_virtual_sample_if_needed(group_rewards, has_edb_solution=False)
    assert len(injected) == 4
    assert injected == group_rewards
