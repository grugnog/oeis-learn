"""Unit tests for EXP3.S non-stationary bandit scheduler and learning progress feedback."""

from __future__ import annotations

import pytest
from oeis_learn.curriculum.symple_bandit import Exp3SBanditScheduler


def test_exp3s_bandit_initialization():
    seq_ids = [f"A{i:06d}" for i in range(1, 21)]
    bandit = Exp3SBanditScheduler(sequence_ids=seq_ids, gamma=0.15, alpha=0.05)

    assert len(bandit.sequence_ids) == 20
    assert len(bandit.weights) == 20
    # Initially uniform
    prompts = bandit.sample_active_prompts(batch_size=2)
    assert len(prompts) == 2
    assert len(set(prompts)) == 2


def test_exp3s_learning_progress_feedback():
    seq_ids = ["A000001", "A000002", "A000003"]
    bandit = Exp3SBanditScheduler(sequence_ids=seq_ids, gamma=0.15, alpha=0.05)

    # A000001: 10 attempts alternating 1 and 0 -> p=0.5 (peak dispersion)
    # A000002: 10 successes out of 10 -> p=1.0 (mastered)
    # A000003: 0 successes out of 10 -> p=0.0 (unmastered)
    for _ in range(5):
        bandit.update_feedback("A000001", success_count=1, group_size=1)
        bandit.update_feedback("A000001", success_count=0, group_size=1)

    for _ in range(10):
        bandit.update_feedback("A000002", success_count=1, group_size=1)

    for _ in range(10):
        bandit.update_feedback("A000003", success_count=0, group_size=1)

    probs = bandit.get_selection_probabilities()
    # A000001 (frontier, p=0.5) should have higher sampling weight than mastered or unmastered
    assert probs["A000001"] > probs["A000002"]
    assert probs["A000001"] > probs["A000003"]
    # Verify non-zero exploration floor
    assert all(p > 0.0 for p in probs.values())
