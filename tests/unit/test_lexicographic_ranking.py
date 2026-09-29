"""Unit tests for Covariant Parsimony Pressure (CPP) and Lexicographical Group Ranking."""

from __future__ import annotations

import pytest
from oeis_learn.rl.reward import (
    compute_covariant_parsimony_penalty,
    compute_lexicographic_advantages,
)


def test_covariant_parsimony_penalty_bloat():
    # Bloat scenario: Longer programs have equal or worse rewards
    # Lengths: [10, 20, 30, 40], Rewards: [0.5, 0.5, 0.4, 0.3], Waste ratios: [0.0, 0.1, 0.2, 0.3]
    lengths = [10, 20, 30, 40]
    rewards = [0.5, 0.5, 0.4, 0.3]
    waste_ratios = [0.0, 0.1, 0.2, 0.3]

    cpp_rewards = compute_covariant_parsimony_penalty(lengths, rewards, waste_ratios, lambda_waste=0.20)
    assert len(cpp_rewards) == 4
    # The shortest program should receive the highest net return
    assert cpp_rewards[0] > cpp_rewards[1] > cpp_rewards[2] > cpp_rewards[3]


def test_covariant_parsimony_penalty_progress():
    # Progress scenario: Longer programs have higher rewards (synthesizing loops)
    # Lengths: [10, 20, 30, 40], Rewards: [0.2, 0.4, 0.7, 0.9], Waste ratios: [0.0, 0.0, 0.0, 0.0]
    lengths = [10, 20, 30, 40]
    rewards = [0.2, 0.4, 0.7, 0.9]
    waste_ratios = [0.0, 0.0, 0.0, 0.0]

    cpp_rewards = compute_covariant_parsimony_penalty(lengths, rewards, waste_ratios, lambda_waste=0.20)
    # Positive covariance means no length penalty is applied (penalty is 0)
    assert cpp_rewards[3] == 0.9
    assert cpp_rewards[0] == 0.2


def test_lexicographic_group_ranking_dominance():
    # Candidates: (reward, opt_length)
    # Candidate 0: (1.0, 15) -> Solved, compact -> Rank 1 (best)
    # Candidate 1: (1.0, 25) -> Solved, bloated -> Rank 2
    # Candidate 2: (0.1, 5)  -> Unsolved, very short -> Rank 3 (functional correctness strictly dominates!)
    # Candidate 3: (0.1, 20) -> Unsolved, longer -> Rank 4
    group_results = [
        (1.0, 15),
        (1.0, 25),
        (0.1, 5),
        (0.1, 20),
    ]

    advs = compute_lexicographic_advantages(group_results)
    assert len(advs) == 4
    # Check ordinal ranking order: adv[0] > adv[1] > adv[2] > adv[3]
    assert advs[0] > advs[1] > advs[2] > advs[3]
    assert pytest.approx(advs[0], 1e-4) == 1.0  # Best rank -> +1.0
    assert pytest.approx(advs[3], 1e-4) == -1.0  # Worst rank -> -1.0
