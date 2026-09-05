"""Unit tests for continuous log-distance return and hard waste threshold gating."""

from __future__ import annotations

import pytest
from oeis_learn.rl.reward import compute_dense_log_distance_reward, compute_validity_reward


def test_dense_log_distance_reward_exact_match():
    # Exact match: |P(n) - y_n| == 0 -> log10(1) == 0 -> 1/(1+0) == 1.0
    targets = [n * 2 for n in range(20)]
    outputs = [n * 2 for n in range(20)]

    r_dense = compute_dense_log_distance_reward(outputs, targets)
    assert pytest.approx(r_dense, 1e-5) == 1.0


def test_dense_log_distance_reward_slight_deviation():
    # Target: 5n + 2, Model: 4n + 2 (slope error)
    targets = [5 * n + 2 for n in range(20)]
    outputs = [4 * n + 2 for n in range(20)]

    r_dense = compute_dense_log_distance_reward(outputs, targets)
    # Should get substantial partial credit (approx 0.50 - 0.75)
    assert 0.40 < r_dense < 0.90


def test_dense_log_distance_reward_flatline():
    # Target: 5n + 2, Model: constant 2 (flatline)
    targets = [5 * n + 2 for n in range(20)]
    outputs = [2 for _ in range(20)]

    r_dense = compute_dense_log_distance_reward(outputs, targets)
    assert 0.10 <= r_dense <= 0.45


def test_validity_reward_hard_waste_cutoff():
    # Waste ratio <= 0.30 -> exponential scaling
    # Waste ratio == 0.0 -> 0.1 * exp(0) = 0.1
    r_val_0 = compute_validity_reward(waste_ratio=0.0, threshold=0.30, kappa=2.0)
    assert pytest.approx(r_val_0, 1e-5) == 0.1

    # Waste ratio == 0.15 -> 0.1 * exp(-0.30) approx 0.074
    r_val_mid = compute_validity_reward(waste_ratio=0.15, threshold=0.30, kappa=2.0)
    assert 0.05 < r_val_mid < 0.10

    # Waste ratio == 0.31 (> 0.30) -> zeroed out
    r_val_exceeded = compute_validity_reward(waste_ratio=0.31, threshold=0.30, kappa=2.0)
    assert r_val_exceeded == 0.0
