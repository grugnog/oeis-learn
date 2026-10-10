"""Actual sampled groups cannot acquire synthetic informative advantages."""

import pytest
import torch
from oeis_learn.rl.prompt_weighting import compute_group_advantages, compute_sgrpo_advantages


def test_all_failed_groups_have_true_zero_variance():
    rewards = torch.full((8,), -1.0)
    assert torch.equal(compute_group_advantages(rewards), torch.zeros(8))
    with pytest.raises(ValueError, match="asymmetric"):
        compute_group_advantages(rewards, asymmetric_penalty_weight=1.5)


def test_standard_advantages_use_population_variance_and_zero_mean():
    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    advantages = compute_group_advantages(rewards)
    torch.testing.assert_close(advantages, (rewards - rewards.mean()) / rewards.std(unbiased=False))
    assert advantages[0] > 0 and advantages[2] < 0


def test_injected_reference_and_virtual_anchor_are_explicitly_rejected():
    rewards = torch.full((4,), -1.0)
    with pytest.raises(ValueError, match="SFT"):
        compute_sgrpo_advantages(rewards, ref_injected=True)
    with pytest.raises(ValueError, match="anchor"):
        compute_sgrpo_advantages(rewards, use_avspo_anchor=True)
    assert compute_sgrpo_advantages(rewards).abs().sum() == 0
