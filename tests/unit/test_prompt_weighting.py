"""Unit tests for asymmetric prompt weighting in GRPO."""

import torch
from oeis_learn.rl.prompt_weighting import compute_group_advantages


def test_asymmetric_prompt_weighting_failed_group():
    # When all 8 rollouts fail (reward = -1.0)
    rewards = torch.full((8,), -1.0)
    advantages = compute_group_advantages(rewards, asymmetric_penalty_weight=1.5)

    # Advantages must be non-zero negative (not zero!)
    assert advantages.shape == (8,)
    assert (advantages == -1.5).all()


def test_standard_advantages_mixed_group():
    # 2 successes (+1.0), 6 failures (-1.0)
    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    advantages = compute_group_advantages(rewards)

    assert advantages[0] > 0.0
    assert advantages[1] > 0.0
    assert advantages[2] < 0.0


def test_sgrpo_advantages_with_cgi_injection():
    from oeis_learn.rl.prompt_weighting import compute_sgrpo_advantages

    # Injected reference at index 0 (r=+1.0), followed by 4 failed rollouts (r=-1.0)
    rewards = torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0])
    advs = compute_sgrpo_advantages(rewards, ref_injected=True)

    assert advs[0] > 0.0  # Injected reference receives positive advantage
    assert (advs[1:] < 0.0).all()  # Failed samples receive negative advantages


def test_sgrpo_advantages_with_avspo_virtual_anchor():
    from oeis_learn.rl.prompt_weighting import compute_sgrpo_advantages

    # All 4 rollouts fail, no reference injected -> AVSPO virtual anchor
    rewards = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    advs = compute_sgrpo_advantages(rewards, ref_injected=False, use_avspo_anchor=True)

    # Advantages must be non-zero negative
    assert (advs < 0.0).all()
    assert not torch.isnan(advs).any()
