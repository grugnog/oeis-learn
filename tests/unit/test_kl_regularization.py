"""Unit tests for unbiased Schulman per-token KL divergence and entropy bonus."""

import pytest
import torch
from oeis_learn.rl.egca_grpo import (
    compute_egca_grpo_loss,
    compute_policy_entropy,
    compute_schulman_kl,
)


def test_schulman_kl_identical_distributions():
    log_pi = torch.tensor([-0.5, -1.2, -2.0])
    # When pi == pi_ref, KL must be exactly 0.0
    kl = compute_schulman_kl(log_pi, log_pi)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_schulman_kl_divergent_distributions():
    log_pi = torch.tensor([-0.1, -2.5])
    log_pi_ref = torch.tensor([-1.5, -0.2])

    kl = compute_schulman_kl(log_pi, log_pi_ref)
    # KL must be non-negative
    assert (kl >= 0.0).all()
    assert (kl > 0.0).any()


def test_policy_entropy_calculation():
    # Uniform logits -> maximum entropy
    vocab_size = 10
    uniform_logits = torch.zeros(2, 5, vocab_size)
    entropy = compute_policy_entropy(uniform_logits)

    expected_max_entropy = torch.log(torch.tensor(float(vocab_size)))
    assert torch.isclose(entropy, expected_max_entropy, atol=1e-4)


def test_loss_incorporates_kl_and_entropy_bonus():
    group_size, seq_len, vocab_size = 2, 4, 16
    logits = torch.randn(group_size, seq_len, vocab_size)
    old_log_probs = torch.randn(group_size, seq_len)
    token_ids = torch.randint(0, vocab_size, (group_size, seq_len))
    advantages = torch.tensor([0.5, -0.5])

    loss_with_kl = compute_egca_grpo_loss(
        logits=logits,
        old_log_probs=old_log_probs,
        token_ids=token_ids,
        advantages=advantages,
        beta_kl=0.1,
        reference_logits=logits + torch.randn_like(logits) * 0.2,
        alpha_ent=0.05,
    )

    assert isinstance(loss_with_kl, torch.Tensor)
    assert loss_with_kl.ndim == 0
    assert not torch.isnan(loss_with_kl)


def test_sampled_stale_reference_cannot_masquerade_as_exact_kl():
    with pytest.raises(ValueError, match="full reference"):
        compute_egca_grpo_loss(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1),
            torch.ones(1, 1, dtype=torch.long),
            torch.ones(1),
            ref_log_probs=torch.zeros(1, 1),
        )
