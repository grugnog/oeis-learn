"""Unit tests for Partitioned Semantic Policy Entropy and stack-depth temperature scaling."""

from __future__ import annotations

import pytest
import torch
from oeis_learn.decoder.wat_grammar import TOKEN_TO_ID
from oeis_learn.rl.egca_grpo import compute_partitioned_semantic_entropy, get_dynamic_sampling_temperature


def test_dynamic_sampling_temperature_decay():
    # Stack height 0: T == 0.80 * (1 - 0) = 0.80
    t_0 = get_dynamic_sampling_temperature(stack_height=0, max_stack=16, t_base=0.80, decay=0.60)
    assert pytest.approx(t_0, 1e-4) == 0.80

    # Stack height 8 (mid): T == 0.80 * (1 - 0.60 * 8/16) = 0.80 * 0.70 = 0.56
    t_mid = get_dynamic_sampling_temperature(stack_height=8, max_stack=16, t_base=0.80, decay=0.60)
    assert pytest.approx(t_mid, 1e-4) == 0.56

    # Stack height 16 (max): T == 0.80 * (1 - 0.60 * 1) = 0.80 * 0.40 = 0.32
    t_max = get_dynamic_sampling_temperature(stack_height=16, max_stack=16, t_base=0.80, decay=0.60)
    assert pytest.approx(t_max, 1e-4) == 0.32


def test_partitioned_semantic_entropy_calculation():
    batch_size = 2
    vocab_size = 50
    logits = torch.randn(batch_size, 10, vocab_size)
    valid_mask = torch.zeros(batch_size, 10, vocab_size, dtype=torch.bool)
    valid_mask[:, :, :10] = True  # First 10 tokens valid

    sem_indices = [0, 1, 2, 3, 4]  # Arithmetic & variable ops
    struct_indices = [5, 6]  # drop, nop

    loss_ent = compute_partitioned_semantic_entropy(
        logits=logits,
        valid_mask=valid_mask,
        sem_indices=sem_indices,
        struct_indices=struct_indices,
        alpha_sem=0.02,
        beta_pen=0.05,
    )
    assert loss_ent.dim() == 0  # Scalar loss tensor
    assert not torch.isnan(loss_ent)
