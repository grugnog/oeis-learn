"""Unit tests for global summary tokens (z_affine, z_geom) and auxiliary regression heads."""

from __future__ import annotations

import pytest
import torch
from oeis_learn.encoder.heads import SummaryRegressionHeads
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder


def test_summary_regression_heads():
    heads = SummaryRegressionHeads(d_model=256)

    # z_affine and z_geom embeddings: (batch_size, d_model)
    z_affine = torch.randn(2, 256)
    z_geom = torch.randn(2, 256)

    pred_slope, pred_ratio = heads(z_affine, z_geom)
    assert pred_slope.shape == (2, 1)
    assert pred_ratio.shape == (2, 1)

    target_slope = torch.tensor([[5.0], [2.0]])
    target_ratio = torch.tensor([[1.0], [3.0]])

    loss = heads.compute_auxiliary_loss(pred_slope, pred_ratio, target_slope, target_ratio)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_tri_stream_encoder_v2_forward_with_summary_tokens():
    encoder = TriStreamEncoder(d_model=256, n_heads=4, n_encoder_layers=2, enable_summary_tokens=True)

    # Linear sequence: a(n) = 5n + 2
    terms = [5 * n + 2 for n in range(20)]
    batch_terms = [terms]

    # Output should include 2 summary tokens + 20 sequence tokens = 22 tokens in length
    z = encoder.forward_from_sequences(batch_terms)
    assert z.shape == (1, 22, 256)
    assert z.dtype == torch.float32
