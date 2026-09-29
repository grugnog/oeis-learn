"""Unit tests for Hierarchical FiLM Fusion block."""

import pytest
import torch
from oeis_learn.encoder.film_fusion import HierarchicalFilmFusion


def test_film_fusion_forward():
    fusion = HierarchicalFilmFusion(d_model=64)
    batch_size, seq_len = 4, 12

    s1 = torch.randn(batch_size, seq_len, 64, dtype=torch.float32)
    s2 = torch.randn(batch_size, seq_len, 64, dtype=torch.float32)
    s3 = torch.randn(batch_size, seq_len, 64, dtype=torch.float32)

    z = fusion(s1, s2, s3)

    assert z.shape == (batch_size, seq_len, 64)
    assert z.dtype == torch.float32
    assert not torch.isnan(z).any()
    assert not torch.isinf(z).any()


def test_film_fusion_gradient():
    fusion = HierarchicalFilmFusion(d_model=64)
    s1 = torch.randn(2, 5, 64, dtype=torch.float32, requires_grad=True)
    s2 = torch.randn(2, 5, 64, dtype=torch.float32, requires_grad=True)
    s3 = torch.randn(2, 5, 64, dtype=torch.float32, requires_grad=True)

    z = fusion(s1, s2, s3)
    loss = z.sum()
    loss.backward()

    assert s1.grad is not None and not torch.isnan(s1.grad).any()
    assert s2.grad is not None and not torch.isnan(s2.grad).any()
    assert s3.grad is not None and not torch.isnan(s3.grad).any()
