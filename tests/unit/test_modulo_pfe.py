"""Unit tests for 16-prime orthogonal Prime Fourier Embeddings (PFE)."""

from __future__ import annotations

import pytest
import torch
from oeis_learn.encoder.modulo_stream import ModuloStreamEncoder


def test_16_prime_fourier_embeddings():
    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    encoder = ModuloStreamEncoder(d_model=256, primes=primes)

    terms = [n for n in range(20)]
    batch_terms = [terms]

    # Raw PFE features have 2 * 16 = 32 dimensions (sine/cosine per prime)
    raw_pfe = encoder.compute_raw_pfe(terms)
    assert raw_pfe.shape == (20, 32)

    # Verify sine/cosine unit circle properties: cos^2 + sin^2 == 1 for every prime
    for p_idx in range(16):
        cos_col = raw_pfe[:, 2 * p_idx]
        sin_col = raw_pfe[:, 2 * p_idx + 1]
        norm_sq = cos_col**2 + sin_col**2
        assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-5)

    # Forward projection
    out = encoder.forward_from_terms(batch_terms)
    assert out.shape == (1, 20, 256)
    assert out.dtype == torch.float32
