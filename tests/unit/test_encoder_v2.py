"""Unit tests for Tri-Stream Encoder v2 normalized Newton forward differences."""

from __future__ import annotations

import pytest
import torch
from oeis_learn.encoder.difference_stream import DifferenceStreamEncoder


def test_newton_forward_difference_quotients():
    encoder = DifferenceStreamEncoder(d_model=256)

    # Quadratic sequence: a(n) = 3n^2 - 4n + 7
    # y = [7, 6, 11, 22, 39, 62, 91, 126, 167, 214, 267, 326, 391, 462, 539, 622, 711, 806, 907, 1014]
    terms = [3 * n * n - 4 * n + 7 for n in range(20)]
    batch_terms = [terms]

    # Compute raw Newton quotients
    diffs = encoder.compute_newton_quotients(terms)
    # diffs has shape (20, 3) where columns are D^(1), D^(2), D^(3)
    # D^(2) = Delta^2 y / 2! = (6) / 2 = 3.0 for all n
    d2_vals = diffs[:, 1]
    assert pytest.approx(d2_vals[:-2].mean().item(), 1e-4) == 3.0
    # D^(3) = Delta^3 y / 6! = 0.0 for all n
    d3_vals = diffs[:, 2]
    assert pytest.approx(d3_vals[:-3].abs().max().item(), 1e-4) == 0.0

    # Test forward projection
    out = encoder.forward_from_terms(batch_terms)
    assert out.shape == (1, 20, 256)
    assert out.dtype == torch.float32
