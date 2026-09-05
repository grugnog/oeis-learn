"""Unit tests for synthetic dataset procedural generator with randomized affine scaling sweeps."""

from __future__ import annotations

import pytest
from oeis_learn.data.synthetic_generator import SyntheticDemonstrationGenerator


def test_synthetic_generator_affine_sweeps():
    gen = SyntheticDemonstrationGenerator(
        seed=42,
        enable_affine_sweeps=True,
        scale_min_pow=0.0,
        scale_max_pow=4.0,
    )

    dataset = gen.generate_dataset(num_samples=100)
    assert len(dataset.samples) == 100

    # Verify that multiplier constants span diverse ranges across samples
    max_terms = [max(abs(x) for x in sample.terms) for sample in dataset.samples]
    # Some samples should have terms > 100 due to affine scaling
    assert any(m > 50 for m in max_terms)
