"""Unit tests for synthetic multi-limb demonstration generation with signed coefficients."""

from __future__ import annotations

import pytest
from oeis_learn.data.synthetic_generator import SyntheticDemonstrationGenerator


def test_multilimb_demonstration_generation():
    gen = SyntheticDemonstrationGenerator(seed=123)

    samples = []
    for i in range(35):
        s = gen.generate_multilimb_sample(i)
        if s is not None:
            samples.append(s)

    assert len(samples) >= 20

    # Verify all samples have 20 terms
    for s in samples:
        assert len(s.terms) == 20
        assert len(s.wat_code) > 0

    # Verify at least some samples contain negative terms (10.2% corpus representation)
    has_negative_terms = any(any(t < 0 for t in s.terms) for s in samples)
    assert has_negative_terms is True

    # Verify diversity across families
    families = {s.family for s in samples}
    assert len(families) >= 3
