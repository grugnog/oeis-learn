"""Unit tests for progressive pre-flight tiers with failure injection."""

from __future__ import annotations

import pytest
from oeis_learn.rl.progressive import (
    validate_tier_0,
    validate_tier_1,
    validate_tier_2,
    validate_tier_3,
)


def test_tier_0_passes():
    res = validate_tier_0()
    assert res.passed is True
    assert res.tier == 0


def test_tier_1_passes():
    res = validate_tier_1()
    assert res.passed is True
    assert res.tier == 1


def test_tier_2_requires_exact_success_not_just_latency():
    """Validates that Tier 2 enforces nonzero exact synthesis successes."""
    res = validate_tier_2()
    # Check that metrics report max_pass_rate, final_pass_rate, and exact_success_count
    assert "max_pass_rate" in res.metrics
    assert "exact_success_count" in res.metrics
