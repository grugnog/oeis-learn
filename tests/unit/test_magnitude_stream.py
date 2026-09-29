"""Unit tests for Magnitude Stream (S1)."""

import math
import torch
from oeis_learn.encoder.magnitude_stream import MagnitudeStream, compute_signed_log_scalar


def test_signed_log_scalar_values():
    assert compute_signed_log_scalar(0) == 0.0
    assert compute_signed_log_scalar(1) == 1.0 + math.log10(2.0)
    assert compute_signed_log_scalar(-1) == -(1.0 + math.log10(2.0))
    assert compute_signed_log_scalar(9) == 2.0
    assert compute_signed_log_scalar(-9) == -2.0

    # Test extreme astronomical values
    huge_pos = 10**25
    huge_neg = -(10**25)
    v_pos = compute_signed_log_scalar(huge_pos)
    v_neg = compute_signed_log_scalar(huge_neg)

    assert v_pos > 25.0
    assert v_neg < -25.0
    assert math.isclose(v_pos, -v_neg, rel_tol=1e-5)


def test_magnitude_stream_module():
    stream = MagnitudeStream(d_model=64, d_ff=128)
    seqs = [[0, 1, 1, 2, 3, 5, 8, 13, 10**10, -(10**12)], [1, 2, 4, 8, 16, 32]]
    out = stream(seqs)

    assert out.shape == (2, 10, 64)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
