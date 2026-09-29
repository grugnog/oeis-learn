"""Unit tests for Difference & p-Adic Stream (S3)."""

import torch
from oeis_learn.encoder.difference_stream import (
    DifferenceStream,
    compute_p_adic_valuation,
    compute_sequence_differences_and_padic,
)


def test_padic_valuation():
    assert compute_p_adic_valuation(0, 2) == 16
    assert compute_p_adic_valuation(8, 2) == 3
    assert compute_p_adic_valuation(27, 3) == 3
    assert compute_p_adic_valuation(25, 5) == 2
    assert compute_p_adic_valuation(14, 7) == 1
    assert compute_p_adic_valuation(14, 3) == 0


def test_sequence_differences_and_padic():
    seq = [1, 2, 4, 7, 11, 16]
    d1s, d2s, pvals = compute_sequence_differences_and_padic(seq)

    assert len(d1s) == 6
    assert len(d2s) == 6
    assert len(pvals) == 6
    assert d1s[0] == 0.0
    assert d2s[0] == 0.0
    assert d2s[1] == 0.0


def test_difference_stream_forward():
    stream = DifferenceStream(d_model=64, d_padic=8)
    seqs = [[1, 2, 3, 4, 5, 6], [2, 4, 8, 16, 32, 64, 128]]
    out = stream(seqs)

    assert out.shape == (2, 7, 64)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_difference_stream_custom_constants():
    custom_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    stream = DifferenceStream(d_model=64, d_padic=8, primes=custom_primes, max_valuation=32)
    seqs = [[1, 2, 4, 8, 16, 32, 64, 128]]
    out = stream(seqs)

    assert stream.num_primes == 8
    assert stream.max_valuation == 32
    assert out.shape == (1, 8, 64)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
