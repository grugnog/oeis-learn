"""Unit tests for Modulo-Spectrum Stream (S2)."""

import math
import torch
from oeis_learn.encoder.modulo_stream import BASE_MODULI, ModuloSpectrumStream, compute_fourier_phase_vector


def test_fourier_phase_vector_bounds():
    vec = compute_fourier_phase_vector(42)
    assert len(vec) == 200  # 100 moduli * 2
    for val in vec:
        assert -1.0001 <= val <= 1.0001

    # When x = 0, sin(0) = 0 and cos(0) = 1 for all moduli
    vec_zero = compute_fourier_phase_vector(0)
    for i in range(100):
        assert math.isclose(vec_zero[2 * i], 0.0, abs_tol=1e-7)
        assert math.isclose(vec_zero[2 * i + 1], 1.0, abs_tol=1e-7)


def test_modulo_spectrum_stream_forward():
    stream = ModuloSpectrumStream(d_model=64)
    seqs = [[0, 1, 1, 2, 3, 5, 8], [10, 20, 30, 40]]
    out = stream(seqs)

    assert out.shape == (2, 7, 64)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_modulo_spectrum_stream_custom_moduli():
    custom_moduli = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    stream = ModuloSpectrumStream(d_model=64, base_moduli=custom_moduli)
    assert stream.moduli_count == 10
    assert stream.input_dim == 20

    seqs = [[1, 2, 3, 4, 5]]
    out = stream(seqs)
    assert out.shape == (1, 5, 64)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
