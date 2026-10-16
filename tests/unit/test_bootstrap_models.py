import pytest
import torch
from oeis_learn.experiments.bootstrap_models import (
    bootstrap_profile,
    construct_bootstrap_models,
    PRIMES,
)
from oeis_learn.encoder.hybrid import HybridEncoder


def test_explicit_matched_encoder_profiles():
    counts = []
    decoder_weights = []
    for name in ("tri_stream", "hybrid", "integer_bytes"):
        profile = bootstrap_profile(name)
        torch.manual_seed(123)
        encoder, decoder, count = construct_bootstrap_models(profile)
        decoder_weights.append(decoder.lm_head.weight.detach().clone())
        counts.append(count["decoder"])
        assert profile["primes"] == list(PRIMES)
        assert profile["base_moduli"] == list(PRIMES)
        assert profile["max_valuation"] == 32
        assert encoder.d_model == 256
        assert profile["dropout"] == 0.1
        changed = dict(profile, dropout=0.0)
        with pytest.raises(ValueError, match="changed"):
            construct_bootstrap_models(changed)
    assert len(set(counts)) == 1
    assert all(torch.equal(decoder_weights[0], weight) for weight in decoder_weights[1:])
    assert (
        len(
            {
                bootstrap_profile(k)["identity_sha256"]
                for k in ("tri_stream", "hybrid", "integer_bytes")
            }
        )
        == 3
    )


def test_hybrid_large_exact_inputs_gradient_and_padding():
    torch.set_num_threads(1)
    encoder = HybridEncoder(
        d_model=16,
        n_heads=2,
        n_encoder_layers=1,
        d_ff=32,
        dropout=0.0,
        primes=PRIMES,
        base_moduli=PRIMES,
        max_valuation=32,
    )
    rows = [[-(2**255), 2**255 - 1], [2**200 + 1]]
    output = encoder.forward_from_sequences(rows)
    assert output.shape == (2, 2, 16)
    assert torch.isfinite(output).all()
    assert torch.equal(output[1, 1], torch.zeros(16))
    output[0, 1, 0].backward()
    assert encoder.byte_projection.weight.grad.abs().sum() > 0
    assert encoder.byte_gate.grad.abs().sum() > 0
    encoder.eval()
    with torch.no_grad():
        assert torch.allclose(
            encoder.forward_from_sequences([rows[1]])[0, 0],
            encoder.forward_from_sequences(rows)[1, 0],
            atol=1e-5,
        )


def test_large_adjacent_congruences_and_difference_features_remain_explicit():
    from math import prod
    from oeis_learn.encoder.modulo_stream import compute_fourier_phase_vector
    from oeis_learn.encoder.difference_stream import compute_sequence_differences_and_padic
    from oeis_learn.encoder.magnitude_stream import compute_signed_log_scalar

    n = 2**200
    # Log scalars can collide at this scale; modular features must use integers.
    assert compute_signed_log_scalar(n) == compute_signed_log_scalar(n + 1)
    assert compute_fourier_phase_vector(n, PRIMES) != compute_fourier_phase_vector(n + 1, PRIMES)
    assert compute_fourier_phase_vector(n, PRIMES) == compute_fourier_phase_vector(
        n + prod(PRIMES), PRIMES
    )
    seq = [n + i * i for i in range(20)]
    d1, d2, valuations = compute_sequence_differences_and_padic(seq, PRIMES, 32)
    assert d1[1:] == [compute_signed_log_scalar(2 * i - 1) for i in range(1, 20)]
    assert d2[2:] == [compute_signed_log_scalar(2)] * 18
    assert all(len(v) == 16 and max(v) <= 32 for v in valuations)
