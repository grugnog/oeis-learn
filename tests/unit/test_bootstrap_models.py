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
    for name in ("tri_stream", "hybrid", "integer_bytes"):
        profile = bootstrap_profile(name)
        encoder, decoder, count = construct_bootstrap_models(profile)
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
