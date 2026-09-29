"""End-to-end numerical stability test for Tri-Stream Encoder across 1,000 sequences."""

import random
import pytest
import torch
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder


def generate_1000_benchmark_sequences():
    """Generates 1,000 sequences with diverse dynamic ranges from -10^6 to 10^30."""
    sequences = []
    random.seed(42)

    for i in range(1000):
        length = random.randint(15, 30)
        category = i % 5

        if category == 0:
            # Small integers / polynomials
            a, b = random.randint(1, 5), random.randint(-5, 5)
            seq = [a * n * n + b * n for n in range(length)]
        elif category == 1:
            # Exponential growth (Fibonacci / Powers)
            base = random.randint(2, 4)
            seq = [base**n for n in range(length)]
        elif category == 2:
            # Extreme astronomical values (up to 10^30)
            seq = [random.randint(10**20, 10**30) * (-1 if n % 2 == 0 else 1) for n in range(length)]
        elif category == 3:
            # Negative & alternating values
            seq = [(-1) ** n * (n**3 + 10) for n in range(length)]
        else:
            # Modular & periodic sequences with zeros
            seq = [(n * 7) % 23 for n in range(length)]

        sequences.append(seq)

    return sequences


def test_tri_stream_encoder_numerical_stability():
    encoder = TriStreamEncoder(d_model=128, n_heads=4, n_encoder_layers=2, d_ff=256)
    encoder.eval()

    sequences = generate_1000_benchmark_sequences()
    batch_size = 50

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            encoded = encoder.forward_from_sequences(batch)

            assert encoded.dtype == torch.float32
            assert not torch.isnan(encoded).any()
            assert not torch.isinf(encoded).any()
            assert encoded.shape[0] == len(batch)
            assert encoded.shape[2] == 128


def test_tri_stream_encoder_aux_heads():
    encoder = TriStreamEncoder(d_model=128, n_heads=4, n_encoder_layers=2, d_ff=256)
    seqs = [[1, 2, 3, 5, 8, 13], [0, 1, 4, 9, 16, 25]]
    encoded = encoder.forward_from_sequences(seqs)
    outputs = encoder.aux_heads(encoded)

    assert "pred_magnitude" in outputs
    assert "pred_sign_logits" in outputs
    assert outputs["pred_sign_logits"].shape == (2, 6, 3)
    assert not torch.isnan(outputs["pred_magnitude"]).any()
