import math
import pytest
import torch

from oeis_learn.encoder.large_integer_features import signed_log, features, PRIMES
from oeis_learn.encoder.integer_chunks import encode_chunks, ChunkEncoder


def test_finite_logs_exact_differences_residues_and_zero_valuations():
    for bits in (256, 512, 1024, 4096):
        a = (1 << bits) - 1
        assert math.isfinite(signed_log(a))
        assert abs(signed_log(a) - (1 + bits * math.log10(2))) < 1e-10
        f = features([-a, a, 0])
        assert f["differences"] == [0, 2 * a, -a]
        assert f["residues"][0] == [(-a) % p for p in PRIMES]
        assert f["valuations"][-1] == [32] * 16
    assert signed_log(0) == 0
    with pytest.raises(ValueError):
        features([1.5])


def test_chunk_boundaries_sign_zero_and_padding():
    x = encode_chunks([[0, 1 << 63, 1 << 64], [-((1 << 4096) - 1), (1 << 4096) - 1]])
    assert x.magnitudes.shape[-1] == 8
    assert x.lengths[0, :3].tolist() == [1, 8, 9]
    assert x.difference_lengths[1, 1].item() == 513
    assert x.signs[0, 0].item() == 0
    assert x.padding.tolist() == [[False, False, False], [False, False, True]]
    assert x.chunk_lengths[0, 2, :2].tolist() == [8, 1]


def test_chunk_encoder_padding_invariance_width_independent_and_backward():
    torch.manual_seed(8)
    model = ChunkEncoder(dropout=0).eval()
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    x = encode_chunks([[1, (1 << 4096) - 1]])
    y = model(x)
    assert y.shape == (1, 2, 256) and torch.isfinite(y).all()
    y.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    alone = model(encode_chunks([[0, 1, 256]]))
    batch = model(encode_chunks([[0, 1, 256], [(1 << 4096) - 1] * 5]))
    torch.testing.assert_close(alone[0], batch[0, :3], atol=2e-6, rtol=2e-5)
    assert shapes == {n: tuple(p.shape) for n, p in model.named_parameters()}
    assert model.position.pe.device == next(model.parameters()).device


def test_full_hybrid_finite_update_and_frozen_feature_constants():
    from oeis_learn.encoder.hybrid import ChunkHybridEncoder
    from oeis_learn.fresh.bootstrap_contract import PRIMES as frozen

    assert list(PRIMES) == list(frozen)
    encoder = ChunkHybridEncoder(dropout=0).eval()
    rows = [[0, -((1 << 4096) - 1), (1 << 4096) - 1], [1, 2]]
    output = encoder.forward_from_sequences(rows)
    assert output.shape == (2, 3, 256) and torch.isfinite(output).all()
    assert not output[1, 2].any()
    output[0, 2, 0].backward()
    assert encoder.chunk_encoder.projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())


@pytest.mark.parametrize(
    "actual,requested,current,accepted",
    [
        ("cuda:0", "cuda", 0, True),
        ("cuda:1", "cuda", 1, True),
        ("cuda:0", "cuda:0", 1, True),
        ("cuda:0", None, 1, True),
        ("cuda:0", "cuda", 1, False),
        ("cuda:0", "cuda:1", 0, False),
        ("cuda:0", "cpu", 0, False),
        ("cpu", "cpu", 0, True),
        ("cpu", "cuda", 0, False),
    ],
)
def test_chunk_hybrid_device_binding(monkeypatch, actual, requested, current, accepted):
    from types import SimpleNamespace
    from oeis_learn.encoder.hybrid import ChunkHybridEncoder
    from oeis_learn.encoder import integer_chunks

    # Exercise the real forward entry guard without allocating an accelerator
    # tensor. Stop at encoding and verify its device is the model's concrete one.
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device(actual))]))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)

    class ReachedEncoding(Exception):
        pass

    def encode(sequences, *, device):
        assert accepted
        assert device == torch.device(actual)
        raise ReachedEncoding

    monkeypatch.setattr(integer_chunks, "encode_chunks", encode)
    if accepted:
        with pytest.raises(ReachedEncoding):
            ChunkHybridEncoder.forward_from_sequences(model, [[1]], device=requested)
    else:
        with pytest.raises(ValueError, match="explicit device differs from model device"):
            ChunkHybridEncoder.forward_from_sequences(model, [[1]], device=requested)


def test_old_width_mathematical_feature_tolerance():
    from oeis_learn.encoder.magnitude_stream import compute_signed_log_scalar
    from oeis_learn.encoder.difference_stream import compute_sequence_differences_and_padic

    values = [0, 1, -1, 1 << 255, (1 << 255) + 1, -(1 << 255)]
    f = features(values)
    assert f["magnitudes"] == pytest.approx(
        [compute_signed_log_scalar(n) for n in values], abs=1e-12
    )
    a, b, p = compute_sequence_differences_and_padic(values, PRIMES, 32)
    assert f["difference_logs"] == pytest.approx(a, abs=1e-12)
    assert f["second_difference_logs"] == pytest.approx(b, abs=1e-12)
    assert f["valuations"] == p
