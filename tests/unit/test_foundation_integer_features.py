from types import SimpleNamespace

import pytest
import torch

from oeis_learn.encoder.integer_bytes import (
    IntegerByteEncoder,
    decode_integer,
    encode_integer,
    encode_sequences,
)


def test_boundaries_neighbors_and_wide_differences():
    values = [-(1 << 255), -(1 << 255) + 1, -1, 0, 1, (1 << 255) - 2, (1 << 255) - 1]
    assert all(decode_integer(*encode_integer(v)) == v for v in values)
    assert len({encode_integer(v) for v in values}) == len(values)
    batch = encode_sequences([[values[0], values[-1]], [0]])
    assert (
        decode_integer(int(batch.difference_signs[0, 1]), batch.differences[0, 1].tolist())
        == (1 << 256) - 1
    )
    assert batch.valuations[1, 0] == 257 and batch.padding.tolist() == [
        [False, False],
        [False, True],
    ]
    assert batch.indices.tolist() == [[0, 1], [0, 1]]
    with pytest.raises(ValueError):
        decode_integer(1, [0] * 32)
    for value in [True, 1.0, 1 << 255, -(1 << 255) - 1]:
        with pytest.raises(ValueError):
            encode_sequences([[value]])


def test_all_declared_paths_have_finite_gradients_and_padding_is_inert():
    model = IntegerByteEncoder(16, 2, 1, 32)
    output = model.forward_from_sequences([[0, 1, -(1 << 255), (1 << 255) - 1], [2, 3]])
    assert torch.isfinite(output).all() and not output[1, 2:].any()
    output.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_visible_term_cap_cannot_enable_hidden_continuations():
    for cap in (0, 21, True, 20.0):
        with pytest.raises(ValueError, match="cap"):
            encode_sequences([[0] * 21], max_terms=cap)
    with pytest.raises(ValueError, match="1..20"):
        encode_sequences([[0] * 21])


@pytest.mark.parametrize(
    "actual,current,requested,accepted",
    [
        ("cuda:0", 0, "cuda", True),
        ("cuda:2", 2, torch.device("cuda"), True),
        ("cuda:2", 0, "cuda:2", True),
        ("cuda:2", 0, None, True),
        ("cpu", 0, "cpu", True),
        ("cuda:0", 2, "cuda", False),
        ("cuda:0", 0, "cuda:1", False),
        ("cuda:0", 0, "cpu", False),
        ("cpu", 0, "cuda", False),
    ],
)
def test_requested_device_resolves_current_gpu_and_rejects_mismatch(
    monkeypatch, actual, current, requested, accepted
):
    # Exercise the real boundary without requiring a GPU. Only tensor creation
    # and forward math are replaced; placement checking must happen first.
    model = IntegerByteEncoder(16, 2, 1, 32)
    monkeypatch.setattr(
        model, "parameters", lambda: iter([SimpleNamespace(device=torch.device(actual))])
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    calls = []
    batch = object()

    def encode(sequences, *, device):
        calls.append((sequences, device))
        return batch

    monkeypatch.setattr("oeis_learn.encoder.integer_bytes.encode_sequences", encode)
    monkeypatch.setattr(model, "forward", lambda value: value)
    if accepted:
        assert model.forward_from_sequences([[1, 2]], device=requested) is batch
        assert calls == [([[1, 2]], torch.device(actual))]
    else:
        with pytest.raises(ValueError, match="encoder and requested device differ"):
            model.forward_from_sequences([[1, 2]], device=requested)
        assert not calls
