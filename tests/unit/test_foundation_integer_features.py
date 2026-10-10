import pytest
import torch

from oeis_learn.encoder.integer_bytes import (
    encode_integer,
    decode_integer,
    encode_sequences,
    IntegerByteEncoder,
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
