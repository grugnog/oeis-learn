"""Positional checkpoint buffers preserve values, destination device and dtype."""

from contextlib import nullcontext

import pytest
import torch

from oeis_learn.decoder.wat_decoder import PositionalEncodingDecoder
from tests.unit.test_foundation_cache import decoder


@pytest.mark.parametrize("saved_length,destination_length", [(8, 8), (16, 8), (4, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_saved_positions_and_destination_dtype_survive_load(
    saved_length, destination_length, dtype
):
    source = PositionalEncodingDecoder(16, saved_length, dropout=0)
    # Saved buffers are part of the checkpoint state. Distinct saved values
    # expose accidental regeneration even when ordinary sine tables look equal.
    source.pe.add_(0.125)
    state = {name: value.clone() for name, value in source.state_dict().items()}
    target = PositionalEncodingDecoder(16, destination_length, dropout=0).to(dtype=dtype)
    headroom = target.pe[:, saved_length:].clone()
    target.load_state_dict(state, strict=True)
    assert target.pe.device.type == "cpu" and target.pe.dtype == dtype
    assert target.pe.size(1) == max(saved_length, destination_length)
    torch.testing.assert_close(target.pe[:, :saved_length], source.pe.to(dtype), rtol=0, atol=0)
    if saved_length < destination_length:
        torch.testing.assert_close(target.pe[:, saved_length:], headroom, rtol=0, atol=0)
    torch.testing.assert_close(state["pe"], source.pe, rtol=0, atol=0)
    output = target(torch.zeros(1, saved_length, 16, dtype=dtype))
    torch.testing.assert_close(output, source.pe.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("saved_length,destination_length", [(8, 8), (16, 8), (4, 8)])
def test_cpu_checkpoint_cannot_move_destination_buffer_to_cpu(saved_length, destination_length):
    source = PositionalEncodingDecoder(16, saved_length, dropout=0)
    target = PositionalEncodingDecoder(16, destination_length, dropout=0).to(
        device="meta", dtype=torch.float64
    )
    # Meta exercises the cross-device loader in CPU CI; numerical and actual
    # HIP generation/restore behavior are covered separately.
    with (
        pytest.warns(UserWarning, match="non-meta")
        if saved_length == destination_length
        else nullcontext()
    ):
        target.load_state_dict(source.state_dict())
    assert target.pe.device.type == "meta"
    assert target.pe.dtype == torch.float64
    assert target.pe.size(1) == max(saved_length, destination_length)


def test_extended_checkpoint_cached_generation_matches_full_prefix():
    source = decoder(max_seq_len=4, attention_backend="eager")
    source.pos_encoder._extend_pe(12, torch.device("cpu"))
    target = decoder(max_seq_len=8, attention_backend="eager")
    target.load_state_dict(
        {name: value.cpu().clone() for name, value in source.state_dict().items()}
    )
    torch.testing.assert_close(target.pos_encoder.pe, source.pos_encoder.pe, rtol=0, atol=0)
    tokens = torch.tensor([[1, 2, 3]])
    memory = torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(14))
    with torch.no_grad():
        expected = source(tokens, memory)
        actual = target(tokens, memory)
        cached, cache = target.prefill(tokens, memory, capacity=12)
        continued = target.decode_step(torch.tensor([4]), cache)
        full = target(torch.tensor([[1, 2, 3, 4]]), memory)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cached, actual, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(continued, full[:, -1:], rtol=1e-4, atol=1e-5)


def test_malformed_position_width_is_not_silently_regenerated():
    target = PositionalEncodingDecoder(16, 8)
    with pytest.raises(RuntimeError, match="size mismatch"):
        target.load_state_dict({"pe": torch.zeros(1, 8, 12)})
