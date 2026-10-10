import pytest
import torch
from oeis_learn.experiments.profiles import qualified_model_profile, construct_qualified_models
from oeis_learn.decoder.program_codec import BOS_ID, encode_body


@pytest.mark.parametrize("position", ["sinusoidal", "rope", "prefix_ancestor"])
def test_declared_qualified_model_forward_backward(position):
    profile = qualified_model_profile(position_style=position)
    encoder, decoder, count = construct_qualified_models(profile)
    assert 20_000_000 < count["total"] < 30_000_000
    memory = encoder.forward_from_sequences([[0, 1, 2], [1, 2, 3]])
    tokens = torch.tensor([[BOS_ID] + encode_body("i256.const 1")[:-1]] * 2)
    logits = decoder(tokens, memory)
    loss = logits.square().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in list(encoder.parameters()) + list(decoder.parameters())
    )
    assert count["total"] == count["encoder"] + count["decoder"]
