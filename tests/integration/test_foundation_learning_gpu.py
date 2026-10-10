"""Pending accelerator acceptance: never replace these cases with CPU success."""

import pytest
import torch
from oeis_learn.experiments.profiles import construct_qualified_models, qualified_model_profile

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="actual accelerator required"),
]


@pytest.mark.parametrize("position", ["sinusoidal", "rope", "prefix_ancestor"])
@pytest.mark.parametrize("length", [1, 2, 513, 1024])
def test_gpu_fp32_cached_full_prefix(length, position):
    torch.manual_seed(7)
    encoder, decoder, _ = construct_qualified_models(
        qualified_model_profile(position_style=position), device="cuda"
    )
    decoder.eval()
    encoder.eval()
    tokens = torch.randint(1, decoder.vocab_size, (1, length), device="cuda")
    memory = torch.randn(1, 20, 384, device="cuda")
    with torch.no_grad():
        full = decoder(tokens, memory)
        actual, _ = decoder.prefill(tokens, memory, capacity=1024)
    torch.testing.assert_close(actual, full, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("position", ["sinusoidal", "rope", "prefix_ancestor"])
def test_gpu_fp16_scaler_finite_optimizer_update(position):
    encoder, decoder, _ = construct_qualified_models(
        qualified_model_profile(position_style=position), device="cuda"
    )
    encoder.train()
    decoder.train()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    scaler = torch.amp.GradScaler("cuda")
    tokens = torch.randint(1, decoder.vocab_size, (2, 32), device="cuda")
    before = decoder.lm_head.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        memory = encoder.forward_from_sequences([[n * n for n in range(20)], list(range(20))])
        logits = decoder(tokens, memory)
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tokens)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert torch.isfinite(loss) and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in parameters
    )
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, decoder.lm_head.weight)
