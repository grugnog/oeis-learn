"""Pending accelerator acceptance: never replace these cases with CPU success."""

import pytest
import torch
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="actual accelerator required"),
]


@pytest.mark.parametrize("position", ["sinusoidal", "rope", "prefix_ancestor"])
@pytest.mark.parametrize("length", [1, 2, 513, 1024])
def test_gpu_fp32_cached_full_prefix(length, position):
    torch.manual_seed(7)
    decoder = (
        WatTransformerDecoder(
            d_model=384,
            n_heads=6,
            n_decoder_layers=6,
            d_ff=1536,
            dropout=0,
            max_seq_len=1024,
            position_style=position,
        )
        .cuda()
        .eval()
    )
    tokens = torch.randint(1, decoder.vocab_size, (1, length), device="cuda")
    memory = torch.randn(1, 20, 384, device="cuda")
    with torch.no_grad():
        full = decoder(tokens, memory)
        actual, _ = decoder.prefill(tokens, memory, capacity=1024)
    torch.testing.assert_close(actual, full, atol=1e-5, rtol=1e-4)


def test_gpu_fp16_scaler_finite_optimizer_update():
    decoder = (
        WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2, d_ff=128, dropout=0)
        .cuda()
        .train()
    )
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda")
    tokens = torch.randint(1, decoder.vocab_size, (2, 32), device="cuda")
    memory = torch.randn(2, 20, 64, device="cuda")
    before = decoder.lm_head.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = decoder(tokens, memory)
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tokens)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert torch.isfinite(loss) and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters()
    )
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, decoder.lm_head.weight)
