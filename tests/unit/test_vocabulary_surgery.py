"""Unit tests for vocabulary surgery, convex hull projection, and logit soft-capping."""

from __future__ import annotations

import torch
import pytest
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.vocabulary_surgery import perform_vocabulary_surgery, SEMANTIC_ANCESTORS


def test_vocabulary_surgery_expansion_and_norm_calibration():
    old_vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "result", "i64", "$a", "$temp", "func", "i64.add"]
    old_token_to_id = {tok: i for i, tok in enumerate(old_vocab)}

    new_vocab = list(old_vocab) + ["result_i64_x4", "i256.add", "$a0", "$t0"]
    new_token_to_id = {tok: i for i, tok in enumerate(new_vocab)}

    d_model = 64
    decoder = WatTransformerDecoder(
        vocab_size=len(old_vocab),
        d_model=d_model,
        n_heads=2,
        n_decoder_layers=2,
        d_ff=128,
        pad_idx=0,
    )

    old_embed_copy = decoder.token_embedding.weight.data.clone()
    old_head_copy = decoder.lm_head.weight.data.clone()
    mean_old_head_norm = float(torch.norm(old_head_copy, p=2, dim=1).mean().item())

    # Perform surgery
    perform_vocabulary_surgery(
        decoder=decoder,
        old_vocab_size=len(old_vocab),
        new_vocab_size=len(new_vocab),
        old_token_to_id=old_token_to_id,
        new_token_to_id=new_token_to_id,
        logit_cap_threshold=30.0,
    )

    assert decoder.vocab_size == len(new_vocab)
    assert decoder.token_embedding.weight.shape == (len(new_vocab), d_model)
    assert decoder.lm_head.weight.shape == (len(new_vocab), d_model)

    # Check preserved weights
    assert torch.allclose(decoder.token_embedding.weight.data[:len(old_vocab)], old_embed_copy)
    assert torch.allclose(decoder.lm_head.weight.data[:len(old_vocab)], old_head_copy)

    # Check result_i64_x4 convex hull: 0.50 * result + 0.50 * i64
    res_idx = new_token_to_id["result_i64_x4"]
    expected_embed = 0.50 * old_embed_copy[old_token_to_id["result"]] + 0.50 * old_embed_copy[old_token_to_id["i64"]]
    assert torch.allclose(decoder.token_embedding.weight.data[res_idx], expected_embed, atol=1e-5)

    # Check norm calibration: new head rows should be close to mean_old_head_norm
    for idx in range(len(old_vocab), len(new_vocab)):
        head_norm = float(torch.norm(decoder.lm_head.weight.data[idx], p=2).item())
        assert abs(head_norm - mean_old_head_norm) < 1e-4


def test_logit_soft_capping():
    decoder = WatTransformerDecoder(
        vocab_size=32,
        d_model=64,
        n_heads=2,
        n_decoder_layers=2,
        d_ff=128,
        logit_cap_threshold=30.0,
    )

    tgt = torch.randint(0, 32, (2, 10))
    mem = torch.randn(2, 5, 64)

    logits = decoder(tgt, mem)
    # With C_cap = 30.0, max absolute logit must be strictly <= 30.0
    assert float(logits.abs().max().item()) <= 30.0
