"""Unit tests for token length invariants, positional encoding headroom, and CGI bounds."""

import math
import pytest
import torch
import torch.nn as nn

from oeis_learn.decoder.environment_tracker import EnvironmentTracker, StructuralPhase
from oeis_learn.decoder.sampler import finalize_wat_tokens
from oeis_learn.decoder.wat_decoder import PositionalEncodingDecoder, WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    TOKEN_TO_ID,
    VOCAB_SIZE,
    encode_wat,
    tokenize_wat,
)
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer


def test_positional_encoding_dynamic_expansion():
    """PositionalEncodingDecoder must not crash when sequence length exceeds initial max_len."""
    pe_mod = PositionalEncodingDecoder(d_model=256, max_len=256)
    assert pe_mod.pe.size(1) >= 256

    # Test within initial bounds
    x_short = torch.randn(2, 100, 256)
    out_short = pe_mod(x_short)
    assert out_short.shape == (2, 100, 256)

    # Test exceeding initial bounds (e.g., 288 tokens like the Epoch 54 crash)
    x_long = torch.randn(2, 288, 256)
    out_long = pe_mod(x_long)
    assert out_long.shape == (2, 288, 256)
    assert pe_mod.pe.size(1) >= 288


def test_decoder_checkpoint_pe_size_compatibility():
    """WatTransformerDecoder must cleanly load state dicts with different positional buffer sizes."""
    dec_256 = WatTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        n_heads=4,
        n_decoder_layers=2,
        d_ff=512,
        max_seq_len=256,
    )
    state_256 = dec_256.state_dict()
    assert state_256["pos_encoder.pe"].shape == (1, 256, 256)

    # Initialize a decoder with max_seq_len=512
    dec_512 = WatTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        n_heads=4,
        n_decoder_layers=2,
        d_ff=512,
        max_seq_len=512,
    )

    # Loading the 256 state dict into the 512 decoder must succeed without shape mismatch
    dec_512.load_state_dict(state_256, strict=False)

    # Running a sequence of 288 tokens through dec_512 must succeed
    tokens_288 = torch.randint(0, VOCAB_SIZE, (2, 288))
    memory = torch.randn(2, 20, 256)
    logits = dec_512(tokens_288, memory)
    assert logits.shape == (2, 288, VOCAB_SIZE)


def test_finalize_wat_tokens_handles_unclosed_frames():
    """finalize_wat_tokens must balance stacks without infinite expansion."""
    tracker = EnvironmentTracker()
    tracker.reset()
    tracker.update("<bos>")
    tracker.phase = StructuralPhase.BODY
    tracker.paren_depth = 4
    tracker.result_types = ["i64", "i64", "i64", "i64"]

    # Incomplete token sequence
    initial_tokens = [BOS_ID, TOKEN_TO_ID["("], TOKEN_TO_ID["loop"], TOKEN_TO_ID["$loop"]]
    code, final_tokens = finalize_wat_tokens(initial_tokens, tracker)

    assert len(final_tokens) < 100
    assert tracker.paren_depth == 0
    assert final_tokens.count(TOKEN_TO_ID[")"]) >= 4


def test_elite_buffer_canonical_canaries_within_budget():
    """All seeded canary programs in elite buffer must be strictly within 512 tokens."""
    edb = EliteSeedDemonstrationBuffer()
    edb.seed_canonical_multilimb_canaries()

    for sid, entries in edb.canonical_archive.items():
        for entry in entries:
            toks = [BOS_ID] + encode_wat(entry.canonical_wat) + [EOS_ID]
            assert len(toks) <= 512, f"Canary {sid} exceeds 512 tokens: length={len(toks)}"
            # Multi-limb canaries must also fit within 256 tokens for efficient RL batching
            assert len(toks) <= 256, f"Canary {sid} exceeds 256 tokens: length={len(toks)}"
