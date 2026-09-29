"""Unit tests for sampler determinism, candidate-local generators, and budget prefixes."""

from __future__ import annotations

import pytest
import torch
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.evaluation.protocol import derive_candidate_seed


@pytest.fixture
def tiny_decoder():
    torch.manual_seed(100)
    decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2, d_ff=128)
    decoder.eval()
    return decoder


def test_sampler_candidate_seed_determinism(tiny_decoder):
    sampler = WatProgramSampler(decoder=tiny_decoder, max_length=32)
    memory = torch.randn(1, 20, 64)

    seed_0 = 12345
    # Two separate sample calls with candidate-local seed 12345
    progs1, tokens1 = sampler.sample_candidate(memory, seed=seed_0, temperature=0.8, top_p=0.95)
    progs2, tokens2 = sampler.sample_candidate(memory, seed=seed_0, temperature=0.8, top_p=0.95)

    assert progs1 == progs2
    assert torch.equal(tokens1, tokens2)


def test_budget_prefix_stability(tiny_decoder):
    """Budget 1 is prefix of Budget 8, which is prefix of Budget 16."""
    sampler = WatProgramSampler(decoder=tiny_decoder, max_length=32)
    memory = torch.randn(1, 20, 64)
    base_seed = 999
    protocol_id = "sha256:" + "0" * 64
    seq_id = "A000045"

    seeds_16 = [derive_candidate_seed(base_seed, protocol_id, seq_id, i) for i in range(16)]

    candidates_16 = []
    for s in seeds_16:
        wat, tok = sampler.sample_candidate(memory, seed=s, temperature=0.8, top_p=0.95)
        candidates_16.append(wat)

    # Sample budget 1 candidate
    cand_1, _ = sampler.sample_candidate(memory, seed=seeds_16[0], temperature=0.8, top_p=0.95)
    assert cand_1 == candidates_16[0]

    # Sample budget 8 candidates
    cands_8 = [
        sampler.sample_candidate(memory, seed=seeds_16[i], temperature=0.8, top_p=0.95)[0]
        for i in range(8)
    ]
    assert cands_8 == candidates_16[:8]
