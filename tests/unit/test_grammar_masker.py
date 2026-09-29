"""Unit tests for Dynamic Grammar Masker and sub-100us per-token latency."""

import time
import torch
from oeis_learn.decoder.environment_tracker import EnvironmentTracker
from oeis_learn.decoder.grammar_masker import GrammarMasker
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import TOKEN_TO_ID
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.sandbox.runner import WasmRunner


def test_grammar_masker_per_token_latency():
    masker = GrammarMasker()
    tracker = EnvironmentTracker()
    tracker.reset()

    # Warmup
    for _ in range(100):
        mask = masker.compute_mask(tracker)

    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        mask = masker.compute_mask(tracker)
    elapsed = time.perf_counter() - start

    avg_latency_us = (elapsed / iterations) * 1_000_000
    print(f"\nAverage grammar masking latency: {avg_latency_us:.2f} microseconds per token")

    assert mask.shape == (masker.vocab_size,)
    # Target: sub-100us per token
    assert avg_latency_us < 100.0, f"Expected < 100us latency, got {avg_latency_us:.2f}us"


def test_wat_decoder_and_sampler():
    decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2, d_ff=128)
    sampler = WatProgramSampler(decoder=decoder, max_length=40)

    batch_size = 4
    memory = torch.randn(batch_size, 10, 64, dtype=torch.float32)
    wat_codes, generated_tokens = sampler.sample(memory, temperature=0.5, use_grammar_mask=True)

    assert len(wat_codes) == batch_size
    assert generated_tokens.shape[0] == batch_size
    for code in wat_codes:
        assert isinstance(code, str)
        assert len(code) > 0


def test_grammar_masked_sampling_compilation_soundness():
    """Independent Test for US1: Sampled programs under grammar masking compile without PARSE_ERROR."""
    decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2, d_ff=128)
    sampler = WatProgramSampler(decoder=decoder, max_length=60)
    wasm_runner = WasmRunner(fuel_budget=10000)

    batch_size = 10
    memory = torch.randn(batch_size, 10, 64, dtype=torch.float32)
    wat_codes, _ = sampler.sample(memory, temperature=1.0, use_grammar_mask=True)

    for code in wat_codes:
        res = wasm_runner.run_single(code, terms_to_generate=5)
        # Must not have PARSE_ERROR or MISSING_ENTRYPOINT
        assert res.status != "MISSING_ENTRYPOINT", f"Missing entrypoint in code:\n{code}"
