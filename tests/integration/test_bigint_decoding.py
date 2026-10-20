import pytest
import torch
from oeis_learn.decoder import bigint_codec as codec
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder


def model():
    torch.manual_seed(20261018)
    return WatTransformerDecoder(
        vocab_size=codec.FOUNDATION_VOCAB_SIZE,
        d_model=16,
        n_heads=2,
        n_decoder_layers=2,
        d_ff=32,
        dropout=0,
        pad_idx=codec.PAD_ID,
        max_seq_len=4,
        codec_sha256=codec.codec_digest(),
        attention_backend="eager",
        input_version="decoder-inputs/v1",
        position_scale=16,
    ).eval()


def check_generation_batch_parity(decoder, device, prompt, temperature):
    import time
    from oeis_learn.decoder.bigint_generation import GenerationRequest, sample_batch
    from oeis_learn.fresh.conditioning_controls import condition_memory

    torch.manual_seed(20261018)
    # Different prompt contents and per-candidate RNG streams, including a tail.
    memory = torch.randn(4, 3, decoder.d_model, device=device)
    memory, _ = condition_memory(
        memory,
        torch.zeros(4, 3, dtype=torch.bool, device=device),
        prompt,
        order=(1, 2, 3, 0) if prompt == "shuffled" else None,
    )
    requests = [
        GenerationRequest(
            memory[i : i + 1], seed, temperature, time.monotonic() + 60, max_length=48
        )
        for i, seed in enumerate((17, 19, 23, 29))
    ]
    together = sample_batch(decoder, requests)
    alone = [sample_batch(decoder, [r])[0] for r in requests]
    for batch, single in zip(together, alone):
        assert (batch["tokens"], batch["complete"], batch["failure"], batch["source"]) == (
            single["tokens"],
            single["complete"],
            single["failure"],
            single["source"],
        )
    # Reordering the batch must not change a candidate's seeded stream.
    reverse = sample_batch(decoder, requests[::-1])
    assert [r["tokens"] for r in reverse[::-1]] == [r["tokens"] for r in together]


@pytest.mark.parametrize("prompt", ["correct", "shuffled", "absent"])
@pytest.mark.parametrize("temperature", [0, 0.8])
def test_mixed_prompt_batched_greedy_and_stochastic_parity(prompt, temperature):
    check_generation_batch_parity(model(), "cpu", prompt, temperature)


@pytest.mark.parametrize("prompt", ["correct", "shuffled", "absent"])
def test_parameter_teacher_forcing_eos_masks_cached_full_logits(prompt):
    decoder = model()
    source = "(program (return (add (param.get p0) (mul (param.get p0) $index))))"
    body = codec.encode_body(source)
    inputs = torch.tensor([[codec.BOS_ID] + body[:-1]])
    targets = torch.tensor([body])
    memory = torch.randn(2, 3, 16)
    from oeis_learn.fresh.conditioning_controls import condition_memory

    memory, _ = condition_memory(
        memory,
        torch.zeros(2, 3, dtype=torch.bool),
        prompt,
        order=(1, 0) if prompt == "shuffled" else None,
    )
    memory = memory[:1]
    with torch.no_grad():
        expected = decoder(inputs, memory)
        cached, cache = decoder.prefill(inputs[:, :1], memory, capacity=1024)
        logits = [cached]
        for token in inputs[:, 1:].unbind(1):
            logits.append(decoder.decode_step(token, cache))
        torch.testing.assert_close(torch.cat(logits, 1), expected, atol=2e-5, rtol=2e-4)
    state = codec.BodyState()
    for position, target in enumerate(targets[0].tolist()):
        assert target in state.allowed()
        state.consume(target)
    assert state.finished and targets[0, -1] == codec.EOS_ID
    decoder.train()
    loss = torch.nn.functional.cross_entropy(
        decoder(inputs, memory).flatten(0, 1), targets.flatten()
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(decoder.lm_head.weight.grad).all()


def test_old_vocabulary_checkpoint_never_resized():
    from oeis_learn.decoder import program_codec as old

    new = model()
    with pytest.raises(ValueError, match="codec"):
        WatTransformerDecoder(
            vocab_size=old.FOUNDATION_VOCAB_SIZE, codec_sha256=codec.codec_digest()
        )
    old_model = WatTransformerDecoder(
        vocab_size=old.FOUNDATION_VOCAB_SIZE,
        d_model=16,
        n_heads=2,
        n_decoder_layers=2,
        d_ff=32,
        dropout=0,
        codec_sha256=old.codec_digest(),
    )
    with pytest.raises(RuntimeError, match="codec|size mismatch"):
        new.load_state_dict(old_model.state_dict())


@pytest.mark.parametrize("prompt", ["correct", "shuffled", "absent"])
def test_sampler_uses_new_typed_grammar_and_never_repairs_incomplete_source(prompt):
    from oeis_learn.decoder.sampler import WatProgramSampler

    decoder = model()
    from oeis_learn.fresh.conditioning_controls import condition_memory

    memory, _ = condition_memory(
        torch.randn(2, 3, 16),
        torch.zeros(2, 3, dtype=torch.bool),
        prompt,
        order=(1, 0) if prompt == "shuffled" else None,
    )
    memory = memory[:1]
    results = []
    for cache in (False, True):
        sampler = WatProgramSampler(
            decoder, max_length=24, temperature=0, codec_profile="bigint-codec/v1", use_cache=cache
        )
        results.append(sampler.sample_trajectories(memory, [17])[0])
    assert results[0].tokens == results[1].tokens
    for result in results:
        assert len(result.tokens) <= 25
        if not result.complete:
            assert result.source == "" and result.failure == "token_cap_without_eos"


def test_codec_change_invalidates_cache():
    decoder = model()
    with torch.no_grad():
        _, cache = decoder.prefill(torch.tensor([[codec.BOS_ID]]), torch.ones(1, 1, 16))
        decoder.codec_sha256 = "sha256:" + "0" * 64
        with pytest.raises(ValueError, match="stale"):
            decoder.decode_step(torch.tensor([codec.TOKEN_TO_ID["("]]), cache)


@pytest.mark.parametrize("codec_name", ["program_codec", "bigint_codec"])
def test_explicit_codec_decoder_can_be_copied_for_frozen_reference(codec_name):
    from copy import deepcopy
    from importlib import import_module

    selected = import_module("oeis_learn.decoder." + codec_name)
    decoder = WatTransformerDecoder(
        vocab_size=selected.FOUNDATION_VOCAB_SIZE,
        codec_sha256=selected.codec_digest(),
        pad_idx=selected.PAD_ID,
        d_model=16,
        n_heads=2,
        n_decoder_layers=1,
        d_ff=32,
        dropout=0,
        position_style="prefix_ancestor",
    ).eval()
    reference = deepcopy(decoder)
    assert reference.program_codec is selected
    assert reference.codec_sha256 == decoder.codec_sha256
    assert reference.lm_head.weight.data_ptr() != decoder.lm_head.weight.data_ptr()
    tokens = torch.tensor([[selected.BOS_ID]])
    memory = torch.randn(1, 3, 16)
    with torch.no_grad():
        expected = decoder(tokens, memory)
        torch.testing.assert_close(reference(tokens, memory), expected)
        decoder.lm_head.bias.add_(1)
        torch.testing.assert_close(reference(tokens, memory), expected)
