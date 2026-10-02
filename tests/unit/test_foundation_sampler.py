"""Real CPU PyTorch tests of the existing sampler's explicit body-codec path."""

import copy
import inspect

import pytest
import torch

from oeis_learn.decoder import program_codec as c
from oeis_learn.decoder.environment_tracker import FoundationEnvironmentTracker
from oeis_learn.decoder.grammar_masker import GrammarMasker
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.experiments.config import load_config


class ScriptedDecoder(torch.nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens
        self.codec_sha256 = c.codec_digest()

    def forward(self, tokens, memory):
        logits = torch.full((1, tokens.size(1), c.FOUNDATION_VOCAB_SIZE), -100.0)
        logits[0, -1, self.tokens[min(tokens.size(1) - 1, len(self.tokens) - 1)]] = 100.0
        return logits.to(memory.device)


def test_existing_sampler_generates_exact_body_without_repair():
    ids = c.encode_body("i256.const -12345678901234567890")
    model = ScriptedDecoder(ids)
    sampler = WatProgramSampler(model, codec_profile="wat_body_decimal_v1", temperature=0)
    before = torch.get_rng_state().clone()
    source, actual = sampler.sample_candidate(torch.zeros(1, 20, 8), seed=23)
    assert actual.tolist() == ids
    assert source == c.decode_body(ids)
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(c.CodecError, match="cap"):
        sampler.sample_candidate(torch.zeros(1, 20, 8), seed=23, max_length=2)
    assert model.training
    with pytest.raises(c.CodecError):
        sampler.sample_candidate(torch.zeros(1, 20, 8), seed=23, prefix_wat="i256.zero")
    with pytest.raises(c.CodecError):
        sampler.sample_candidate(torch.zeros(1, 20, 8), seed=23, use_grammar_mask=False)


def test_high_id_mask_and_wrong_mask_width():
    tracker = FoundationEnvironmentTracker()
    mask = GrammarMasker(c.FOUNDATION_VOCAB_SIZE).compute_mask(tracker)
    assert mask.shape == (c.FOUNDATION_VOCAB_SIZE,)
    assert mask[c.TOKEN_TO_ID["i256.zero"]] == 0
    assert c.TOKEN_TO_ID["i256.zero"] >= 128
    assert torch.isneginf(mask[c.EOS_ID])
    with pytest.raises(ValueError, match="vocabularies"):
        GrammarMasker(128).compute_mask(tracker)


def decoder(**kwargs):
    return WatTransformerDecoder(
        vocab_size=c.FOUNDATION_VOCAB_SIZE,
        d_model=8,
        n_heads=2,
        n_decoder_layers=1,
        d_ff=16,
        pad_idx=c.PAD_ID,
        **kwargs,
    )


def test_codec_identity_rejects_legacy_or_tampered_weights_even_nonstrict():
    model = decoder(codec_sha256=c.codec_digest())
    state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    legacy = decoder()
    with pytest.raises(RuntimeError, match="legacy"):
        model.load_state_dict(legacy.state_dict(), strict=False)
    with pytest.raises(RuntimeError, match="legacy"):
        legacy.load_state_dict(state, strict=False)
    state["_codec_identity"][0] = 0
    with pytest.raises(RuntimeError, match="identity"):
        model.load_state_dict(state)
    with pytest.raises(ValueError, match="identity"):
        WatProgramSampler(legacy, codec_profile="wat_body_decimal_v1")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or torch.version.hip is None,
                reason="requires AMD HIP GPU via lab-gpu; CPU diagnostics are not hardware evidence",
            ),
        ),
    ],
)
def test_complete_effective_constructors_and_real_forward_backward(tmp_path, device):
    cfg = load_config("configs/foundation/wat_smoke.yaml")
    kwargs = cfg.model_constructor_kwargs()
    for key, cls in [("encoder", TriStreamEncoder), ("decoder", WatTransformerDecoder)]:
        assert set(kwargs[key]) == set(inspect.signature(cls).parameters)
    # Small diagnostic; not a GPU, full-context or training qualification gate.
    for values in kwargs.values():
        values.update(d_model=8, n_heads=2, d_ff=16)
    kwargs["encoder"]["n_encoder_layers"] = 1
    kwargs["decoder"]["n_decoder_layers"] = 1
    encoder, model = (
        TriStreamEncoder(**kwargs["encoder"]).to(device),
        WatTransformerDecoder(**kwargs["decoder"]).to(device),
    )
    assert encoder.s2_modulo.base_moduli == list(range(2, 102))
    assert encoder.s3_diff_padic.primes == [2, 3, 5, 7, 11, 13]
    assert model.token_embedding.padding_idx == c.PAD_ID
    memory = encoder.forward_from_sequences(
        [[0] * 20, [2**255 - 1] * 20], device=torch.device(device)
    )
    if isinstance(memory, tuple):
        memory = memory[0]
    ids = torch.tensor([[c.BOS_ID, c.TOKEN_TO_ID["i256.zero"]]] * 2, device=device)
    assert memory.device.type == torch.device(device).type
    assert next(model.parameters()).device.type == torch.device(device).type
    loss = model(ids, memory).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert cfg.persist_effective(tmp_path).as_path().is_file()
