import copy

import pytest
import torch

from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.experiments.bootstrap_models import construct_bootstrap_models
from oeis_learn.fresh.bootstrap_contract import bootstrap_profile, decoder_controls


def small(**overrides):
    torch.manual_seed(71)
    return WatTransformerDecoder(
        vocab_size=19,
        d_model=16,
        n_heads=2,
        n_decoder_layers=2,
        d_ff=32,
        dropout=0,
        pad_idx=0,
        max_seq_len=4,
        **overrides,
    ).eval()


def test_v1_frozen_and_v2_controls_change_identity_and_validate():
    assert bootstrap_profile("hybrid")["identity_sha256"] == (
        "sha256:03821e6b01bea554d8e5c8bb0aff6928c3ef6de620de567a3f797f0b266984fe"
    )
    profiles = [
        bootstrap_profile("hybrid", decoder_inputs=decoder_controls(**kw))
        for kw in (
            {},
            {"position_scale": 16},
            {"token_scale": 0.5},
            {"dropout": 0},
        )
    ]
    assert len({p["identity_sha256"] for p in profiles}) == 4
    for field, value in (
        ("token_scale", 0),
        ("token_scale", True),
        ("position_scale", -1),
        ("position_scale", float("inf")),
        ("dropout", float("nan")),
        ("dropout", 1),
    ):
        with pytest.raises(ValueError):
            decoder_controls(**{field: value})
    changed = copy.deepcopy(profiles[0])
    changed["decoder_inputs"]["position_scale"] = 16
    with pytest.raises(ValueError, match="changed"):
        construct_bootstrap_models(changed)


def test_explicit_v2_default_is_numerically_identical_to_v1_and_dropout_is_global():
    original = bootstrap_profile("hybrid")
    for selected in (decoder_controls(), decoder_controls(dropout=0)):
        torch.manual_seed(71)
        encoder, decoder, counts = construct_bootstrap_models(
            bootstrap_profile("hybrid", decoder_inputs=selected)
        )
        torch.manual_seed(71)
        old_encoder, old_decoder, old_counts = construct_bootstrap_models(original)
        assert counts == old_counts
        for a, b in zip(encoder.parameters(), old_encoder.parameters()):
            assert torch.equal(a, b)
        for a, b in zip(decoder.parameters(), old_decoder.parameters()):
            assert torch.equal(a, b)
        decoder.eval()
        old_decoder.eval()
        with torch.no_grad():
            tokens = torch.tensor([[1, 2, 3]])
            memory = torch.randn(1, 2, 256)
            torch.testing.assert_close(decoder(tokens, memory), old_decoder(tokens, memory), atol=0, rtol=0)
        for model in (encoder, decoder):
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    assert module.p == selected["dropout"]
                if isinstance(module, torch.nn.MultiheadAttention):
                    assert module.dropout == selected["dropout"]
        assert "_input_identity" not in old_decoder.state_dict()


@pytest.mark.parametrize("token_scale,position_scale", [(1, 16), (0.5, 2), (2, 0)])
def test_scaled_checkpoint_growth_and_cached_full_equivalence(
    tmp_path, token_scale, position_scale
):
    torch.set_num_threads(1)
    options = dict(
        input_version="decoder-inputs/v1", token_scale=token_scale, position_scale=position_scale
    )
    model = small(**options)
    # Force extension beyond the saved initial extent in both full and cached paths.
    tokens = torch.randint(1, 19, (1, 17))
    memory = torch.randn(1, 3, 16)
    with torch.no_grad():
        expected = model(tokens, memory)
        cached, _ = model.prefill(tokens, memory, capacity=32)
        torch.testing.assert_close(cached, expected, atol=1e-5, rtol=1e-4)
        # Positional storage remains unscaled; extension cannot reset/double the gain.
        torch.testing.assert_close(model.pos_encoder.pe[0, 0, 1::2], torch.ones(8))
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    restored = small(**options)
    restored.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens, memory), expected, atol=0, rtol=0)
        cached, cache = restored.prefill(tokens[:, :-1], memory, capacity=32)
        step = restored.decode_step(tokens[:, -1], cache)
        torch.testing.assert_close(step, expected[:, -1:], atol=1e-5, rtol=1e-4)
        # Cross the extended checkpoint boundary and compare with a fresh construction.
        restored.pos_encoder._extend_pe(600, memory.device)
        reference = small(**options)
        reference.load_state_dict(model.state_dict())
        long_tokens = tokens.repeat(1, 33)[:, :530]
        torch.testing.assert_close(
            restored(long_tokens, memory), reference(long_tokens, memory), atol=0, rtol=0
        )


@pytest.mark.parametrize(
    "destination",
    [
        {},
        {"token_scale": 0.5},
        {"position_scale": 2},
    ],
)
def test_checkpoint_refuses_implicit_reinterpretation(destination):
    versioned = small(input_version="decoder-inputs/v1", position_scale=16)
    target = small(
        **(dict(input_version="decoder-inputs/v1", **destination) if destination else {})
    )
    with pytest.raises(RuntimeError, match="decoder input"):
        target.load_state_dict(versioned.state_dict())
    with pytest.raises(RuntimeError, match="decoder input"):
        versioned.load_state_dict(small().state_dict())


def test_dropout_bound_to_checkpoint_and_live_scale_change_invalidates_cache():
    model = small(input_version="decoder-inputs/v1", position_scale=16)
    other = WatTransformerDecoder(
        vocab_size=19,
        d_model=16,
        n_heads=2,
        n_decoder_layers=2,
        d_ff=32,
        dropout=0.1,
        pad_idx=0,
        max_seq_len=4,
        input_version="decoder-inputs/v1",
        position_scale=16,
    )
    with pytest.raises(RuntimeError, match="decoder input"):
        other.load_state_dict(model.state_dict())
    with torch.no_grad():
        _, cache = model.prefill(torch.tensor([[1]]), torch.randn(1, 2, 16), capacity=4)
        model.position_scale = 8
        with pytest.raises(ValueError, match="stale"):
            model.decode_step(torch.tensor([2]), cache)
    assert not cache.valid


def test_v2_runtime_dispatch_requires_explicit_input_version():
    from oeis_learn.fresh.worker import models
    profile = bootstrap_profile("hybrid", decoder_inputs=decoder_controls(position_scale=16, dropout=0))
    encoder, decoder, counts = models(dict(model=profile, seed=71, device="cpu", cpu_threads=1))
    assert encoder.byte_gate is not None
    assert decoder.position_scale == 16 and counts["total"] > 0
    with pytest.raises(ValueError, match="require decoder-inputs/v1"):
        small(position_scale=16)
