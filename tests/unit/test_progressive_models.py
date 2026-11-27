"""Fresh research shapes/FP32/dropout without changing old hardware profiles."""

import copy

import pytest
import torch

from oeis_learn.fresh.bigint_config import resolve


def test_shape_resolution_is_closed_and_hardware_profiles_remain_separate():
    from oeis_learn.generalization.progressive.models import resolve_profile, validate_profile
    from oeis_learn.pipeline_validation.model_sizes import validate_profile as hardware

    recipe = resolve({"version": "bigint-bootstrap/v1"})
    profiles = {
        name: resolve_profile(recipe, shape=name, seed=13, dropout=0.1)
        for name in ("reference", "smaller_width", "shallower", "larger")
    }
    assert profiles["reference"]["model"]["d_model"] == 256
    assert profiles["smaller_width"]["model"]["d_model"] == 128
    assert profiles["shallower"]["model"]["encoder_layers"] == 2
    assert profiles["larger"]["model"]["d_model"] > 256
    for p in profiles.values():
        assert validate_profile(p) == p
        with pytest.raises(ValueError):
            hardware(p)
    from oeis_learn.pipeline_validation.identity import seal

    bad = copy.deepcopy(profiles["reference"])
    bad["model"]["d_model"] = 255
    with pytest.raises(ValueError):
        validate_profile(seal(bad))
    for kwargs in ({"dropout": True}, {"dropout": 0.5}, {"seed": True}, {"shape": "m100"}):
        with pytest.raises(ValueError):
            resolve_profile(recipe, **kwargs)


@pytest.mark.parametrize("shape", ["reference", "smaller_width", "shallower", "larger"])
def test_actual_parameter_counts_dimensions_fp32_and_effective_dropout(shape):
    from oeis_learn.generalization.progressive.models import construct, resolve_profile

    p = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}), shape=shape, seed=13, dropout=0.1
    )
    e, d, counts = construct(p, device="cpu")
    assert e.d_model == d.d_model == p["model"]["d_model"]
    assert all(v.dtype == torch.float32 for v in [*e.parameters(), *d.parameters()])
    assert counts["total_parameters"] == sum(v.numel() for v in [*e.parameters(), *d.parameters()])
    dropouts = [m.p for model in (e, d) for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropouts and set(dropouts) == {0.1}
    e.eval()
    d.eval()
    with torch.no_grad():
        memory = e.forward_from_sequences([[1, 2, 3, 4]], device="cpu")
        logits = d(torch.tensor([[1, 2, 3]]), memory)
    assert torch.isfinite(logits).all()
    if shape == "reference":
        from oeis_learn.pipeline_validation.model_sizes import PROFILES, parameters

        assert counts["total_parameters"] == parameters(PROFILES["m8"])["total"]


def test_fresh_seeded_models_dispatch_through_existing_worker():
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.fresh.bigint_worker import models
    from oeis_learn.generalization.progressive.models import resolve_profile

    recipe = resolve({"version": "bigint-bootstrap/v1"})
    a = models(resolve_profile(recipe, shape="smaller_width", seed=13), device="cpu")
    b = models(resolve_profile(recipe, shape="smaller_width", seed=13), device="cpu")
    c = models(resolve_profile(recipe, shape="smaller_width", seed=14), device="cpu")
    assert model_snapshot(*a[:2]) == model_snapshot(*b[:2])
    assert model_snapshot(*a[:2]) != model_snapshot(*c[:2])
    assert recipe == resolve({"version": "bigint-bootstrap/v1"})


def test_unsupported_device_never_falls_back_to_cpu():
    from oeis_learn.generalization.progressive.models import construct, resolve_profile

    p = resolve_profile(resolve({"version": "bigint-bootstrap/v1"}), seed=13)
    with pytest.raises(ValueError):
        construct(p, device="auto")
    if not torch.version.hip or not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="HIP"):
            construct(p, device="cuda")


def test_new_dropout_shape_performs_finite_canonical_cpu_update():
    from oeis_learn.decoder.bigint_codec import BOS_ID, PAD_ID, encode_body
    from oeis_learn.generalization.progressive.models import construct, resolve_profile
    from oeis_learn.rl.sft_trainer import supervised_step

    p = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}), shape="smaller_width", dropout=0.1, seed=13
    )
    e, d, _ = construct(p, device="cpu")
    e.train()
    d.train()
    tokens = torch.tensor([[BOS_ID, *encode_body("(program (return (add $index (int 2))))")]])
    optimizer = torch.optim.AdamW([*e.parameters(), *d.parameters()], lr=0.001, weight_decay=0.01)
    before = [v.detach().clone() for v in [*e.parameters(), *d.parameters()]]
    result = supervised_step(
        e, d, optimizer, [list(range(20))], tokens, device="cpu", pad_id=PAD_ID, observe=False
    )
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert all(torch.isfinite(v).all() for v in [*e.parameters(), *d.parameters()])
    assert any(not torch.equal(a, b) for a, b in zip(before, [*e.parameters(), *d.parameters()]))


def test_unprofiled_custom_chunk_width_stays_forbidden():
    from oeis_learn.encoder.integer_chunks import ChunkEncoder

    with pytest.raises(ValueError, match="registered chunk projection"):
        ChunkEncoder(d_model=128)


def test_reference_shape_preserves_qualified_initialization():
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.generalization.progressive.baselines import load_random
    from oeis_learn.generalization.progressive.models import construct, resolve_profile

    recipe = resolve({"version": "bigint-bootstrap/v1"})
    old = load_random(recipe, seed=13)
    new = construct(resolve_profile(recipe, seed=13), device="cpu")
    assert model_snapshot(*old[:2]) == model_snapshot(*new[:2])
