"""Bounded actual-parent compatibility checks; no sustained launcher claim."""

import os
from pathlib import Path
import pytest
import torch
from oeis_learn.evaluation.checkpoint import load_checkpoint_v2, save_checkpoint_v2
from oeis_learn.evaluation.legacy_wrappers import load_legacy_configuration
from oeis_learn.experiments.artifacts import compute_file_hash


def inputs():
    parent = os.environ.get("OEIS_LEGACY_PARENT")
    config = os.environ.get("OEIS_LEGACY_CONFIG")
    if not parent or not config:
        pytest.skip("Actual legacy parent/configuration required by later artifact stage")
    device = os.environ.get("OEIS_LEGACY_DEVICE", "cpu")
    if device == "cuda":
        assert torch.cuda.is_available() and torch.version.hip, "Actual HIP required"
    load_legacy_configuration(config)
    e, d, identity = load_checkpoint_v2(parent, device=torch.device(device))
    assert identity.epoch > 0, "Actual trained parent required"
    out = Path(os.environ["OEIS_LEGACY_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    return parent, e, d, identity, device, out


def update(e, d, device):
    from oeis_learn.rl.sft_trainer import supervised_step
    from oeis_learn.decoder.wat_grammar import BOS_ID, EOS_ID, PAD_ID, TOKEN_TO_ID

    e.train()
    d.train()
    optimizer = torch.optim.AdamW([*e.parameters(), *d.parameters()], lr=1e-4)
    tokens = torch.tensor(
        [[BOS_ID, TOKEN_TO_ID["i64.const"], TOKEN_TO_ID["7"], EOS_ID]], device=device
    )
    before = [p.detach().clone() for p in d.parameters()]
    result = supervised_step(e, d, optimizer, [[7] * 20], tokens, device=device, pad_id=PAD_ID)
    assert result["loss"] > 0 and result["gradient_norm"] >= 0
    assert any(not torch.equal(a, b) for a, b in zip(before, d.parameters()))
    assert all(torch.isfinite(p).all() for p in [*e.parameters(), *d.parameters()])
    return result


def test_legacy_train():
    parent, e, d, identity, device, out = inputs()
    update(e, d, device)
    cp = out / "trained.v2.pt"
    saved = save_checkpoint_v2(
        str(cp),
        e,
        d,
        identity.encoder_config,
        identity.decoder_config,
        identity.epoch + 1,
        source_checkpoint_sha256=compute_file_hash(parent),
    )
    assert saved.source_checkpoint_sha256 == compute_file_hash(parent)
    assert load_checkpoint_v2(str(cp), device=torch.device(device))[2].epoch == identity.epoch + 1


def test_legacy_resume():
    parent, e, d, identity, device, out = inputs()
    # v2 is weights-only: compare reproducible next updates after reloading the
    # same bytes; do not claim optimizer/RNG restoration.
    cp = out / "trained.v2.pt"
    assert cp.is_file(), "Run the owning train case first"
    e, d, _ = load_checkpoint_v2(str(cp), device=torch.device(device))
    torch.manual_seed(7001)
    first = update(e, d, device)
    expected = [p.detach().clone() for p in [*e.parameters(), *d.parameters()]]
    e, d, _ = load_checkpoint_v2(str(cp), device=torch.device(device))
    torch.manual_seed(7001)
    second = update(e, d, device)
    assert first["loss"] == pytest.approx(second["loss"], abs=1e-6)
    for a, b in zip(expected, [*e.parameters(), *d.parameters()]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_legacy_warmstart():
    parent, e, d, identity, device, out = inputs()
    from oeis_learn.cli.train_warmstart import run_warmstart_pipeline

    result = run_warmstart_pipeline(parent, str(out / "warmstart"), stage="surgery", dry_run=True)
    assert (
        result["status"] == "SUCCESS"
        and result["purpose"] == "legacy_unverified"
        and not result["qualified"]
    )
    payload = torch.load(result["checkpoint"], map_location=device, weights_only=False)
    assert all(torch.equal(v, payload["encoder_state_dict"][k]) for k, v in e.state_dict().items())


def test_legacy_evaluate():
    parent, e, d, identity, device, out = inputs()
    from oeis_learn.cli.train_warmstart import evaluate_bridge_model

    e.train()
    d.eval()
    modes = {m: m.training for model in (e, d) for m in model.modules()}
    result = evaluate_bridge_model(e, d, device, [[7] * 20, list(range(20))])
    assert result["candidate_origin"] == "model_sample" and not result["qualified"]
    assert result["sampling_seeds"] == [[7001, 7002, 7003, 7004], [7005, 7006, 7007, 7008]]
    assert all(m.training == mode for m, mode in modes.items())
