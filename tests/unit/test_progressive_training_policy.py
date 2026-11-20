"""Actual CPU policy updates, schedule endpoints and complete-boundary recovery."""

import copy
import random

import numpy as np
import pytest
import torch

from oeis_learn.fresh.bigint_config import resolve
from oeis_learn.generalization.budgets import Ledger
from oeis_learn.generalization.progressive.models import resolve_profile
from oeis_learn.pipeline_validation import progressive_training as training


def profile(dropout=0):
    return resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}), shape="smaller_width", seed=13, dropout=dropout
    )


def batch(n=16):
    from oeis_learn.decoder import bigint_codec as codec

    ids = [codec.BOS_ID] + codec.encode_body("(program (return (add $index (int 1))))")
    return [[1, 2, 3, 4]] * n, torch.tensor([ids] * n), codec.PAD_ID


def policy(p, **kwargs):
    return training.resolve_update_policy(p, schedule_horizon_tokens=10000, **kwargs)


def test_schedule_has_explicit_token_horizon_warmup_and_final_rate():
    p = profile()
    cfg = policy(p, schedule="warmup_cosine", warmup_tokens=500)
    base = p["base_recipe"]["learning"]["learning_rate"]
    for tokens, factor in [(0, 0), (250, 0.5), (500, 1), (5250, 0.55), (10000, 0.1), (12000, 0.1)]:
        assert training.learning_rate_at(cfg, tokens) == pytest.approx(base * factor)
    constant = policy(p, lr_multiplier=3)
    assert training.learning_rate_at(constant, 10000) == pytest.approx(base * 3)
    # Confirmation warmup is fixed from 25M; its decay spans the 400M trajectory.
    long = training.resolve_update_policy(
        p, schedule="warmup_cosine", schedule_horizon_tokens=400_000_000, warmup_tokens=1_250_000
    )
    assert training.learning_rate_at(long, 25_000_000) > base * 0.9
    assert training.learning_rate_at(long, 400_000_000) == pytest.approx(base * 0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"effective_batch": 8},
        {"effective_batch": True},
        {"microbatch_size": 0},
        {"microbatch_size": 17},
        {"lr_multiplier": 2},
        {"lr_multiplier": True},
        {"weight_decay": 0.02},
        {"weight_decay": True},
        {"schedule": "linear"},
        {"schedule": "constant", "warmup_tokens": 1},
        {"schedule": "warmup_cosine", "warmup_tokens": 10000},
    ],
)
def test_unsupported_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        policy(profile(), **kwargs)


@pytest.mark.parametrize("n", [16, 32, 64])
def test_actual_effective_batches_use_one_adamw_and_schedule_step(n):
    p = profile(0.1)
    cfg = policy(p, effective_batch=n, microbatch_size=7, weight_decay=0.1)
    runner = training.PolicyUpdate(p, cfg, device="cpu")
    x, t, pad = batch(n)
    charge = int(t[:, 1:].ne(pad).sum())
    ledger = Ledger(10000).reserve("batch-0", charge)
    before = next(runner.decoder.parameters()).detach().clone()
    result, committed = runner.update(x, t, ledger=ledger)
    assert committed.updates == runner.scheduler.updates == 1
    assert committed.consumed == runner.scheduler.tokens == charge
    assert result["examples"] == n and result["charged_tokens"] == charge
    assert result["microbatches"] == (n + 6) // 7
    assert not torch.equal(before, next(runner.decoder.parameters()))
    assert all(int(v["step"]) == 1 for v in runner.optimizer.state.values())
    assert runner.optimizer.param_groups[0]["weight_decay"] == 0.1
    assert runner.counts["effective_dropout"] == 0.1
    assert runner.state_dict()["scheduler"]["updates"] == 1


def test_interruption_keeps_debit_blocks_snapshot_and_restores_rng_optimizer_schedule(
    tmp_path, monkeypatch
):
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore

    p = profile(0.2)
    cfg = policy(p, microbatch_size=7, schedule="warmup_cosine")
    runner = training.PolicyUpdate(p, cfg, device="cpu")
    x, t, pad = batch()
    charge = int(t[:, 1:].ne(pad).sum())
    ledger = Ledger(10000)
    store = GeneralizationCheckpointStore(tmp_path, cfg["identity"], ledger.cap)
    states = {
        k: {}
        for k in (
            "weights",
            "optimizer",
            "scheduler",
            "rng",
            "data",
            "replay",
            "history",
            "pins",
            "selection",
        )
    }
    states["data"] = {"cursor": 0, "pending_batch": None}
    states["weights"] = {"e": runner.encoder.state_dict(), "d": runner.decoder.state_dict()}
    states["optimizer"] = runner.state_dict()
    states["rng"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "device": [],
    }
    cut = store.save(states, ledger)
    reserved = ledger.reserve("interrupted", charge)
    original = runner.encoder.forward_from_sequences
    calls = []

    def interrupt(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            with pytest.raises(ValueError, match="complete"):
                runner.state_dict()
            raise InterruptedError("partial batch")
        return original(*a, **kw)

    monkeypatch.setattr(runner.encoder, "forward_from_sequences", interrupt)
    with pytest.raises(InterruptedError):
        runner.update(x, t, ledger=reserved)
    assert reserved.consumed == charge and reserved.updates == 0
    assert runner.scheduler.updates == 0
    with pytest.raises(ValueError, match="complete"):
        runner.state_dict()
    with pytest.raises(ValueError, match="restore"):
        runner.update(x, t, ledger=reserved)
    restored = store.load(cut)["states"]
    a = training.PolicyUpdate(p, cfg, device="cpu")
    b = training.PolicyUpdate(p, cfg, device="cpu")
    # Retry is charged again, while both arms restore exactly the same durable cut.
    retry = Ledger(10000, consumed=charge).reserve("retry", charge)
    outputs = []
    for model in (a, b):
        model.encoder.load_state_dict(restored["weights"]["e"])
        model.decoder.load_state_dict(restored["weights"]["d"])
        model.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["rng"]["torch"])
        report, done = model.update(x, t, ledger=retry)
        outputs.append((report["loss"], done, torch.get_rng_state()))
    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][1].consumed == 2 * charge
    assert torch.equal(outputs[0][2], outputs[1][2])
    assert all(torch.equal(u, v) for u, v in zip(a.decoder.parameters(), b.decoder.parameters()))
    states["weights"] = {"e": a.encoder.state_dict(), "d": a.decoder.state_dict()}
    states["optimizer"] = a.state_dict()
    states["rng"]["torch"] = outputs[0][2]
    states["data"]["cursor"] = 1
    complete_cut = store.save(states, outputs[0][1])
    complete = store.load(complete_cut)["states"]
    saved = complete["optimizer"]
    restored_runner = training.PolicyUpdate(p, cfg, device="cpu")
    restored_runner.encoder.load_state_dict(complete["weights"]["e"])
    restored_runner.decoder.load_state_dict(complete["weights"]["d"])
    restored_runner.load_state_dict(saved)
    assert restored_runner.scheduler.tokens == 2 * charge and restored_runner.scheduler.updates == 1
    next_ledger = outputs[0][1].reserve("next", charge)
    results = []
    for model in (a, restored_runner):
        torch.set_rng_state(complete["rng"]["torch"])
        random.setstate(complete["rng"]["python"])
        np.random.set_state(complete["rng"]["numpy"])
        report, _ = model.update(x, t, ledger=next_ledger)
        results.append(report["loss"])
    assert results[0] == results[1]
    assert all(
        torch.equal(u, v)
        for u, v in zip(a.decoder.parameters(), restored_runner.decoder.parameters())
    )


def test_policy_and_reservation_cannot_silently_change():
    from oeis_learn.pipeline_validation.identity import seal

    p = profile()
    cfg = policy(p)
    bad = seal(dict(cfg, effective_batch=8))
    with pytest.raises(ValueError):
        training.PolicyUpdate(p, bad, device="cpu")
    runner = training.PolicyUpdate(p, cfg, device="cpu")
    x, t, _ = batch()
    for ledger in (Ledger(10000), Ledger(10000).reserve("bad", 1), Ledger(1).reserve("large", 100)):
        with pytest.raises(ValueError):
            runner.update(x, t, ledger=ledger)
    assert runner.scheduler.updates == 0
    assert not runner.optimizer.state


def test_allocation_advances_once_per_complete_update_and_never_for_partial(monkeypatch):
    from oeis_learn.generalization.progressive.curriculum import Controller

    p = profile()
    cfg = policy(p, microbatch_size=7)
    runner = training.PolicyUpdate(p, cfg, device="cpu")
    layout = {
        "bins": ["familiar", "frontier"],
        "probes": {
            "familiar": {"role": "training", "id": "probe-a"},
            "frontier": {"role": "training", "id": "probe-b"},
        },
    }
    controller = Controller(layout, seed=13)
    controller.select()
    x, t, pad = batch()
    charge = int(t[:, 1:].ne(pad).sum())
    _, done = runner.update(
        x, t, ledger=Ledger(10000).reserve("one", charge), allocation=controller
    )
    assert controller.updates == controller.block_position == runner.scheduler.updates == 1
    assert controller.charged_tokens == charge
    before = controller.snapshot()

    def failed(*args, **kwargs):
        raise InterruptedError("not a committed allocation step")

    monkeypatch.setattr(runner.encoder, "forward_from_sequences", failed)
    reserved = done.reserve("two", charge)
    with pytest.raises(InterruptedError):
        runner.update(x, t, ledger=reserved, allocation=controller)
    assert controller.snapshot() == before
    assert reserved.consumed == 2 * charge and runner.scheduler.updates == 1


def test_restore_rejects_changed_optimizer_semantics_and_schedule():
    p = profile()
    cfg = policy(p)
    a = training.PolicyUpdate(p, cfg, device="cpu")
    state = a.state_dict()
    for mutate in (
        lambda s: s["optimizer"]["param_groups"][0].update(weight_decay=0.5),
        lambda s: s["scheduler"].update(policy="changed"),
        lambda s: s.update(extra="unknown"),
    ):
        bad = copy.deepcopy(state)
        mutate(bad)
        b = training.PolicyUpdate(p, cfg, device="cpu")
        with pytest.raises(ValueError):
            b.load_state_dict(bad)
