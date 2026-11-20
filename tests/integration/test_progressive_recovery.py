"""Actual tensor publication failures and exact discovery cut restoration."""

import random

import numpy as np
import pytest
import torch

from oeis_learn.fresh.bigint_state import ProgressiveCheckpointStore
from oeis_learn.generalization.budgets import Ledger
from oeis_learn.generalization.progressive.contracts import digest


def states():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.01)
    parameter.square().sum().backward()
    optimizer.step()
    return {
        "weights": {"parameter": parameter},
        "optimizer": optimizer.state_dict(),
        "scheduler": {},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "device": [],
        },
        "data": {"cursor": 0, "pending_batch": None},
        "replay": {},
        "history": {},
        "pins": {},
        "selection": {},
    }


def test_complete_state_lineage_and_pending_microbatch_rejection(tmp_path):
    h = digest("learning")
    c = digest("campaign")
    t = digest("task")
    store = ProgressiveCheckpointStore(tmp_path, compatibility=h, campaign=c, task=t, token_cap=100)
    cuts = {
        k: digest(k)
        for k in (
            "stream",
            "allocation",
            "archive",
            "library",
            "cost_journal",
            "selection",
            "milestones",
            "consumed",
        )
    }
    s = states()
    ledger = Ledger(100)
    r = store.save(
        s,
        ledger,
        cuts=cuts,
        created_at="2026-11-08T00:00:00Z",
        purpose="acceptance",
        track="synthetic_strict",
    )
    restored = store.load(r)
    assert torch.equal(restored["states"]["weights"]["parameter"], s["weights"]["parameter"])
    assert torch.equal(restored["states"]["rng"]["torch"], s["rng"]["torch"])
    for kwargs in (
        {"compatibility": digest("changed objective"), "campaign": c, "task": t},
        {"compatibility": h, "campaign": c, "task": digest("new task")},
    ):
        with pytest.raises(ValueError):
            ProgressiveCheckpointStore(tmp_path, token_cap=100, **kwargs).load(r)
    pending = ledger.reserve("partial", 5)
    with pytest.raises(ValueError):
        store.save(
            s,
            pending,
            cuts=cuts,
            created_at=r["created_at"],
            purpose="acceptance",
            track="synthetic_strict",
        )
    assert pending.consumed == 5


@pytest.mark.parametrize(
    "point", ["weights", "optimizer", "scheduler", "latest", "fsync", "rename"]
)
def test_failed_publication_keeps_previous_complete_cut(tmp_path, monkeypatch, point):
    import oeis_learn.fresh.bigint_state as module

    store = ProgressiveCheckpointStore(
        tmp_path,
        compatibility=digest("learning"),
        campaign=digest("campaign"),
        task=digest("task"),
        token_cap=100,
    )
    cuts = {
        k: digest(k)
        for k in (
            "stream",
            "allocation",
            "archive",
            "library",
            "cost_journal",
            "selection",
            "milestones",
            "consumed",
        )
    }
    kw = {
        "cuts": cuts,
        "created_at": "2026-11-08T00:00:00Z",
        "purpose": "acceptance",
        "track": "synthetic_strict",
    }
    old = store.save(states(), Ledger(100), **kw)
    original = module.write

    def interrupted(path, *args, **kwargs):
        if (
            point == "latest" and path.name == "generalization-latest.json"
        ) or path.name == point + ".pt":
            raise InterruptedError("publication fault")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module, "write", interrupted)
    if point in {"fsync", "rename"}:
        from oeis_learn.fresh import common

        def failure(*args, **kwargs):
            raise InterruptedError("durability fault")

        monkeypatch.setattr(common.os, "fsync" if point == "fsync" else "replace", failure)
    new = states()
    new["weights"]["parameter"] = torch.tensor([2.0])
    new["optimizer"] = {"changed": 1}
    new["scheduler"] = {"changed": 1}
    with pytest.raises(InterruptedError):
        store.save(new, Ledger(100), **kw)
    assert torch.equal(
        store.load(old)["states"]["weights"]["parameter"], states()["weights"]["parameter"]
    )


def test_interrupted_optimizer_rolls_back_tensors_but_recharges_spent_work(tmp_path):
    """Independent tiny AdamW/RNG oracle around durable debit and optimizer work."""
    store = ProgressiveCheckpointStore(
        tmp_path,
        compatibility=digest("learning"),
        campaign=digest("campaign"),
        task=digest("task"),
        token_cap=100,
    )
    s = states()
    cuts = {
        k: digest(k)
        for k in (
            "stream",
            "allocation",
            "archive",
            "library",
            "cost_journal",
            "selection",
            "milestones",
            "consumed",
        )
    }
    kw = {
        "cuts": cuts,
        "created_at": "2026-11-08T00:00:00Z",
        "purpose": "acceptance",
        "track": "synthetic_strict",
    }
    old = store.save(s, Ledger(100), **kw)
    pending = Ledger(100).reserve("unfinished", 3)
    # The external debit survives a lost in-memory update; loading tensors alone
    # cannot establish a new allowance. Reconcile before retrying the same batch.
    outputs = []
    for _ in range(2):
        restored = store.load(old)["states"]
        parameter = torch.nn.Parameter(restored["weights"]["parameter"].detach().clone())
        optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.01)
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["rng"]["torch"])
        inputs = torch.rand(3)
        (parameter * inputs).sum().backward()
        optimizer.step()
        outputs.append((parameter.detach().clone(), inputs, optimizer.state_dict()))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    retry = Ledger(100, consumed=pending.consumed).reserve("retry", 3).commit("retry")
    assert retry.consumed == 6 and retry.updates == 1
    assert pending.consumed == 3 and pending.updates == 0
