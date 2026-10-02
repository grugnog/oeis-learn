"""G4 actual stochastic SFT, complete state, rollback, crash publication."""

from copy import deepcopy
import shutil
import pytest
import torch
from tests.helpers.foundation_training import prepare, trainer
from tests.unit.test_foundation_config_metrics import Clock
from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
from oeis_learn.tracking.budget_ledger import BudgetLedger
from oeis_learn.experiments.artifacts import compute_file_hash


def assert_same(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


def test_full_interrupted_continuation(training_inputs, tmp_path):
    whole, split = tmp_path / "whole", tmp_path / "split"
    run = prepare(whole, training_inputs)
    shutil.copytree(whole, split)
    states, orders, spent = [], [], []
    for root, interrupt in ((whole, False), (split, True)):
        clock = Clock()
        ledger = BudgetLedger(root, run["run_id"], run["budget_ns"], clock=clock)
        learner = trainer(root)
        order = []
        for i in range(3):
            ledger.transition("training")
            # Discarded prefetch must not consume order or dropout RNG.
            prefetch = learner.peek()
            assert learner.peek() == prefetch
            result = learner.update()
            order.append(result["sample_ids"])
            clock.advance(10)
            ledger.transition("checkpoint")
            path = learner.save(ledger.snapshot(), final=i == 2)
            clock.advance(2)
            ledger.transition("paused")
            if interrupt and i == 0:
                expected = deepcopy(learner.payload())
                clock.advance(123)  # clean pause is not charged
                ledger = BudgetLedger(root, run["run_id"], run["budget_ns"], clock=clock)
                learner = trainer(root, resume=True)
                ledger.check_checkpoint(load_foundation_checkpoint(path).manifest)
                assert_same(expected, learner.payload())
        states.append(deepcopy(learner.payload()))
        loaded = load_foundation_checkpoint(path, include_payload=True)
        assert len(loaded.payload) == 11
        counters = loaded.payload["counters"]
        assert counters["completed_update"] == 3
        assert counters["charged_budget_ns"] == 34
        assert counters["generation"] == 2
        assert all(loaded.manifest[key] == value for key, value in counters.items())
        orders.append(order)
        spent.append(ledger.state["charged_budget_ns"])
    assert_same(*states)
    assert orders[0] == orders[1]
    assert spent == [36, 36]


def test_corrupt_newest_rolls_back_without_refund(training_inputs, tmp_path):
    root = tmp_path / "run"
    run = prepare(root, training_inputs)
    clock = Clock()
    ledger = BudgetLedger(root, run["run_id"], run["budget_ns"], clock=clock)
    learner = trainer(root)
    ledger.transition("training")
    learner.update()
    clock.advance(10)
    first = learner.save(ledger.heartbeat())
    expected = deepcopy(learner.payload())
    learner.update()
    clock.advance(10)
    second = learner.save(ledger.heartbeat())
    second.with_suffix(".pt").write_bytes(b"corrupt")
    clock.advance(9)
    ledger = BudgetLedger(root, run["run_id"], run["budget_ns"], clock=clock)
    resumed = trainer(root, resume=True)
    assert resumed.completed_update == 1
    assert_same(expected, resumed.payload())
    assert ledger.state["charged_budget_ns"] == 29
    assert ledger.state["estimated_ns"] == 9
    assert list((root / "checkpoints/quarantine").rglob("*.pt"))
    assert resumed.previous == compute_file_hash(first)
    third = resumed.save(ledger.snapshot())
    assert third.name == "checkpoint-00000002.json"


@pytest.mark.parametrize("stage", ["blob_flushed", "blob_published", "manifest_published"])
def test_atomic_publication_interruptions(training_inputs, tmp_path, stage):
    root = tmp_path / "run"
    run = prepare(root, training_inputs)
    ledger = BudgetLedger(root, run["run_id"], run["budget_ns"])
    learner = trainer(root)
    learner.update()
    learner.save(ledger.snapshot())
    learner.update()

    def fail(point):
        if point == stage:
            raise RuntimeError("injected interruption")

    with pytest.raises(RuntimeError, match="injected"):
        learner.save(ledger.snapshot(), hook=fail)
    resumed = trainer(root, resume=True)
    assert resumed.completed_update == (2 if stage == "manifest_published" else 1)


def test_retention_preserves_two_plus_pinned_final(training_inputs, tmp_path):
    root = tmp_path / "run"
    run = prepare(root, training_inputs)
    ledger = BudgetLedger(root, run["run_id"], run["budget_ns"])
    learner = trainer(root)
    learner.update()
    pinned = learner.save(ledger.snapshot(), final=True)
    for _ in range(3):
        learner.save(ledger.snapshot())
    assert len(learner.store.manifests()) == 3
    assert pinned.exists()


def test_reject_missing_pool_and_resume_runtime_change(training_inputs, tmp_path):
    from oeis_learn.experiments.artifacts import load_json
    from oeis_learn.tracking.foundation_metrics import replace_json

    root = tmp_path / "run"
    run = prepare(root, training_inputs)
    learner = trainer(root)
    learner.update()
    learner.save(dict(run_id=run["run_id"], sequence=1, charged_budget_ns=1))
    runtime_path = root / "checkpoints/runtime.json"
    runtime = load_json(runtime_path.read_bytes())
    runtime["torch"] = "other"
    replace_json(runtime_path, runtime)
    with pytest.raises(ValueError, match="runtime"):
        trainer(root, resume=True)
    (root / "trainer/examples.json").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        trainer(root)


def test_frozen_run_limits_cannot_change_on_resume(training_inputs, tmp_path):
    from oeis_learn.experiments.artifacts import load_json
    from oeis_learn.tracking.foundation_metrics import replace_json

    root = tmp_path / "run"
    prepare(root, training_inputs)
    path = root / "run.json"
    altered = load_json(path.read_bytes())
    altered["free_disk_floor_bytes"] = 0
    replace_json(path, altered)
    with pytest.raises(ValueError, match="runtime"):
        trainer(root)


pytest_plugins = ["tests.helpers.foundation_training"]
