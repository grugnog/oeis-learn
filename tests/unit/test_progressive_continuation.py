"""Successor slots inherit complete learning state, never their predecessor's allowance."""

import copy

import pytest

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.campaign import validate_campaign, validate_task
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.recovery import Recovery
from tests.helpers.progressive_fixture import caps, envelope, graph, task
from tests.integration.test_progressive_recovery import states
from tests.unit.test_progressive_freeze import campaign as base_campaign


def campaign():
    c = base_campaign()
    p = c["payload"]
    g = graph()
    first = task()["payload"]
    first.pop("campaign")
    first.update(
        id="first", slot="capability", stage="capability", output="first", caps=caps(tokens=10)
    )
    second = copy.deepcopy(first)
    second.update(
        id="second",
        slot="diversity",
        stage="diversity",
        output="second",
        initializer={
            "mode": "full_state",
            "artifact": None,
            "compatibility": digest("learning"),
            "parent_slot": "capability",
        },
    )
    p["tasks"] = [first, second]
    p["graph"] = g
    c = envelope("campaign", p)
    return validate_campaign(c)


def tasks(c):
    return [envelope("task", dict(p, campaign=c["identity"])) for p in c["payload"]["tasks"]]


def test_slot_continuation_preserves_spent_exposure_and_full_cut(tmp_path):
    c = campaign()
    one, two = tasks(c)
    j = BudgetJournal(tmp_path, c["identity"], c["payload"]["caps"])
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
    with Recovery(j, one, digest("learning")) as a:
        a.initialize(states(), cuts)
        a.begin(boot="b", monotonic_us=0)
        ledger = a.reserve(["x"], 3, boot="b", monotonic_us=1)
        s = states()
        s["data"]["cursor"] = 1
        a.commit(s, ledger.commit(ledger.pending[0]), cuts)
        a.close(boot="b", monotonic_us=100, complete=True)
        with Recovery(j, two, digest("learning")) as b:
            b.initialize_from(a, campaign=c)
            restored = b.restore()
            assert restored["ledger"].consumed == 3 and restored["ledger"].updates == 1
            assert restored["ledger"].cap == 13
            assert j.task_remaining(two)["tokens"] == 10
            b.begin(boot="b", monotonic_us=200)
            retry = b.reserve(["y"], 2, boot="b", monotonic_us=201)
            s = states()
            s["data"]["cursor"] = 2
            b.commit(s, retry.commit(retry.pending[0]), cuts)
            b.close(boot="b", monotonic_us=300)
    with Recovery(j, two, digest("learning")) as b:
        assert b.restore()["ledger"].consumed == 5
        assert j.task_remaining(two)["tokens"] == 8


def test_unknown_or_nonancestor_parent_slot_rejected():
    c = campaign()
    _one, two = tasks(c)
    two["payload"]["initializer"]["parent_slot"] = "diversity"
    from oeis_learn.pipeline_validation.identity import seal

    with pytest.raises(ValueError):
        validate_task(seal(two), campaign=c)


def test_preallocated_same_slot_continuation_after_terminal_partial(tmp_path):
    c = campaign()
    p = c["payload"]
    second = p["tasks"][1]
    second["slot"] = "capability"
    second["stage"] = "capability"
    second["initializer"].pop("parent_slot")
    second["initializer"]["parent_task"] = "first"
    p["graph"]["nodes"][0]["caps"] = {k: 2 * v for k, v in p["graph"]["nodes"][0]["caps"].items()}
    c = validate_campaign(envelope("campaign", p))
    _one, _two = tasks(c)
    from oeis_learn.generalization.progressive.lifecycle import RunStore

    store = RunStore(tmp_path / "control", c)
    store.append({"tasks": {"first": "PARTIAL"}, "status": "PARTIAL", "invocation": "first"})
    assert store.next_task()["payload"]["id"] == "second"
    p["tasks"][1]["initializer"]["parent_task"] = "second"
    with pytest.raises(ValueError):
        validate_campaign(envelope("campaign", p))


def test_prepared_learning_channel_rejects_truth_alias_and_research_diagnostics(tmp_path):
    import os

    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.projections import describe, project
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    from oeis_learn.pipeline_validation.progressive_learner import validate_input
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY

    _c, tasks, _channel, _ = fixture(tmp_path, interrupt=False)
    package = load(tmp_path / "training-input.json")
    with pytest.raises(ValueError, match="diagnostic"):
        validate_input(package, dict(tasks[0], purpose="research"))
    hidden = tmp_path / "private-truth.json"
    os.link(tmp_path / "training-input.json", hidden)
    with pytest.raises(ValueError):
        project(
            tmp_path,
            tmp_path / "projection",
            "train",
            {"training": [describe(tmp_path, "training-channel.json")]},
            registry=REGISTRY,
            forbidden=[hidden],
        )


def test_retention_keeps_first_milestone_and_ignores_uncommitted_tensor_tip(tmp_path, monkeypatch):
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.generalization.progressive import milestones
    from oeis_learn.generalization.progressive.contracts import load

    # Unit-scale token thresholds; this is not sustained-duration evidence.
    monkeypatch.setattr(milestones, "MILESTONES", (3,))
    c = campaign()
    one, _ = tasks(c)
    j = BudgetJournal(tmp_path, c["identity"], c["payload"]["caps"])
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
    with Recovery(j, one, digest("learning")) as r:
        initial = r.initialize(states(), cuts)
        r.begin(boot="b", monotonic_us=0)
        r.retain(milestone_pins=1)
        committed = []
        for update, count in enumerate((3, 2, 2), 1):
            ledger = r.reserve([str(update)], count, boot="b", monotonic_us=update)
            s = states()
            s["data"]["cursor"] = update
            committed.append(r.commit(s, ledger.commit(ledger.pending[0]), cuts))
            r.retain(milestone_pins=1)
        root = r.store.root
        assert not (root / initial["payload"]["full_state"]["path"]).exists()
        assert (root / committed[0]["payload"]["full_state"]["path"]).is_file()
        assert not (root / committed[1]["payload"]["full_state"]["path"]).exists()
        assert len(list(root.glob("generalization-*/state.json"))) == 2
        s = states()
        s["data"]["cursor"] = 4
        orphan = r.save(s, Ledger(10, consumed=8, updates=4), cuts)
        assert load(root / "generalization-latest.json") == orphan["payload"]["full_state"]
        r.retain(milestone_pins=1)
        assert not (root / orphan["payload"]["full_state"]["path"]).exists()
        assert load(root / "generalization-latest.json") == committed[-1]["payload"]["full_state"]
        assert r.restore()["ledger"].consumed == 7
        with pytest.raises(ValueError, match="changed frozen"):
            r.retain(milestone_pins=2)
