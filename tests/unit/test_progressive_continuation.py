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
