import pytest

from oeis_learn.generalization.progressive.archive import Archive
from oeis_learn.generalization.progressive.contracts import seal
from tests.unit.test_progressive_archive import admitted


def fixture(tmp_path):
    from oeis_learn.generalization.progressive.replay import freeze_replay

    archive = Archive(tmp_path)
    cut = archive.publish([admitted("a", "first"), admitted("b", "second")])
    return freeze_replay(archive, cut)


def test_whole_example_token_mixture_is_deterministic_and_task_balanced(tmp_path):
    from oeis_learn.generalization.progressive.replay import Mixer

    package = fixture(tmp_path)
    mixer = Mixer(package)
    state = mixer.initial()
    seen = set()
    synthetic = [
        dict(draw_identity="synthetic", values=["1"] * 20, tokens=[1] * 17, membership="training")
    ]
    for i in range(100):
        rows, charge = mixer.plan(synthetic, state, count=1)
        assert mixer.plan(synthetic, state, count=1) == (rows, charge)
        seen.update(r["task"] for r in rows if "task" in r)
        event = dict(
            index=i,
            event="RESERVE",
            operation="task/update/" + str(i),
            cost=dict(tokens=sum(len(r["tokens"]) for r in rows)),
            details=dict(replay=charge),
        )
        state = mixer.reconcile(state, [event], task="task")
    total = state["oeis_tokens"] + state["synthetic_tokens"]
    assert abs(4 * state["oeis_tokens"] - total) <= 4 * 17
    assert seen == {"a", "b"}
    assert mixer.reconcile(state, [event], task="task") == state


def test_discarded_retry_charges_are_not_erased(tmp_path):
    from oeis_learn.generalization.progressive.replay import Mixer

    mixer = Mixer(fixture(tmp_path))
    state = mixer.initial()
    rows, charge = mixer.plan([dict(tokens=[1] * 9, membership="training")], state, count=4)
    events = [
        dict(
            index=i,
            event="RESERVE",
            operation=f"task/update/{i}",
            cost=dict(tokens=charge["oeis_tokens"] + charge["synthetic_tokens"]),
            details=dict(replay=charge),
        )
        for i in range(2)
    ]
    restored = mixer.reconcile(state, events, task="task")
    assert restored["oeis_tokens"] == 2 * charge["oeis_tokens"]
    assert restored["synthetic_tokens"] == 2 * charge["synthetic_tokens"]
    assert restored["oeis_examples"] == 2 * charge["oeis_examples"]


def test_control_never_receives_discoveries_and_changed_archive_fails(tmp_path):
    from oeis_learn.generalization.progressive.replay import Mixer, freeze_replay

    archive = Archive(tmp_path)
    cut = archive.publish([admitted("a", "first")])
    control = freeze_replay(archive, cut, arm="continued_synthetic")
    assert control["archive"] is None and control["admissions"] == []
    mixer = Mixer(control)
    rows, charge = mixer.plan(
        [dict(tokens=[1] * 4, membership="training")], mixer.initial(), count=8
    )
    assert charge["oeis_tokens"] == 0 and len(rows) == 8
    bad = fixture(tmp_path / "other")
    bad["admissions"][0]["payload"]["membership"] = "assessment"
    with pytest.raises(ValueError):
        Mixer(seal(bad))
    with pytest.raises(ValueError):
        Mixer(fixture(tmp_path / "third")).plan(rows, mixer.initial(), count=1)


def test_original_cost_journal_retains_mixture_debits_and_rejects_changed_retry(tmp_path):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.generalization.progressive.replay import Mixer
    from tests.helpers.progressive_fixture import caps

    mixer = Mixer(fixture(tmp_path / "archive"))
    state = mixer.initial()
    _, charge = mixer.plan([dict(tokens=[1] * 9, membership="training")], state, count=4)
    n = charge["oeis_tokens"] + charge["synthetic_tokens"]
    journal = BudgetJournal(tmp_path / "costs", digest("campaign"), caps(tokens=10 * n))
    cost = dict.fromkeys(caps(), 0) | dict(tokens=n)
    event = journal.reserve(
        "task", "update", 0, cost, boot="boot", monotonic_us=0, details=dict(replay=charge)
    )
    journal.commit(event["key"])
    with journal.lock():
        events = journal._read()[0]
    result = mixer.reconcile(state, events, task="task")
    assert result["oeis_tokens"] == charge["oeis_tokens"]
    assert result["synthetic_tokens"] == charge["synthetic_tokens"]
    with pytest.raises(ValueError, match="conflicting"):
        journal.reserve(
            "task",
            "update",
            0,
            cost,
            boot="boot",
            monotonic_us=0,
            details=dict(replay=charge | dict(oeis_tokens=1, synthetic_tokens=n - 1)),
        )
    before = journal.path.read_bytes()
    with pytest.raises(ValueError, match="charge"):
        journal.reserve(
            "task",
            "update",
            1,
            cost,
            boot="boot",
            monotonic_us=0,
            details=dict(replay=charge | dict(oeis_tokens=n + 1)),
        )
    assert journal.path.read_bytes() == before
