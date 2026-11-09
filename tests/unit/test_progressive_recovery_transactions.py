import pytest

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.recovery import Recovery
from tests.helpers.progressive_fixture import caps, task
from tests.integration.test_progressive_recovery import states


def setup(tmp_path):
    t = task(caps=caps(tokens=100, elapsed_us=10000000))
    j = BudgetJournal(tmp_path, t["payload"]["campaign"], caps(tokens=100, elapsed_us=10000000))
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
    return t, j, cuts


def test_newer_uncommitted_checkpoint_is_ignored_but_charges_survive(tmp_path, monkeypatch):
    t, j, cuts = setup(tmp_path)
    with Recovery(j, t, digest("learning")) as r:
        first = r.initialize(states(), cuts)
        r.begin(boot="one", monotonic_us=0)
        ledger = r.reserve(["row-a"], 3, boot="one", monotonic_us=1)
        ledger = ledger.commit(ledger.pending[0])
        changed = states()
        changed["data"]["cursor"] = 1
        original = r.append

        def fail(**kwargs):
            if "checkpoint" in kwargs:
                raise InterruptedError("after tensor publication")
            return original(**kwargs)

        monkeypatch.setattr(r, "append", fail)
        with pytest.raises(InterruptedError):
            r.commit(changed, ledger, dict(cuts, library=digest("newer")))
        assert r.restore()["cuts"]["library"] == cuts["library"]
        assert r.restore()["ledger"].consumed == 3 and r.restore()["ledger"].updates == 0
        r.close(boot="one", monotonic_us=100)
    with Recovery(j, t, digest("learning")) as r:
        restored = r.begin(boot="two", monotonic_us=500)
        assert restored["discarded_tokens"] == 3
        assert r.state["checkpoint"] == first
        with pytest.raises(ValueError):
            r.reserve(["different"], 3, boot="two", monotonic_us=501)
        retry = r.reserve(["row-a"], 3, boot="two", monotonic_us=501)
        assert retry.consumed == 6
        s = states()
        s["data"]["cursor"] = 1
        r.commit(s, retry.commit(retry.pending[0]), cuts)
        assert r.restore()["discarded_tokens"] == 3
        r.close(boot="two", monotonic_us=600)
    assert j.task_remaining(t)["elapsed_us"] == 9999800


def test_unproven_reboot_exhausts_time_and_changed_output_cannot_reset(tmp_path):
    from oeis_learn.pipeline_validation.identity import seal

    t, j, cuts = setup(tmp_path)
    with Recovery(j, t, digest("learning")) as r:
        r.initialize(states(), cuts)
        r.begin(boot="old", monotonic_us=0)
    with Recovery(j, t, digest("learning")) as r:
        with pytest.raises(ValueError, match="exhausted"):
            r.begin(boot="new", monotonic_us=0)
        assert r.state["status"] == "PARTIAL"
        with pytest.raises(ValueError, match="terminal"):
            r.begin(boot="new", monotonic_us=1)
    t["payload"]["output"] = "different"
    with pytest.raises(ValueError):
        Recovery(j, seal(t), digest("learning"))


def test_exclusive_lease_and_protected_finalization(tmp_path):
    t, j, cuts = setup(tmp_path)
    with Recovery(j, t, digest("learning")) as r:
        r.initialize(states(), cuts)
        r.begin(boot="one", monotonic_us=0)
        with pytest.raises(ValueError, match="lease"), Recovery(j, t, digest("learning")):
            pass
        with pytest.raises(ValueError, match="finalization"):
            r.reserve(["x"], 3, boot="one", monotonic_us=9999999)
        r.close(boot="one", monotonic_us=100, complete=True)
        with pytest.raises(ValueError, match="terminal"):
            r.begin(boot="one", monotonic_us=101)


def test_orphan_debit_with_clean_supervisor_proof_does_not_consume_paused_time(
    tmp_path, monkeypatch
):
    t, j, cuts = setup(tmp_path)
    with Recovery(j, t, digest("learning")) as r:
        r.initialize(states(), cuts)
        r.begin(boot="boot", monotonic_us=0)
        commit = j.commit

        def fault(key, **kwargs):
            if "/update/" in key:
                raise InterruptedError("after durable debit")
            return commit(key, **kwargs)

        with monkeypatch.context() as m:
            m.setattr(j, "commit", fault)
            with pytest.raises(InterruptedError):
                r.reserve(["x"], 3, boot="boot", monotonic_us=1)
        r.close(boot="boot", monotonic_us=100)
    with Recovery(j, t, digest("learning")) as r:
        state = r.begin(boot="reboot", monotonic_us=100000000)
        assert state["ledger"].consumed == 3 and state["discarded_tokens"] == 3
        assert r.clock["end"] - r.clock["start"] == 9999900
        r.close(boot="reboot", monotonic_us=100000100)


def test_clock_can_cover_fresh_model_startup_and_failed_startup_cannot_reset(tmp_path):
    t, j, cuts = setup(tmp_path)
    with Recovery(j, t, digest("learning")) as r:
        assert r.begin(boot="boot", monotonic_us=0) is None
        r.initialize(states(), cuts)
        assert r.state["status"] == "RUNNING"
        r.close(boot="boot", monotonic_us=200)
    assert j.task_remaining(t)["elapsed_us"] == 9999800
    other = tmp_path / "other"
    t, j, cuts = setup(other)
    with Recovery(j, t, digest("learning")) as r:
        r.begin(boot="boot", monotonic_us=0)
        # Power loss before initializer publication: no end proof, no free retry.
    with Recovery(j, t, digest("learning")) as r, pytest.raises(ValueError, match="exhausted"):
        r.begin(boot="reboot", monotonic_us=0)


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "tokens",
        "current_train_tokens",
        "pending",
        "prior_train",
        "wrong_stage",
        "wrong_track",
    ],
)
def test_first_oeis_initialization_retains_only_completed_preparation_charges(tmp_path, change):
    from oeis_learn.generalization.progressive.contracts import seal

    t, j, cuts = setup(tmp_path)
    t = seal(dict(t, track="oeis_derived", payload=dict(t["payload"], stage="oeis")))
    if change == "wrong_stage":
        t = seal(dict(t, payload=dict(t["payload"], stage="components")))
    if change == "wrong_track":
        t = seal(dict(t, track="synthetic_strict"))
    cost = dict.fromkeys(t["payload"]["caps"], 0)
    cost.update(draws=2, attempts=2, native_work=3, elapsed_us=100)
    if change == "tokens":
        cost["tokens"] = 1
    prep = j.reserve(
        t["payload"]["id"],
        "train" if change == "prior_train" else "generate",
        0,
        cost,
        boot="boot",
        monotonic_us=0,
    )
    if change != "pending":
        j.commit(prep["key"])
    train_cost = dict.fromkeys(cost, 0)
    train_cost["elapsed_us"] = 5000000
    if change == "current_train_tokens":
        train_cost["tokens"] = 1
    train = j.reserve(t["payload"]["id"], "train", 1, train_cost, boot="boot", monotonic_us=100)
    with Recovery(j, t, digest("learning")) as owner:
        owner.attach(dict(reservation_key=train["key"], boot="boot", start_us=100, end_us=5000100))
        before = j.snapshot()
        if change != "none":
            with pytest.raises(ValueError, match="spent task"):
                owner.initialize(states(), cuts)
            assert owner.state["checkpoint"] is None
        else:
            owner.initialize(states(), cuts)
            assert owner.restore()["ledger"].consumed == 0
            assert owner.restore()["ledger"].updates == 0
            with pytest.raises(ValueError, match="existing task"):
                owner.initialize(states(), cuts)
        assert j.snapshot() == before
        assert j.task_remaining(t)["attempts"] == t["payload"]["caps"]["attempts"] - 2
        assert j.task_remaining(t)["native_work"] == t["payload"]["caps"]["native_work"] - 3
