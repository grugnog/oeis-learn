"""Independent expected debits for crashes, repeats and active intervals."""

import pytest
from tests.helpers.progressive_fixture import caps, task
from oeis_learn.generalization.progressive.budgets import BudgetJournal, TaskLease


def work(**kw):
    return (
        caps(
            tokens=0,
            draws=0,
            rejections=0,
            attempts=0,
            native_work=0,
            elapsed_us=0,
            disk_bytes=0,
            archive_bytes=0,
        )
        | kw
    )


def test_crash_charges_pending_tokens_and_unknown_elapsed(tmp_path):
    journal = BudgetJournal(tmp_path, "campaign", caps())
    journal.reserve(
        "t",
        "update",
        0,
        work(tokens=12, draws=3, rejections=2, elapsed_us=20_000_000),
        boot="b",
        monotonic_us=10,
    )
    resumed = BudgetJournal(tmp_path, "campaign", caps())
    assert resumed.reconcile()["consumed"]["tokens"] == 12
    assert resumed.snapshot()["consumed"]["elapsed_us"] == 20_000_000
    resumed.commit("t/update/0")  # no termination proof: consumes full reservation
    resumed.reserve("t", "update", 1, work(tokens=12), boot="new-boot", monotonic_us=1)
    assert resumed.snapshot()["consumed"]["tokens"] == 24
    assert resumed.reserve(
        "t",
        "update",
        0,
        work(tokens=12, draws=3, rejections=2, elapsed_us=20_000_000),
        boot="b",
        monotonic_us=10,
    )["committed"]
    assert resumed.snapshot()["consumed"]["tokens"] == 24
    with pytest.raises(ValueError):
        resumed.reserve("t", "update", 0, work(tokens=1), boot="b", monotonic_us=10)


def test_clean_pause_excludes_downtime_and_reboot_cannot_infer_refund(tmp_path):
    j = BudgetJournal(tmp_path, "c", caps())
    j.reserve("t", "train", 0, work(elapsed_us=1000), boot="b", monotonic_us=20)
    with pytest.raises(ValueError):
        j.commit("t/train/0", boot="reboot", monotonic_us=30, termination_proven=True)
    j.commit("t/train/0", boot="b", monotonic_us=120, termination_proven=True)
    j.reserve("t", "train", 1, work(elapsed_us=1000), boot="b", monotonic_us=9000)
    j.commit("t/train/1", boot="b", monotonic_us=9200, termination_proven=True)
    assert j.snapshot()["consumed"]["elapsed_us"] == 300


def test_no_new_allowance_on_reopen_and_update_overshoot(tmp_path):
    j = BudgetJournal(tmp_path, "c", caps(tokens=20))
    j.reserve("t", "update", 0, work(tokens=12), boot="b", monotonic_us=0)
    with pytest.raises(ValueError):
        j.reserve("t", "update", 1, work(tokens=12), boot="b", monotonic_us=1)
    with pytest.raises(ValueError):
        BudgetJournal(tmp_path, "c", caps(tokens=21))
    with TaskLease(tmp_path, task()):
        with pytest.raises(ValueError):
            with TaskLease(tmp_path, task(output="sibling")):
                pass


def test_unproven_supervisor_interval_exhausts_remaining_task_clock(tmp_path):
    from oeis_learn.pipeline_validation.identity import digest

    t = task(campaign=digest("c"))
    journal = BudgetJournal(tmp_path, digest("c"), caps())
    journal.task_remaining(t)
    journal.reserve(
        t["payload"]["id"], "role", 0, work(elapsed_us=20_000_000), boot="old", monotonic_us=5
    )
    journal.reconcile(t)
    assert journal.task_remaining(t)["elapsed_us"] == 0
    assert journal.snapshot()["consumed"]["elapsed_us"] == 100_000_000
    journal.reconcile(t)
    assert journal.snapshot()["consumed"]["elapsed_us"] == 100_000_000


def test_budget_state_rejects_symlink_aliases(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "alias").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError):
        BudgetJournal(tmp_path / "alias", "c", caps())
    state = tmp_path / "state"
    state.mkdir()
    (state / "budget.lock").symlink_to(other / "unrelated")
    with pytest.raises(ValueError):
        BudgetJournal(state, "c", caps())
