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


def role_authority(root, operation, values, *, costs=None, useful_us=5_000_000):
    import time
    from oeis_learn.generalization.progressive.budgets import role_work_requirements
    from oeis_learn.generalization.progressive.contracts import record, digest, seal
    from tests.unit.test_progressive_freeze import campaign

    c = campaign()
    plan = c["payload"]["tasks"][0]
    plan["finalization_us"] = 1000
    plan["startup_us"] = 1000
    c = seal(c)
    t = record(
        "task",
        dict(plan, campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    j = BudgetJournal(root / "journal", c["identity"], c["payload"]["caps"])
    j.task_remaining(t)
    cost = role_work_requirements(operation, values) if costs is None else dict(costs)
    cost["elapsed_us"] = useful_us + plan["finalization_us"]
    boot = __import__("pathlib").Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    j.reserve(plan["id"], operation, 0, cost, boot=boot, monotonic_us=time.monotonic_ns() // 1000)
    event = j._read()[0][-1]
    a = seal(
        dict(
            version="progressive-role-budget/v1",
            campaign=c,
            task=t,
            operation=operation,
            inputs=digest(sorted(values, key=lambda v: v["identity"])),
            reservation=event,
        )
    )
    return a, j, t


@pytest.mark.parametrize("operation", ["generate", "score", "prepare-stream", "admit-training"])
def test_installed_work_requires_full_prepaid_enclosing_reservation(tmp_path, operation):
    from oeis_learn.generalization.progressive.budgets import RoleWorkBudget, role_work_requirements
    from oeis_learn.generalization.progressive.generator import default_policy
    from oeis_learn.pipeline_validation.identity import seal

    kinds = {
        "generate": "visible_prompt",
        "score": "confirmation_truth",
        "prepare-stream": "generation_policy",
        "admit-training": "candidate",
    }
    p = default_policy(seed=13) | {"max_draws": 2, "max_work_units": 3}
    values = [seal(dict(kind=kinds[operation], data=p if operation == "prepare-stream" else {}))]
    cost = role_work_requirements(operation, values)
    for index, key in enumerate(k for k, v in cost.items() if v):
        inadequate = dict(cost, **{key: cost[key] - 1})
        a, _, _ = role_authority(tmp_path / str(index), operation, values, costs=inadequate)
        with pytest.raises(ValueError, match="insufficient enclosing"):
            RoleWorkBudget(a, operation, values, tmp_path / f"forbidden-{index}")
        assert not (tmp_path / f"forbidden-{index}").exists()


def test_host_refuses_zero_work_before_role_projection(tmp_path):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.progressive_roles import _invoke_reserved
    from oeis_learn.generalization.progressive.budgets import KEYS
    from oeis_learn.generalization.progressive.contracts import seal
    from tests.contract.test_progressive_roles import artifact

    values = [seal(dict(kind="candidate", data={}))]
    a, j, t = role_authority(tmp_path, "admit-training", values)
    c = a["campaign"]
    runtime = SimpleNamespace(
        execution_ready=True,
        identity=c["payload"]["runtime"],
        track=c["track"],
        purpose=c["purpose"],
        payload={"operations": ["admit-training"], "scope": {"images": ["image"]}},
    )
    item = artifact(tmp_path, "candidate.json", "candidate", "admission", {})
    cost = dict.fromkeys(KEYS, 0) | {"elapsed_us": 1_000_000}
    before = j.path.read_bytes()
    with pytest.raises(ValueError, match="insufficient enclosing"):
        _invoke_reserved(
            None,
            "image",
            tmp_path,
            "admit-training",
            {"candidates": [item], "admission_truth": []},
            output="never",
            campaign=c,
            task=t,
            journal=j,
            ordinal=1,
            reservation=cost,
            seconds=1,
            runtime=runtime,
        )
    assert j.path.read_bytes() == before and not (tmp_path / "never").exists()


def test_installed_work_counter_stops_retries_without_refunding_reservation(tmp_path):
    from oeis_learn.generalization.progressive.budgets import RoleWorkBudget
    from oeis_learn.generalization.progressive.contracts import seal

    values = [seal(dict(kind="candidate", data={}))]
    authority, journal, task = role_authority(tmp_path, "admit-training", values)
    budget = RoleWorkBudget(authority, "admit-training", values, tmp_path / "out")
    budget.consume(attempts=1, rejections=1, native_work=1)
    budget.consume(native_work=1)
    before = budget.path.read_bytes()
    with pytest.raises(TimeoutError, match="reservation exhausted"):
        budget.consume(native_work=1)
    assert budget.path.read_bytes() == before
    assert journal.task_remaining(task)["native_work"] == task["payload"]["caps"]["native_work"] - 2


def test_storage_guard_covers_shared_evidence_writers_and_copy(tmp_path):
    from oeis_learn.generalization.progressive.storage import StorageGuard, copy_file
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.pilot_evidence import Evidence

    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source"
    source.write_bytes(b"x" * 200)
    with StorageGuard([output], max_bytes=100, max_inodes=16).activate():
        with pytest.raises(TimeoutError, match="storage"):
            write(output / "index.json", dict(payload="x" * 200))
        with pytest.raises(TimeoutError, match="storage"):
            copy_file(source, output / "copied")
        evidence = Evidence(output, [dict(id="task")])
        with pytest.raises(TimeoutError, match="storage"):
            evidence("task", "correct", 0, "STARTED", dict(tokens=[]))
    assert list(output.iterdir()) == []


def test_verified_index_avoids_reparse_and_rechecks_external_edits(tmp_path, monkeypatch):
    j=BudgetJournal(tmp_path,'c',caps())
    j.reserve('task','update',0,work(tokens=3),boot='b',monotonic_us=0)
    j.commit('task/update/0')
    # An unchanged file and our own appends do not reparse its historical lines.
    with monkeypatch.context() as m:
        from pathlib import Path
        original=Path.open
        def guarded(path,mode='r',*args,**kwargs):
            if path==j.path and mode=='r':pytest.fail('reparsed journal prefix')
            return original(path,mode,*args,**kwargs)
        m.setattr(Path,'open',guarded)
        for i in range(1,4):
            j.reserve('task','update',i,work(tokens=3),boot='b',monotonic_us=i)
            j.commit('task/update/'+str(i))
        assert j.snapshot()['consumed']['tokens']==12
    # Another legitimate owner invalidates the cache and is fully reconciled.
    other=BudgetJournal(tmp_path,'c',caps())
    other.reserve('task','update',4,work(tokens=5),boot='b',monotonic_us=4)
    assert j.snapshot()['consumed']['tokens']==17
    raw=j.path.read_text()
    j.path.write_text(raw.replace('"tokens": 3','"tokens": 2',1))
    with pytest.raises(ValueError,match='chain broken'):j.snapshot()


def test_large_journal_allowance_is_frozen_and_legacy_default_stays_bounded(tmp_path):
    from oeis_learn.generalization.progressive.storage import journal_limit,JOURNAL_BYTES
    j=BudgetJournal(tmp_path/'research','c',caps(),journal_bytes=JOURNAL_BYTES)
    j.reserve('t','update',0,work(tokens=1),boot='b',monotonic_us=0)
    assert journal_limit(j.root)==2<<30
    assert BudgetJournal(j.root,'c',caps()).snapshot()['consumed']['tokens']==1
    with pytest.raises(ValueError,match='storage changed'):
        BudgetJournal(j.root,'c',caps(),journal_bytes=128<<20)
    legacy=BudgetJournal(tmp_path/'legacy','c',caps())
    assert journal_limit(legacy.root)==64<<20
    BudgetJournal(legacy.root,'c',caps(),journal_bytes=128<<20)
    assert journal_limit(legacy.root)==128<<20
