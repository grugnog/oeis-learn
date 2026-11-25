"""Run the research compiler's confirmation initializer through the bounded CPU learner."""

import json
from types import SimpleNamespace

import pytest
from screening_restore import compare_state, inspect_copy

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.campaign import validate_campaign
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, record, seal
from oeis_learn.generalization.progressive.projections import describe
from oeis_learn.generalization.progressive.recovery import Recovery
from oeis_learn.generalization.progressive.research_roster import construct
from oeis_learn.pipeline_validation import progressive_commands, progressive_roles
from oeis_learn.pipeline_validation.progressive_acceptance import fixture
from oeis_learn.pipeline_validation.progressive_research_bindings import Compiler
from oeis_learn.pipeline_validation.progressive_research_design import scientific_round
from tests.helpers.progressive_engine import LocalProcessEngine
from tests.unit.test_progressive_research_roster import fixture as roster_inputs


def research_rows(seed):
    _, panels, screening = roster_inputs()
    data = seal(
        {"version": "progressive-registered-data/v1", "design": seal({"seeds": [seed, 19]})}
    )
    rows = construct(scientific_round(data, expanded=True, streamlined=True), panels, screening)[
        "tasks"
    ]
    return {row["id"]: row for row in rows}


def compile_pair(
    root,
    rows,
    seed,
    compatibility,
    study,
    parameters=(),
    scopes=None,
    channel="training-channel.json",
):
    atomic_write(root / "compiler/qualification.json", {"images": [digest("cpu"), digest("gpu")]})
    compiler = Compiler(
        root,
        root / "compiler",
        qualification=root / "compiler",
        recipe=None,
        checkpoint=None,
        created_at="2026-11-08T00:00:00Z",
    )
    for row in rows:
        compiler.task(
            row,
            {"training": [channel]},
            seed=seed,
            compatibility=compatibility,
            study={"identity": study},
            parameters=parameters,
            scopes=scopes,
            maximum_update=1,
            checkpoint_bytes=1,
        )
    return compiler.plans


def test_second_confirmation_restores_first(tmp_path, monkeypatch):
    monkeypatch.setenv("OEIS_PROGRESSIVE_ACCEPTANCE_DEVICE", "cpu")
    campaign, _, _, charge = fixture(tmp_path, interrupt=False)
    plans = campaign["payload"]["tasks"]
    parent, child = plans
    compatibility = parent["initializer"]["compatibility"]
    seed = parent["seed"]
    rows = research_rows(seed)
    second = rows[f"confirmation-0-{seed}"]
    first = rows[second["parent"]]
    assert (first["seed"], second["seed"]) == (seed, seed)
    assert (first["token_target"], second["token_target"]) == (12_500_000, 25_000_000)
    compiled = compile_pair(tmp_path, (first, second), seed, compatibility, digest("fixture"))
    # Keep the research parent identity and initializer while charging two small updates per half.
    for i, (plan, original) in enumerate(zip(plans, compiled, strict=True), 1):
        plan.update(
            id=original["id"],
            seed=original["seed"],
            initializer=original["initializer"],
            token_target=2 * i * charge,
        )
    assert child["initializer"]["mode"] == "full_state"
    assert child["initializer"]["parent_task"] == parent["id"]
    bad_child = dict(child, initializer=dict(child["initializer"], mode="weights_only"))
    with pytest.raises(ValueError, match="compatible task"):
        validate_campaign(
            seal(dict(campaign, payload=dict(campaign["payload"], tasks=[parent, bad_child])))
        )
    exercise_pair(tmp_path, campaign, charge)


def exercise_pair(tmp_path, campaign, charge, *, proposal=None, channel="training-channel.json"):
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.pipeline_validation.progressive_study import project_variant

    campaign = validate_campaign(seal(campaign))
    cid = campaign["identity"].split(":")[1]
    store = RunStore(tmp_path / "run-state/discovery-control" / cid, campaign)
    plans = campaign["payload"]["tasks"]
    if proposal is not None:
        with store.lease():
            store.choose_trial(proposal)
        store.read()
        plans = [store.resolved_plan(p) for p in plans]
        for plan in plans:
            assert plan["trial_choice"] == 1
            assert plan["trial_choices"][1]["parameters"] == proposal["parameters"]
            assert (
                load(tmp_path / plan["trial_runtime"]["path"])["identity"]
                == campaign["payload"]["runtime"]
            )
    parent, child = plans
    compatibility = parent["initializer"]["compatibility"]
    tasks = [
        record(
            "task",
            dict(plan, campaign=campaign["identity"]),
            created_at=campaign["created_at"],
            track=campaign["track"],
            purpose=campaign["purpose"],
        )
        for plan in plans
    ]
    journal = BudgetJournal(
        tmp_path / "run-state/discovery-accounting" / cid,
        campaign["identity"],
        campaign["payload"]["caps"],
        journal_bytes=campaign["payload"]["storage"]["journal_bytes"],
    )
    host = SimpleNamespace(
        engine=LocalProcessEngine(tmp_path), policy={"cpus": 1, "memory_bytes": 2 << 30}
    )
    image = load(tmp_path / "request-0.json")["image"]
    runtime = SimpleNamespace(
        execution_ready=True,
        identity=campaign["payload"]["runtime"],
        track=campaign["track"],
        purpose=campaign["purpose"],
        payload={"operations": ["train"], "scope": {"images": [image]}},
        producers=(),
    )

    def train(task, ordinal, name):
        seconds = min(60, journal.task_remaining(task)["elapsed_us"] // 1_000_000)
        reservation = dict.fromkeys(task["payload"]["caps"], 0)
        reservation["elapsed_us"] = seconds * 1_000_000
        artifacts, derived = project_variant(
            tmp_path,
            store,
            task,
            "train",
            {"training": [describe(tmp_path, channel)]},
            None,
            tmp_path / (name + "-projection"),
        )
        token = progressive_commands._INVOCATION.set(name)
        try:
            return progressive_roles.invoke(
                host,
                image,
                tmp_path,
                "train",
                artifacts,
                derived=derived,
                output=name,
                campaign=campaign,
                task=task,
                journal=journal,
                ordinal=ordinal,
                reservation=reservation,
                seconds=seconds,
                runtime=runtime,
            )
        finally:
            progressive_commands._INVOCATION.reset(token)

    def saved(task):
        state_root = (
            tmp_path / "run-state/discovery-learning" / cid / task["identity"].split(":")[1]
        )
        with Recovery(journal, task, compatibility, state_root=state_root) as recovery:
            return recovery.restore()

    first, second = tasks
    completed = train(first, 0, "first-segment")
    assert completed["mechanical_status"] == "COMPLETE", completed
    assert completed["metrics"]["device"] == "cpu"
    if proposal is not None:
        from oeis_learn.pipeline_validation.progressive_evidence import finalize_learning
        from oeis_learn.pipeline_validation.progressive_outputs import commit

        role = tmp_path / "first-segment/role"
        state_root = tmp_path / "run-state/discovery-learning" / cid / first["identity"][7:]
        final_receipt = finalize_learning(journal, first, state_root, role)

        with store.lease():
            commit(tmp_path, store, first, tmp_path / "first-segment/role", "train")
            store.boundary(first, "COMPLETE", "first-segment")
        targets = (parent["token_target"], child["token_target"])
        with pytest.raises(ValueError, match="parameters or seed"):
            inspect_copy(
                tmp_path,
                tmp_path / "wrong-choice",
                campaign,
                seal(dict(proposal, seed=proposal["seed"] + 1)),
                targets=targets,
            )
        observation = inspect_copy(
            tmp_path, tmp_path / "offline-copy", campaign, proposal, targets=targets
        )
        atomic_write(tmp_path / "screening-restore-observation.json", observation)
        assert (observation["tokens"], observation["updates"]) == (2 * charge, 2)
        assert (observation["child_tokens"], observation["child_updates"]) == (0, 0)
        assert observation["final_accounting"] == final_receipt["identity"]
        assert observation["final_accounting_journal"] == final_receipt["journal"]
        assert observation["prior_elapsed_us"] == final_receipt["milestone"]["total_cost_us"] > 0
        assert observation["device"] == "cpu"

    before = saved(first)
    remaining = journal.task_remaining(first)
    parent_prefix = journal.path.read_bytes()
    # An owned pause exposes the second worker's restored cut before its first update.
    pause = (
        tmp_path
        / "run-state/discovery-control"
        / cid
        / ("pause-" + digest("inspect-cut")[7:] + ".json")
    )
    atomic_write(pause, seal({"campaign": campaign["identity"], "requested": True}))
    paused = train(second, 0, "inspect-cut")
    assert paused["metrics"]["task_status"] == "PAUSED", paused
    assert paused["metrics"]["device"] == "cpu"
    inherited = saved(second)
    assert before["ledger"].consumed == inherited["ledger"].consumed == 2 * charge
    assert before["ledger"].updates == inherited["ledger"].updates == 2
    assert journal.task_remaining(second)["tokens"] == child["caps"]["tokens"]

    compare_state(before["states"], inherited["states"])
    # Pausing the child legitimately appends an elapsed-time reservation to the same ledger.
    budget = journal.path.read_bytes()
    assert budget.startswith(parent_prefix)
    new_events = [
        json.loads(line)["identity"] for line in budget[len(parent_prefix) :].splitlines()
    ]
    assert inherited["cuts"]["cost_journal"] in new_events
    assert inherited["cuts"] == dict(before["cuts"], cost_journal=inherited["cuts"]["cost_journal"])
    if proposal is not None:
        assert journal.task_remaining(first) == remaining
        return

    resumed = train(second, 1, "finish-second")
    assert resumed["mechanical_status"] == "COMPLETE", resumed
    assert resumed["metrics"]["device"] == "cpu"
    final = saved(second)["ledger"]
    assert (final.consumed, final.updates) == (4 * charge, 4)
    assert journal.task_remaining(second)["tokens"] == child["caps"]["tokens"] - 2 * charge
    assert journal.task_remaining(first) == remaining
