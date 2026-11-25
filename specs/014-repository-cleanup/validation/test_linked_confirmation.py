"""Run the research compiler's confirmation initializer through the bounded CPU learner."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

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


def confirmation_pair(root, seed, compatibility):
    _, panels, screening = roster_inputs()
    data = seal(
        {"version": "progressive-registered-data/v1", "design": seal({"seeds": [seed, 19]})}
    )
    rows = construct(scientific_round(data, expanded=True, streamlined=True), panels, screening)[
        "tasks"
    ]
    by_id = {row["id"]: row for row in rows}
    second = by_id[f"confirmation-0-{seed}"]
    first = by_id[second["parent"]]
    assert (first["seed"], second["seed"]) == (seed, seed)
    assert (first["token_target"], second["token_target"]) == (12_500_000, 25_000_000)
    atomic_write(root / "compiler/qualification.json", {"images": [digest("cpu"), digest("gpu")]})
    compiler = Compiler(
        root,
        root / "compiler",
        qualification=root / "compiler",
        recipe=None,
        checkpoint=None,
        created_at="2026-11-08T00:00:00Z",
    )
    for row in (first, second):
        compiler.task(
            row,
            {"training": ["training-channel.json"]},
            seed=row["seed"],
            compatibility=compatibility,
            study={"identity": digest("fixture")},
            parameters=[],
            scopes={},
            maximum_update=1,
            checkpoint_bytes=1,
        )
    return compiler.plans


def test_second_confirmation_restores_first(tmp_path, monkeypatch):
    import torch

    monkeypatch.setenv("OEIS_PROGRESSIVE_ACCEPTANCE_DEVICE", "cpu")
    campaign, _, _, charge = fixture(tmp_path, interrupt=False)
    plans = campaign["payload"]["tasks"]
    parent, child = plans
    compatibility = parent["initializer"]["compatibility"]
    compiled = confirmation_pair(tmp_path, parent["seed"], compatibility)
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
    campaign = validate_campaign(seal(campaign))
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
    cid = campaign["identity"].split(":")[1]
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
        token = progressive_commands._INVOCATION.set(name)
        try:
            return progressive_roles.invoke(
                host,
                image,
                tmp_path,
                "train",
                {"training": [describe(tmp_path, "training-channel.json")]},
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

    def assert_same_state(expected, actual):
        assert type(actual) is type(expected)
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                assert_same_state(expected[key], actual[key])
        elif isinstance(expected, (tuple, list)):
            for left, right in zip(expected, actual, strict=True):
                assert_same_state(left, right)
        elif isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual, strict=True)
        else:
            assert actual == expected

    assert_same_state(before["states"], inherited["states"])
    # Pausing the child legitimately appends an elapsed-time reservation to the same ledger.
    budget = journal.path.read_bytes()
    assert budget.startswith(parent_prefix)
    new_events = [
        json.loads(line)["identity"] for line in budget[len(parent_prefix) :].splitlines()
    ]
    assert inherited["cuts"]["cost_journal"] in new_events
    assert inherited["cuts"] == dict(before["cuts"], cost_journal=inherited["cuts"]["cost_journal"])
    resumed = train(second, 1, "finish-second")
    assert resumed["mechanical_status"] == "COMPLETE", resumed
    assert resumed["metrics"]["device"] == "cpu"
    final = saved(second)["ledger"]
    assert (final.consumed, final.updates) == (4 * charge, 4)
    assert journal.task_remaining(second)["tokens"] == child["caps"]["tokens"] - 2 * charge
    assert journal.task_remaining(first) == remaining
