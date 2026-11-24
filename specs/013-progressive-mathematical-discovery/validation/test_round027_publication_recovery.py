"""Host-adapter regression tests. Run with pytest from the repository root."""
import copy
import importlib.util
from pathlib import Path

import pytest

from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.contracts import atomic_write, load, record, reference, seal
from oeis_learn.generalization.progressive.decisions import publish, verify_gate_producers
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.projections import describe, project
from oeis_learn.pipeline_validation import progressive_role
from tests.unit.test_progressive_dag_decisions import configured

spec = importlib.util.spec_from_file_location("publication_recovery", Path(__file__).with_name("continue-round027-publication.py"))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_elapsed_only_amendment(tmp_path):
    journal = BudgetJournal(tmp_path, "campaign", dict.fromkeys(KEYS, 10**9))
    costs = dict.fromkeys(KEYS, 0)
    costs.update(native_work=10**6, elapsed_us=20_000_000)
    original = journal.reserve(recovery.NAME, "publish", 95, costs, boot="test", monotonic_us=0)
    journal.commit(original["key"], boot="test", monotonic_us=12_818_869, termination_proven=True)
    replay = dict(ledger=reference(tmp_path, journal.path), budget=journal.snapshot())
    tail, costs = recovery.charge_state(journal, replay)
    assert not tail and costs["native_work"] == 0
    journal.reserve(recovery.NAME, "publish", recovery.ORDINAL, costs, boot="test", monotonic_us=30_000_000)
    assert len(recovery.charge_state(journal, replay)[0]) == 1
    journal.commit(recovery.KEY, boot="test", monotonic_us=35_000_000, termination_proven=True)
    assert len(recovery.charge_state(journal, replay)[0]) == 2
    spent = journal.snapshot()["consumed"]
    assert (spent["native_work"], spent["elapsed_us"]) == (1_000_000, 17_818_869)
    wrong = copy.deepcopy(replay)
    wrong["ledger"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="prefix changed"):
        recovery.charge_state(journal, wrong)
    journal.reserve("other", "publish", 0, costs, boot="test", monotonic_us=0)
    with pytest.raises(ValueError, match="already run"):
        recovery.charge_state(journal, replay)


def test_interrupted_formal_publication(tmp_path, monkeypatch):
    fx, old_bundle, policy, remaining = configured(tmp_path)
    campaign = copy.deepcopy(fx.campaign)
    gate = load(tmp_path/"gate.json")
    rows = old_bundle["rows"]
    for producer in gate["producers"]:
        if producer["arm"] != "candidate" or producer["seed"] != 1:
            continue
        scores = load(tmp_path / producer["scores"]["path"])
        for row in rows:
            if row["arm"] == "candidate" and row["seed"] == 1 and row["outcome"] == "success":
                row["complete"] = False
                scores["outcomes"][row["task"]]["equal_budget"]["complete"] = False
        atomic_write(tmp_path/"partial-scores.json", seal(scores))
        producer["scores"] = reference(tmp_path, "partial-scores.json")
    atomic_write(tmp_path/"partial-gate.json", seal(gate))
    next(r for r in campaign["payload"]["references"] if r["name"] == "gate")["artifact"] = reference(tmp_path, "partial-gate.json")
    plans = campaign["payload"]["tasks"]
    plan = dict(plans[0], id=recovery.NAME, output="compare", kind="evaluation", token_target=0)
    plans.insert(-1, plan)
    plans[-1]["id"] = "trial-00-propose"
    graph = campaign["payload"]["graph"]
    graph["nodes"][0]["caps"] = {k: 3*v for k,v in plan["caps"].items()}
    graph["edges"][0]["condition"] = "COMPLETE"
    campaign = seal(campaign)
    store = RunStore(tmp_path/"control", campaign)
    task = record("task", dict(plan, campaign=campaign["identity"]), created_at=campaign["created_at"],
                  purpose=campaign["purpose"], track=campaign["track"])
    original = dict(store.state, tasks={"parent-1":"COMPLETE", "parent-2":"COMPLETE", recovery.NAME:"FAILED"}, status="FAILED")
    store.append(original)
    assert store.next_task() is None
    journal = BudgetJournal(tmp_path/"account", campaign["identity"], campaign["payload"]["caps"])
    costs = dict.fromkeys(KEYS, 0)
    costs["elapsed_us"] = 180
    journal.reserve(recovery.NAME, "publish", 95, costs, boot="test", monotonic_us=0)
    journal.commit(recovery.NAME + "/publish/95")
    before = reference(tmp_path, journal.path)
    journal.reserve(recovery.NAME, "publish", recovery.ORDINAL, costs, boot="test", monotonic_us=0)
    journal.commit(recovery.KEY)
    data = {k:old_bundle[k] for k in ("source_slot","target_slot","rows","roster","seeds","design")}
    kwargs = dict(target_id="gain", action="RETAIN_BASELINE")
    decision = publish(rows, data["roster"], data["seeds"], data["design"], policy=policy, remaining=remaining,
                       created_at=policy["created_at"], purpose=policy["purpose"], track=policy["track"], **kwargs)
    bundle = seal(dict(data, version="progressive-decision-bundle/v1", decision=decision, kwargs=kwargs))
    replay = dict(ledger=before, decision=decision, decision_bundle=bundle["identity"],
                  gate_registration=verify_gate_producers(tmp_path, campaign, bundle, store=store))
    table = seal(dict(version="discovery-channel/v1", role="verifier", kind="public_evidence",
                      data=dict(data, version="progressive-decision-input/v1", kwargs=kwargs, remaining=remaining, policy=policy)))
    assert recovery.check_table(table, replay) == bundle
    atomic_write(tmp_path/"table.json", table)
    a = describe(tmp_path, "table.json")
    target = tmp_path/"publisher"
    project(tmp_path, target/"inputs", "publish", {"public":[a]}, registry=recovery.progressive_roles.REGISTRY, authorized=[a.ref])
    result = progressive_role.execute("publish", inputs=target/"inputs", output=target/"role")
    atomic_write(target/"supervision/process.json", dict(returncode=0, cleanup_failure=None))
    atomic_write(target/"supervision/cleanup.json", dict(acknowledged=True, within_bound=True))
    assert recovery.terminal(target, bundle) == result
    atomic_write(tmp_path/"authorization.json", seal(dict(test="synthetic")))
    def interrupt(*args):
        raise InterruptedError
    for obj, name in ((recovery.progressive_outputs, "commit"), (store, "boundary")):
        with monkeypatch.context() as patch:
            patch.setattr(obj, name, interrupt)
            with pytest.raises(InterruptedError):
                recovery.finish(tmp_path, store, task, journal, target, bundle, replay, original)
        assert store.next_task() is None
    for _ in range(2):
        recovery.finish(tmp_path, store, task, journal, target, bundle, replay, original)
    assert len(store.state["decisions"]) == 1
    assert store.next_task()["payload"]["id"] == "trial-00-propose"
    assert load(target/"role/decision-audit.json")["original_budget_prefix"] == before
    assert load(target/"role/gain-cost.json")["costs"][recovery.NAME] == 360
