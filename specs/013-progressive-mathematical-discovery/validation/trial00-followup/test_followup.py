"""Run on CPU. Check real journal/publication contracts with small, synthetic records."""
import copy
import json
from pathlib import Path
import runpy

import pytest

from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore, ProgressiveCheckpointStore
from oeis_learn.generalization.budgets import Ledger
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, reference, seal
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.recovery import Recovery
from oeis_learn.pipeline_validation.identity import file_hash
from tests.unit.test_progressive_freeze import campaign as base_campaign
from tests.helpers.progressive_fixture import envelope
from execution import finish, task_for
from evidence import audit, measure, next_action, summarize, verify_history
from import_parent import import_cut
from registration import task_record
from reports import saved_measurement, write_report

case = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                          "test_round027_screening_audit.py"))["case"]


def put(root, name, data):
    p = root / name
    atomic_write(p, data)
    return reference(root, p)


def test_real_parent_import_and_linked_children(tmp_path, monkeypatch):
    # Exercise the installed state format, lineage, transaction and charge code.
    # The tiny CPU fixture uses JSON blobs in place of Torch's tensor codec.
    monkeypatch.setattr(GeneralizationCheckpointStore, "_encode",
                        staticmethod(lambda v: json.dumps(v).encode()))
    monkeypatch.setattr(GeneralizationCheckpointStore, "_decode", staticmethod(json.loads))
    c = base_campaign()
    p = c["payload"]["tasks"][0]
    scope = p["initializer"]["compatibility"]
    source = task_record(c, p)
    cuts = {
        k: digest(None)
        for k in ("stream", "allocation", "archive", "library", "cost_journal", "selection",
                  "milestones", "consumed")
    }
    states = dict(weights={"w": [1]},
                  optimizer={"optimizer": {
                      "state": {
                          "moments": [2, 3]
                      }
                  }},
                  scheduler={
                      "position": 1000564,
                      "controller": {
                          "cursor": 20448
                      }
                  },
                  rng={k: [1, 2, 3]
                       for k in ("python", "numpy", "torch", "device")},
                  data=dict(cursor=1278, pending_batch=None),
                  history={},
                  selection={},
                  replay={},
                  pins={})
    parent = ProgressiveCheckpointStore(tmp_path / "original/states",
                                        compatibility=scope,
                                        campaign=c["identity"],
                                        task=source["identity"],
                                        token_cap=1100000)
    cp = parent.save(states,
                     Ledger(1100000, consumed=1000564, updates=1278),
                     cuts=cuts,
                     created_at=c["created_at"],
                     purpose=c["purpose"],
                     track=c["track"])
    cp_ref = put(tmp_path, "original/cp.json", cp)
    settled = seal(
        dict(version="progressive-final-accounting/v1",
             task=source["identity"],
             pending=[],
             milestone={
                 "tokens": 1000564,
                 "total_cost_us": 578661207
             },
             useful_tokens=1000564,
             checkpoint=cp["identity"],
             ancestor_cost_available=True))
    acc = put(tmp_path, "original/final.json", settled)
    spec = dict(artifact=cp_ref, state_root="original/states", accounting=acc)
    origin = put(tmp_path, "original/origin.json", seal({"fixture": True}))
    donor = dict(p,
                 id="donor",
                 output="new/donor",
                 caps=dict(p["caps"], tokens=0),
                 initializer=dict(mode="full_state",
                                  compatibility=scope,
                                  artifact=cp_ref,
                                  origin=origin))
    plans = [donor]
    for ident in ("look-2", "look-4"):
        plans.append(
            dict(p,
                 id=ident,
                 output="new/" + ident,
                 caps=dict(p["caps"], tokens=10),
                 initializer=dict(mode="full_state",
                                  compatibility=scope,
                                  artifact=None,
                                  parent_task=plans[-1]["id"])))
    graph = copy.deepcopy(c["payload"]["graph"])
    for node in graph["nodes"]:
        node["caps"] = {k: sum(t["caps"][k] for t in plans) for k in p["caps"]}
    c2 = envelope("campaign", dict(c["payload"], tasks=plans, graph=graph))
    tasks = [task_record(c2, t) for t in plans]
    request = seal(
        dict(task=tasks[0],
             source=spec,
             parent=cp["identity"],
             caps=c2["payload"]["caps"],
             checkpoint_bytes=2 << 20,
             journal_bytes=128 << 20))
    before = {f: file_hash(f) for f in (tmp_path / "original").rglob("*") if f.is_file()}
    new = tmp_path / "new"
    import_cut(request, tmp_path, new / "account", new / "donor")
    import_cut(request, tmp_path, new / "account", new / "donor")
    j = BudgetJournal(new / "account", c2["identity"], c2["payload"]["caps"])
    for idx in (1, 2):
        with Recovery(j, tasks[idx - 1], scope, state_root=new / plans[idx - 1]["id"]) as old:
            with Recovery(j, tasks[idx], scope, state_root=new / plans[idx]["id"]) as child:
                child.initialize_from(old, campaign=c2)
                data = child.restore()
                assert data["ledger"].consumed == 1000564 + (idx - 1) * 3
                assert data["ledger"].updates == 1278 + idx - 1
                assert data["states"]["optimizer"] == states["optimizer"] and data["states"][
                    "rng"] == states["rng"]
                assert child.base["ancestor_cost_us"] == 578661207 + (idx - 1) * 100
                child.begin(boot="boot", monotonic_us=1000)
                debit = child.reserve(["example"], 3, boot="boot", monotonic_us=1010)
                updated = data["states"]
                updated["data"]["cursor"] += 1
                child.commit(updated, debit.commit(debit.pending[0]), cuts)
                child.close(boot="boot", monotonic_us=1100, complete=True)
    assert j.snapshot()["consumed"]["tokens"] == 6
    assert before == {f: file_hash(f) for f in before}
    assert load(new / "donor/import.json")["new_updates"] == 0


@pytest.mark.parametrize("settled", [False, True])
def test_interruption_after_worker_before_publication(tmp_path, monkeypatch, settled):
    c = base_campaign()
    p = c["payload"]["tasks"][0]
    p.update(kind="evaluation", id="trial-00-2000000-development-score")
    c = seal(c)
    state, acct, target = (tmp_path / k for k in ("control", "account", "worker"))
    store = RunStore(state, c)
    task = task_for(store, p["id"])
    j = BudgetJournal(acct, c["identity"], c["payload"]["caps"])
    cost = dict.fromkeys(p["caps"], 0)
    cost.update(elapsed_us=500, attempts=1)
    j.task_remaining(task)
    key = j.reserve(p["id"], "score", 0, cost, boot="fixture", monotonic_us=100)["key"]
    if settled:
        j.commit(key, boot="fixture", monotonic_us=300, termination_proven=True)
    put(target, "supervision/process.json",
        dict(outcome="COMPLETE", returncode=0, cleanup_failure=None))
    put(target, "supervision/cleanup.json", dict(acknowledged=True, within_bound=True))
    put(
        target, "role/result.json",
        seal(
            dict(version="discovery-result/v1",
                 operation="score",
                 mechanical_status="COMPLETE",
                 campaign=c["identity"],
                 task=task["identity"],
                 metrics={})))
    from oeis_learn.generalization.progressive.storage import register_invocation
    register_invocation(tmp_path, c, task, j.root, target)

    def crash(*args):
        raise InterruptedError("lost power after output cut")

    monkeypatch.setattr(store, "boundary", crash)
    with pytest.raises(InterruptedError):
        finish(tmp_path, store, j, task, target, key, "score")
    before = (acct / "budget.jsonl").read_bytes()
    for _ in range(2):
        store = RunStore(state, c)
        finish(tmp_path, store, BudgetJournal(acct, c["identity"], c["payload"]["caps"]), task,
               target, key, "score")
    assert (acct / "budget.jsonl").read_bytes() == before
    assert j.snapshot()["consumed"]["elapsed_us"] == (200 if settled else 500)
    assert j.snapshot()["consumed"]["attempts"] == 1 and len(j.snapshot()["committed"]) == 1
    assert store.state["tasks"][p["id"]] == "COMPLETE"


def synthetic_evidence(root, coordinate, bad_count):
    keys = [f"case-{i:03d}" for i in range(128)]
    roster = [
        dict(task=k,
             stratum=("composition", "unseen_structure", "parameter", "oeis")[i // 32],
             group=str(i // 2)) for i, k in enumerate(keys)
    ]
    prompts = {k: dict(indices=["0", "1"], id=k, values=["0", "1"]) for k in keys}
    weights = root / "weights"
    weights.write_bytes(b"measurement fixture")
    refs = dict(checkpoint=reference(root, weights),
                qualification=put(root, "q.json", seal(dict(status="PASS", device="actual_HIP"))))
    outputs, observed, ids = {}, {}, {}
    for name, prefix in ((f"{coordinate}-development", ""),
                         ("retention-baseline", "retention_baseline_"), (f"{coordinate}-retention",
                                                                         "retention_candidate_")):
        recs, scores, clocks = {}, {}, []
        for i, k in enumerate(keys):
            capped = i < 6 and prefix == ""
            row, s = case(capped)
            if i < bad_count and capped:
                row["costs"]["correct"]["details"][0]["generation"]["reason"] = "timeout"
            recs[k], scores[k] = row, {"equal_budget": s}
            clocks.append(audit["clock_record"](prompt=k,
                                                mode="correct",
                                                snapshot="model",
                                                policy=audit["DRAIN"],
                                                start=0,
                                                scope_deadline=500,
                                                finish=2,
                                                expected=8,
                                                started=8,
                                                terminal=8))
        observed[prefix] = {k: r["equal_budget"]["outcome"] for k, r in scores.items()}
        gen = seal(
            dict(version="discovery-candidates/v1",
                 model="model",
                 methods=["correct"],
                 checkpoint=refs["checkpoint"]["sha256"],
                 prompts=digest([prompts[k] for k in keys]),
                 records=recs,
                 clock=seal(dict(policy=audit["DRAIN"], clocks=clocks))))
        for op, tag, metric in (("generate", "generation", "candidates"), ("score", "scoring",
                                                                           "scores")):
            task = "trial-00-" + name + "-" + op
            role = Path(task) / "role"
            doc = gen if op == "generate" else seal(
                dict(version="discovery-scoring/v1",
                     denominator=128,
                     candidates=refs[prefix + "generation"]["sha256"],
                     outcomes=scores))
            refs[prefix + tag] = put(root, role / (metric + ".json"), doc)
            result = put(
                root, role / "result.json",
                seal(
                    dict(operation=op,
                         mechanical_status="COMPLETE",
                         metrics={metric: reference(root / role, metric + ".json")})))
            outputs[task] = dict(result=result, operation=op)
            ids[prefix + tag] = doc["identity"]
    evidence = seal(
        dict(version="progressive-trial-evidence/v1",
             role="development",
             scope="NATIVE_VERIFIED",
             costs=dict(startup=0, training=1, inference=1, search=0, verification=1),
             cost_scope="supervised task intervals; startup and cleanup enclosed in phase costs",
             producer_refs=refs,
             checkpoint=refs["checkpoint"]["sha256"],
             qualification=load(root / "q.json")["identity"],
             generator=ids["generation"],
             scorer=ids["scoring"],
             roster=roster,
             outcomes=[dict(r, outcome=observed[""][r["task"]]) for r in roster],
             retention=[
                 dict(task=k,
                      baseline=observed["retention_baseline_"][k],
                      candidate=observed["retention_candidate_"][k]) for k in keys
             ]))
    return evidence, outputs, prompts.__getitem__


@pytest.mark.parametrize(("look", "missing", "action"), [(2000000, 1, "CONTINUE_TO_4M"),
                                                         (2000000, 2, "STOP_AT_2M"),
                                                         (4000000, 1, "STOP_AT_4M"),
                                                         (4000000, 2, "STOP_AT_4M")])
def test_look_from_sealed_workers_and_interrupted_report(tmp_path, look, missing, action):
    e, outputs, prompt = synthetic_evidence(tmp_path, look, missing)
    path = tmp_path / "private.json"
    # Simulated exit after evidence is saved, before publication. Reopen and verify
    # against the same private producer files without any worker dispatch.
    initial = saved_measurement(tmp_path, path, 0, look, e, outputs, prompt)
    reopened = saved_measurement(tmp_path, path, 0, look, e, outputs, prompt)
    assert initial == reopened and next_action(reopened) == action
    row = summarize(reopened)
    assert row["frozen_reason"] == "development completion below 95 percent"
    assert row["companion"]["development"]["composition"]["unresolved"] == missing
    assert row["frozen"]["verified_yield"] == row["companion"]["verified_yield"] == 0
    path = tmp_path / "report.json"
    report = write_report(path, dict(next_action=next_action(initial), metrics=row))
    assert write_report(path, dict(next_action=next_action(reopened), metrics=row)) == report
    path.write_text(json.dumps(dict(report, next_action="changed")))
    with pytest.raises(ValueError):
        write_report(path, dict(next_action=next_action(reopened), metrics=row))
    (tmp_path / e["producer_refs"]["generation"]["path"]).write_bytes(b"modified")
    with pytest.raises(ValueError):
        saved_measurement(tmp_path, tmp_path / "private.json", 0, look, e, outputs, prompt)


def test_reference_witness_requires_parameter_independent_control():
    from oeis_learn.synthesis.admission import source_identity
    from evidence import reference_check
    for source, verdict in (("(program (return (to_bigint (to_control (int -1)))))",
                             "reproduced_control_range"),
                            ("(program (return (to_bigint (to_control (param.get p0)))))",
                             "control_depends_on_fit_parameters")):
        stop = dict(phase="symbolic", outcome="resource_exhausted", reason="control_range")
        draw = dict(source=source,
                    source_sha256=source_identity(source),
                    complete=True,
                    generation=dict(complete=True),
                    fitting=dict(complete=False, admission=[], **stop),
                    native=dict(assignments=[], searches=[], completed_bounds=[], **stop))
        assert reference_check(draw, ["0", "1"]) == verdict
        draw["native"]["reason"] = "timeout"
        assert reference_check(draw, ["0", "1"]) == "other_incomplete_fit"


def test_history_disallows_missing_or_changed_originals():
    with pytest.raises(ValueError, match="six"):
        verify_history([])
    with pytest.raises(ValueError, match="differs"):
        verify_history(
            [dict(trial=i, coordinate=1000000, frozen={}, companion={}) for i in range(6)])


def test_scoring_original_above_small_receipt_limit(tmp_path):
    e, outputs, prompt = synthetic_evidence(tmp_path, 2000000, 0)
    name = "trial-00-2000000-development-score"
    ref = e["producer_refs"]["scoring"]
    path = tmp_path / ref["path"]
    with path.open("ab") as stream:
        stream.write(b" " * (32 << 20))
    e["producer_refs"]["scoring"] = reference(tmp_path, path)
    result = tmp_path / outputs[name]["result"]["path"]
    old = load(result)
    old["metrics"]["scores"] = reference(result.parent, path)
    result.write_text(json.dumps(seal(old)))
    outputs[name]["result"] = reference(tmp_path, result)
    assert next_action(measure(tmp_path, 0, 2000000, seal(e), outputs, prompt)) == "CONTINUE_TO_4M"


@pytest.mark.parametrize("decision", ["STOP_AT_2M", "CONTINUE_TO_4M", "report interrupted"])
def test_launcher_dispatches_only_after_published_decision(tmp_path, monkeypatch, decision):
    from types import SimpleNamespace
    import launch
    from registration import names
    executed = []

    def dispatch(root, home, child, journal, host, cache, name):
        executed.append(name)

    def publish(root, home, source, child, journal, coord, one, prior, earlier, prompt, replay):
        executed.append("publish-" + str(coord))
        if decision == "report interrupted":
            raise InterruptedError()
        return {}, {"next_action": decision if coord == 2000000 else "STOP_AT_4M"}

    monkeypatch.setattr(launch, "execute", dispatch)
    monkeypatch.setattr(launch, "publish_look", publish)
    args = (tmp_path, tmp_path, None, SimpleNamespace(state={"outputs": {}}), None, None, {}, {},
            None, False)
    if decision == "report interrupted":
        with pytest.raises(InterruptedError):
            launch.drive_looks(*args)
    else:
        launch.drive_looks(*args)
    expected = [*names(2000000), "publish-2000000"]
    if decision == "CONTINUE_TO_4M":
        expected.extend([*names(4000000), "publish-4000000"])
    assert executed == expected
