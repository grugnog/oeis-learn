"""Installed four-input source cycle; all forced outcomes are diagnostic fixtures."""

import json

import pytest

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    record,
    reference,
    seal,
)


def fixture(root):
    from oeis_learn.generalization.progressive.combinatorial import prepare as enumerate_sources
    from oeis_learn.generalization.progressive.generator import structure_id
    from oeis_learn.generalization.progressive.short_learning import prepare
    from oeis_learn.pipeline_validation.progressive_learner import compatibility
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    from tests.contract.test_progressive_sources import source
    from tests.helpers.progressive_oeis_fixture import fixture as oeis_fixture
    from tests.integration.test_progressive_sources import SOURCE

    c, _, old_request, old_plan = oeis_fixture(root)
    # Real fixed-point first-moment values, plus disjoint diagnostic groups.
    exact = enumerate_sources(["fixed_points"], moments=[1])["rows"][0]["values"]
    sources = []
    roles = []
    for group in range(8):
        for variant in range(2):
            values = (
                exact
                if group == variant == 0
                else [str(1000 + 100 * group + 10 * variant + n * n) for n in range(9)]
            )
            sources.append(source(values=values, relations=["group:" + str(group)]))
            roles.append("training" if group < 4 else "evaluation")
    atomic_write(root / "short-roster.json", {"sources": sources, "roles": roles})
    long = "(program (return " + ("(add " * 24) + "$index" + (" (int 1000000))" * 24) + "))"
    row, _ = verify_draw(
        {
            "source": long,
            "ordinal": 99,
            "draw_identity": digest("long-generic-probe"),
            "source_policy": digest("diagnostic"),
            "structure": structure_id(long),
            "detector_limit": "mechanical diagnostic expression",
        }
    )
    assert row is not None
    entries = []
    refs = []
    for entry, p in zip(old_plan["training"], c["payload"]["tasks"]):
        channel = load(root / entry["channel"])
        package = prepare(load(root / channel["data"]["artifact"]["path"]))
        # Add a mechanically checked probe covering the prospective input/target
        # envelope. No external source implementation enters training support.
        package["shards"][0].append(
            seal(dict(row, indices=row["indices"][:4], values=row["values"][:4]))
        )
        package = seal(package)
        arm = "with_source" if p["arm"] == "replay" else "without_source"
        p.update(stage="source_pilot", arm=arm)
        p["initializer"]["compatibility"] = compatibility(package)
        name = p["id"] + "-short-package.json"
        atomic_write(root / name, package)
        cname = p["id"] + "-short-channel.json"
        atomic_write(
            root / cname,
            seal(
                dict(
                    channel,
                    data=dict(
                        channel["data"],
                        artifact=reference(root, name),
                        compatibility=compatibility(package),
                    ),
                )
            ),
        )
        refs.append(
            {
                "name": cname,
                "kind": "prepared",
                "role": "learner",
                "artifact": reference(root, cname),
            }
        )
        model = []
        for path in entry["model"]:
            original = load(root / path)
            original["data"]["compatibility"] = compatibility(package)
            mname = p["id"] + "-short-" + original["kind"] + ".json"
            atomic_write(root / mname, seal(original))
            model.append(mname)
            refs.append(
                {
                    "name": mname,
                    "kind": "prepared",
                    "role": "generator",
                    "artifact": reference(root, mname),
                }
            )
        entries.append({"task": p["id"], "channel": cname, "model": model})
    plan = seal(
        {
            "version": "progressive-source-cycle/v1",
            "roster": reference(root, "short-roster.json"),
            "training": entries,
            "seconds": {"generate": 60, "admit-training": 60, "train": 90, "score": 60},
            "generation": {
                "seed": 13,
                "max_tokens": 128,
                "seconds": 4,
                "forced": {sources[0]["identity"]: SOURCE},
            },
        }
    )
    atomic_write(root / "short-plan.json", plan)
    request = {
        "operation": "source-cycle",
        "artifacts": {"cycle": ["short-plan.json"]},
        "image": old_request["image"],
        "seconds": 600,
        "reservation": old_request["reservation"],
    }
    atomic_write(root / "short-request.json", request)
    for p in c["payload"]["tasks"]:
        p["policy"] = reference(root, "short-request.json")["sha256"]
    c["track"] = "external_assisted"
    c["payload"]["references"] = refs + [
        {
            "name": "short-roster",
            "kind": "source",
            "role": "custodian",
            "artifact": reference(root, "short-roster.json"),
        },
        {
            "name": "short-plan",
            "kind": "policy",
            "role": "custodian",
            "artifact": reference(root, "short-plan.json"),
        },
        {
            "name": "short-request",
            "kind": "policy",
            "role": "custodian",
            "artifact": reference(root, "short-request.json"),
        },
    ]
    graph = c["payload"]["graph"]
    graph["selected_stages"] = [
        "source_pilot" if s == "oeis" else s for s in graph["selected_stages"]
    ]
    for node in graph["nodes"]:
        if node["stage"] == "oeis":
            node["stage"] = "source_pilot"
    # Keep the old historical stage names but add a concrete source node edge.
    # Every task's slot is the existing last node whose stage is now source_pilot.
    c = seal(c)
    manager = record(
        "task",
        dict(c["payload"]["tasks"][0], campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    return c, manager, request


def test_short_cycle_uses_isolated_roles_both_arms_and_full_restore(tmp_path, fitting_client):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.pipeline_validation.progressive_outputs import validate_requests
    from oeis_learn.pipeline_validation.progressive_source_cycle import run_cycle

    c, m, r = fixture(tmp_path)
    validate_requests(
        tmp_path,
        c["payload"]["tasks"],
        c["payload"]["references"],
        c["payload"]["graph"],
        purpose=c["purpose"],
        track=c["track"],
    )
    journal = BudgetJournal(
        tmp_path / "run-state/discovery-accounting" / c["identity"][7:],
        c["identity"],
        c["payload"]["caps"],
    )
    result = run_cycle(tmp_path, tmp_path / "cycle", c, m, r, journal)
    assert result["mechanical_status"] == "COMPLETE", result
    assert result["reason"] == "MATCHED_FINITE_SOURCE_PILOT", result
    assert not result["metrics"]["primary_scaling_eligible"]
    data = load(tmp_path / result["metrics"]["source_results"]["path"])
    assert len(data["results"]) == 4
    assert sorted(v["accepted"] > 0 for v in data["results"].values()) == [False, False, True, True]
    packages = list((tmp_path / "cycle/host/packages").glob("*.json"))
    for path in packages:
        p = load(path)
        assert p["task_profile"] == "combinatorial-short/v1"
        assert all(len(row["values"]) == 4 for shard in p["shards"] for row in shard)
        assert all(len(row["values"]) == 4 for row in p.get("short_rows", []))


def test_short_source_deficit_retains_search_cost_and_no_learning(tmp_path, fitting_client):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.pipeline_validation.progressive_source_cycle import run_cycle

    c, m, r = fixture(tmp_path)
    p = load(tmp_path / "short-plan.json")
    p["generation"]["forced"] = {}
    (tmp_path / "short-plan.json").write_text(json.dumps(seal(p)))
    next(v for v in c["payload"]["references"] if v["name"] == "short-plan")["artifact"] = (
        reference(tmp_path, "short-plan.json")
    )
    c = seal(c)
    m = record(
        "task",
        dict(c["payload"]["tasks"][0], campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    j = BudgetJournal(
        tmp_path / "run-state/discovery-accounting" / c["identity"][7:],
        c["identity"],
        c["payload"]["caps"],
    )
    result = run_cycle(tmp_path, tmp_path / "out", c, m, r, j)
    assert result["reason"] == "SOURCE_YIELD_DEFICIT"
    assert j.task_remaining(m)["draws"] < m["payload"]["caps"]["draws"]
    assert j.task_remaining(m)["tokens"] == m["payload"]["caps"]["tokens"]
    assert not list((tmp_path / "run-state/discovery-learning").glob("**/*.pt"))
    before = j.path.read_bytes()
    assert run_cycle(tmp_path, tmp_path / "resume", c, m, r, j) == result
    assert j.path.read_bytes() == before


def test_short_profile_projection_keeps_nine_term_truth_out_of_generation(tmp_path):
    from oeis_learn.generalization.progressive.cohorts import validate_prompt
    from oeis_learn.generalization.progressive.projections import describe, project
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY

    truth = {
        "id": "private",
        "indices": list(map(str, range(9))),
        "values": list(map(str, range(9))),
    }
    atomic_write(
        tmp_path / "truth.json",
        seal(
            {
                "version": "discovery-channel/v1",
                "kind": "short_training_truth",
                "role": "generator",
                "data": truth,
            }
        ),
    )
    with pytest.raises(ValueError):
        project(
            tmp_path,
            "forbidden",
            "generate",
            {"visible": [describe(tmp_path, "truth.json")], "model": []},
            registry=REGISTRY,
        )
    with pytest.raises(ValueError):
        validate_prompt(
            {"id": "ordinary", "indices": ["0", "1", "2", "3"], "values": ["0", "1", "2", "3"]}
        )


def test_canonical_prepare_freeze_run_resume_short_source_cycle(
    tmp_path, monkeypatch, fitting_client, capsys
):
    from types import SimpleNamespace

    from oeis_learn.pipeline_validation import progressive_source_cycle as source
    from oeis_learn.pipeline_validation.progressive_commands import main
    from tests.helpers.progressive_host_cycle import accept_prior_stages, freeze_inputs

    prefix = freeze_inputs(tmp_path, monkeypatch, source=True)
    assert (
        main(
            SimpleNamespace(
                operation="prepare",
                blueprint="cycle-blueprint.json",
                qualification="qualification",
                output="prepared-cycle",
            ),
            tmp_path,
            None,
        )
        == 0
    )
    assert (
        main(
            SimpleNamespace(
                operation="freeze",
                blueprint="cycle-blueprint.json",
                prepared="prepared-cycle",
                qualification="qualification",
                output="frozen-cycle",
            ),
            tmp_path,
            None,
        )
        == 0
    )
    c = load(tmp_path / "frozen-cycle/campaign.json")
    accept_prior_stages(tmp_path, c, prefix, monkeypatch)
    original = source.run_cycle
    monkeypatch.setattr(
        source,
        "run_cycle",
        lambda root, output, campaign, manager, request, journal, **kw: original(
            root, output, campaign, manager, request, journal
        ),
    )
    assert (
        main(
            SimpleNamespace(
                operation="run",
                campaign="frozen-cycle/campaign.json",
                acceptance="accepted",
                output="run-cycle",
            ),
            tmp_path,
            lambda a: object(),
        )
        == 0
    )
    assert load(tmp_path / "run-cycle/result.json")["mechanical_status"] == "COMPLETE"
    journal = tmp_path / "run-state/discovery-accounting" / c["identity"][7:] / "budget.jsonl"
    before = journal.read_bytes()
    assert (
        main(
            SimpleNamespace(operation="resume", run="run-cycle", output="resumed-cycle"),
            tmp_path,
            lambda a: object(),
        )
        == 0
    )
    assert journal.read_bytes() == before

    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.pipeline_validation.progressive_outputs import commit_host_training
    store=RunStore(tmp_path/'run-state/discovery-control'/c['identity'][7:],c)
    expected={p['id'] for p in c['payload']['tasks'] if p['stage']=='source_pilot'}
    assert set(store.state['outputs'])==expected
    assert all(cut['operation']=='train' for cut in store.state['outputs'].values())
    j=BudgetJournal(journal.parent,c['identity'],c['payload']['caps'])
    marker=load(j.root/'source-cycle.json')
    owner=tmp_path/marker['output']
    step=next(load(p) for p in (owner/'host/steps').glob('*.json') if load(p)['receipt']['operation']=='train')
    original_result=load(tmp_path/step['result']['path'])
    atomic_write(tmp_path/'foreign/result.json',original_result)
    step['result']=reference(tmp_path,'foreign/result.json')
    step['receipt']['result']=step['result']
    atomic_write(owner/'host/steps/999.json',seal(step))
    with pytest.raises(ValueError,match='foreign host child'):
        commit_host_training(tmp_path,store,load(tmp_path/'short-request.json'),j,sorted(expected))
    assert journal.read_bytes()==before


def test_installed_short_learner_resumes_updates_after_owned_pause(tmp_path, fitting_client):
    import threading
    import time

    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.recovery import Recovery
    from oeis_learn.pipeline_validation.progressive_acceptance import invoke_cpu

    c, _, _ = fixture(tmp_path)
    plan = next(p for p in c["payload"]["tasks"] if p["arm"] == "without_source")
    plan["token_target"] = 8000
    c = seal(c)
    task = record(
        "task",
        dict(plan, campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    source = load(tmp_path / "short-plan.json")
    entry = next(e for e in source["training"] if e["task"] == plan["id"])
    channel = load(tmp_path / entry["channel"])
    state = tmp_path / "learning" / task["identity"][7:]
    errors = []
    done = threading.Event()

    def pause():
        try:
            deadline = time.monotonic() + 60
            while not done.is_set() and time.monotonic() < deadline:
                events = sorted(state.glob("event-*.json"))
                if events and load(events[-1])["state"]["useful_tokens"] > 0:
                    atomic_write(
                        tmp_path / "control" / ("pause-" + digest("invocation-0")[7:] + ".json"),
                        seal({"campaign": c["identity"], "requested": True}),
                    )
                    return
                time.sleep(0.02)
            if not done.is_set():
                raise TimeoutError("first committed short update absent")
        except (OSError, ValueError, TimeoutError) as exc:
            errors.append(exc)

    monitor = threading.Thread(target=pause, daemon=True)
    monitor.start()
    try:
        first, _ = invoke_cpu(tmp_path, c, task, channel, ordinal=0, channel_path=entry["channel"])
    finally:
        done.set()
        monitor.join(timeout=2)
    assert not errors, errors
    assert first["metrics"]["task_status"] == "PAUSED", first
    j = BudgetJournal(tmp_path / "accounting", c["identity"], c["payload"]["caps"])
    with Recovery(j, task, plan["initializer"]["compatibility"], state_root=state) as owner:
        prior = owner.restore()["ledger"].updates
    resumed, _ = invoke_cpu(tmp_path, c, task, channel, ordinal=1, channel_path=entry["channel"])
    assert resumed["mechanical_status"] == "COMPLETE", resumed
    assert resumed["metrics"]["initialization"] == "resume"
    with Recovery(j, task, plan["initializer"]["compatibility"], state_root=state) as owner:
        assert owner.restore()["ledger"].updates > prior


def test_research_nested_training_reserves_are_preflighted():
    from oeis_learn.pipeline_validation.progressive_source_cycle import validate_training_allowance
    task = dict(arm='with_source', startup_us=1000000, finalization_us=1800000000,
                iteration_budget=dict(update_us=1, probe_us=1, checkpoint_us=1, producer={}),
                caps=dict(elapsed_us=21600000000))
    seconds = dict(train=600, generate=60, score=60, **{'admit-training':60})
    with pytest.raises(ValueError, match='startup/work/finalization'):
        validate_training_allowance(task, seconds, 512)
    validate_training_allowance(task, dict(seconds, train=3600), 512)
    with pytest.raises(ValueError, match='original child allowance'):
        validate_training_allowance(dict(task, caps=dict(elapsed_us=4000000000)), dict(seconds, train=3600), 512)
