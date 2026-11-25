"""Bounded CPU recovery and next-proposal checks. All state lives under tmp_path."""

import io
import itertools
import json
import shutil
import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from oeis_learn.generalization.budgets import Ledger
from oeis_learn.generalization.progressive.budgets import KEYS, BudgetJournal
from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.decisions import publish, verify_gate_producers
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.projections import project
from oeis_learn.pipeline_validation import progressive_evidence, progressive_role, progressive_roles
from tests.contract.test_progressive_roles import artifact
from tests.unit.test_progressive_dag_decisions import configured


def adapter(filename):
    directory = (
        Path(__file__).resolve().parents[2] / "013-progressive-mathematical-discovery/validation"
    )
    spec = spec_from_file_location(filename, directory / (filename + ".py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_checkpoint(root, output, campaign, task, state, expected_weights):
    from oeis_learn.fresh.bigint_state import ProgressiveCheckpointStore

    directory = (
        root / "run-state/discovery-learning" / campaign["identity"][7:] / task[7:] / "checkpoints"
    )
    storage = ProgressiveCheckpointStore(
        directory,
        campaign=campaign["identity"],
        task=task,
        compatibility=digest("tiny learner"),
        token_cap=100,
    )
    ledger = Ledger(100).reserve("tiny batch", 8).commit("tiny batch")
    state["data"]["cursor"] = ledger.updates
    cuts = {
        key: digest([task, key])
        for key in (
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
    checkpoint = storage.save(
        state,
        ledger,
        cuts=cuts,
        created_at=campaign["created_at"],
        purpose=campaign["purpose"],
        track=campaign["track"],
    )
    full = load(directory / checkpoint["payload"]["full_state"]["path"])
    assert full["states"]["weights"]["sha256"] == expected_weights
    atomic_write(output / "checkpoint.json", checkpoint)
    atomic_write(
        output / "accounting-final.json",
        seal(
            {
                "version": "progressive-final-accounting/v1",
                "task": task,
                "checkpoint": checkpoint["identity"],
                "pending": [],
                "useful_tokens": 8,
                "milestone": {"tokens": 8, "total_cost_us": 1},
            }
        ),
    )
    return {"checkpoint_record": reference(output, "checkpoint.json"), "useful_tokens": 8}


def comparison(root):
    original, data, policy, remaining = configured(root)
    campaign = original.campaign
    gate = load(root / "gate.json")
    import torch

    from tests.integration.test_progressive_recovery import states

    models = {}
    for i, producer in enumerate(gate["producers"]):
        arm, seed = producer["arm"], producer["seed"]
        name = f"{arm}-{seed}"
        model = states()  # Weights and optimizer after one actual CPU AdamW update.
        model["weights"]["arm_seed_buffer"] = torch.tensor(i)
        models[name] = model
        encoded = io.BytesIO()
        torch.save(model["weights"], encoded)
        weights = root / (name + ".pt")
        weights.write_bytes(encoded.getvalue())
        producer["checkpoint"] = reference(root, weights)
        candidates = load(root / producer["candidates"]["path"])
        scores = load(root / producer["scores"]["path"])
        generated = root / (name + "-bound-generated.json")
        atomic_write(generated, seal(dict(candidates, checkpoint=producer["checkpoint"]["sha256"])))
        producer["candidates"] = reference(root, generated)
        scores["candidates"] = producer["candidates"]["sha256"]
        # One partly observed seed/arm makes the correct decision INCOMPLETE.
        if (arm, seed) == ("candidate", 1):
            for row in data["rows"]:
                if (row["arm"], row["seed"], row["outcome"]) == (arm, seed, "success"):
                    row["complete"] = False
                    scores["outcomes"][row["task"]]["equal_budget"]["complete"] = False
        scored = root / (name + "-bound-scores.json")
        atomic_write(scored, seal(scores))
        producer["scores"] = reference(root, scored)
    atomic_write(root / "partial-gate.json", seal(gate))
    next(r for r in campaign["payload"]["references"] if r["name"] == "gate")["artifact"] = (
        reference(root, "partial-gate.json")
    )

    plans = campaign["payload"]["tasks"]
    # Keep 35 upstream cuts in their own producer directories and 12 in the stopped round.
    kinds = {f"predecessor-{i:02d}": "census" for i in range(35)}
    for p in gate["producers"]:
        for op in ("train", "generate", "score"):
            kinds[f"{p['arm']}-{p['seed']}-{op}"] = op
    previous = [dict(plans[0], id=name, output=name) for name in kinds]
    compare = dict(
        plans[0], id="diversity-comparison", output="compare", kind="evaluation", token_target=0
    )
    campaign["payload"]["tasks"] = previous + [compare, dict(plans[-1], id="trial-00-propose")]
    graph = campaign["payload"]["graph"]
    graph["nodes"][0]["caps"] = {
        k: len(previous + [compare]) * v for k, v in compare["caps"].items()
    }
    graph["edges"][0]["condition"] = "COMPLETE"
    campaign = seal(campaign)
    fields = {
        k: data[k] for k in ("source_slot", "target_slot", "rows", "roster", "seeds", "design")
    }
    kwargs = {"target_id": "gain", "action": "RETAIN_BASELINE"}
    decision = publish(
        data["rows"],
        data["roster"],
        data["seeds"],
        data["design"],
        policy=policy,
        remaining=remaining,
        created_at=policy["created_at"],
        purpose=policy["purpose"],
        track=policy["track"],
        **kwargs,
    )
    bundle = seal(
        dict(fields, version="progressive-decision-bundle/v1", decision=decision, kwargs=kwargs)
    )
    table = dict(
        fields,
        version="progressive-decision-input/v1",
        kwargs=kwargs,
        remaining=remaining,
        policy=policy,
    )
    task = record(
        "task",
        dict(compare, campaign=campaign["identity"]),
        created_at=campaign["created_at"],
        purpose=campaign["purpose"],
        track=campaign["track"],
    )
    return campaign, kinds, gate, task, bundle, table, models


def completed_outputs(root, session, campaign, kinds, gate, models):
    cuts = {}
    for i, (name, op) in enumerate(kinds.items()):
        directory = (
            session if i >= 35 else session.with_name(f"013-research-round-{22 + i % 4:03d}")
        )
        task = record(
            "task",
            dict(campaign["payload"]["tasks"][i], campaign=campaign["identity"]),
            created_at=campaign["created_at"],
            purpose=campaign["purpose"],
            track=campaign["track"],
        )
        output = (
            directory / "run-research/execution-0000" / ("task-" + task["identity"][7:]) / "role"
        )
        if op == "census":
            value = seal(
                {"version": "diagnostic-census/v1", "ordinal": i, "selected": digest(name)}
            )
            atomic_write(output / "evidence.json", value)
            metrics = {"evidence": reference(output, "evidence.json")}
        else:
            arm, seed, _ = name.split("-")
            p = next(p for p in gate["producers"] if (p["arm"], p["seed"]) == (arm, int(seed)))
            if op == "train":
                metrics = full_checkpoint(
                    root,
                    output,
                    campaign,
                    task["identity"],
                    models[f"{arm}-{seed}"],
                    p["checkpoint"]["sha256"],
                )
            else:
                ref = p["candidates" if op == "generate" else "scores"]
                atomic_write(output / "evidence.json", load(root / ref["path"]))
                metrics = {"evidence": reference(output, "evidence.json")}
                assert metrics["evidence"]["sha256"] == ref["sha256"]
        progressive_evidence.publish(
            output,
            operation=op,
            mechanical_status="COMPLETE",
            reason="BOUNDED_FIXTURE",
            task=task["identity"],
            metrics=metrics,
        )
        cuts[name] = {
            "task": task["identity"],
            "operation": op,
            "result": reference(root, output / "result.json"),
        }
    return cuts


def test_guarded_representative_recovery(tmp_path, monkeypatch):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.pipeline_validation import progressive_workstation
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file

    audit = adapter("replay-round027-comparison")
    helper = adapter("continue-round027-publication")
    import foundation_local

    campaign, kinds, gate, task, bundle, table, models = comparison(tmp_path)
    cid = campaign["identity"]
    session = tmp_path / helper.SESSION
    control = tmp_path / "run-state/discovery-control" / cid[7:]
    account = tmp_path / "run-state/discovery-accounting" / cid[7:]
    store = RunStore(control, campaign)
    journal = BudgetJournal(account, cid, campaign["payload"]["caps"])
    atomic_write(session / "prepare-research/frozen/campaign.json", campaign)
    cuts = completed_outputs(tmp_path, session, campaign, kinds, gate, models)
    tasks = dict.fromkeys(cuts, "COMPLETE")
    tasks[helper.NAME] = "FAILED"
    store.append(dict(store.state, outputs=cuts, tasks=tasks, status="FAILED"))
    failure = seal({"metrics": {"tasks": tasks}})
    atomic_write(session / "run-research/execution-0000/result.json", failure)
    zero = dict.fromkeys(KEYS, 0)
    for ordinal, (name, op) in enumerate(kinds.items()):
        cost = dict(zero, tokens=8 if op == "train" else 0, elapsed_us=1)
        journal.reserve(name, op, ordinal, cost, boot="fixture", monotonic_us=0)
        journal.commit(f"{name}/{op}/{ordinal}")
    journal.reserve(
        helper.NAME,
        "publish",
        95,
        dict(zero, elapsed_us=5, native_work=1),
        boot="fixture",
        monotonic_us=0,
    )
    journal.commit(helper.NAME + "/publish/95")
    note = {
        "stage": "PASS",
        "decision": bundle["decision"],
        "decision_bundle": bundle["identity"],
        "gate_registration": verify_gate_producers(tmp_path, campaign, bundle, store=store),
    }
    audit.CAMPAIGN, audit.FAILURE = cid, failure["identity"]
    audit.audit(tmp_path, control, account, note)
    checkpoints = list(note["training_checkpoints"].values())
    assert len(cuts) == 47 and len(checkpoints) == 4
    assert all(cp["useful_tokens"] == 8 and cp["files"] for cp in checkpoints)
    assert sum(str(helper.SESSION) in cut["result"]["path"] for cut in cuts.values()) == 12
    sealed = seal(note)
    tools_dir = tmp_path / "run-state/discovery-tools"
    tools_dir.mkdir()
    with ZipFile(tools_dir / "round027-comparison-fixture.zip", "w") as archive:
        archive.writestr("replay.json", json.dumps(sealed))
    helper.ROOT, helper.CAMPAIGN, helper.REPLAY = tmp_path, cid, sealed["identity"]
    checkout = Path(progressive_role.__file__).resolve().parents[3]
    files = collect_snapshot(checkout)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=checkout).decode().split("\0")
    for name in filter(None, tracked):
        if name not in files and not name.startswith((".specify/memory/", "specs/")):
            continue
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkout / name, dest)
    assert collect_snapshot(tmp_path) == files
    atomic_write(
        session / "session.json",
        seal(
            {
                "binding": {
                    "source": digest(collect_snapshot(tmp_path)),
                    "engine": "docker",
                    "memory_gib": 1,
                    "cpus": 1,
                    "render_node": None,
                }
            }
        ),
    )
    atomic_write(
        session / "journal.json",
        seal({"steps": {"prepare-research/acceptance": {"status": "COMPLETE"}}}),
    )
    (session / "session.lock").touch()

    target = session / "run-research/publication-recovery/publisher"
    channel = artifact(tmp_path, "public-table.json", "public_evidence", "verifier", table)
    assert helper.check_table(load(tmp_path / "public-table.json"), sealed) == bundle
    project(
        tmp_path,
        target / "inputs",
        "publish",
        {"public": [channel]},
        registry=progressive_roles.REGISTRY,
    )
    progressive_role.execute("publish", inputs=target / "inputs", output=target / "role")
    atomic_write(target / "supervision/process.json", {"returncode": 0, "cleanup_failure": None})
    atomic_write(target / "supervision/cleanup.json", {"acknowledged": True, "within_bound": True})
    atomic_write(
        target.parent / "authorization.json", seal({"test": "representative CPU publication"})
    )
    _, amendment = helper.charge_state(journal, sealed)
    journal.reserve(
        helper.NAME, "publish", helper.ORDINAL, amendment, boot="fixture", monotonic_us=0
    )

    # The historical envelope binds the private campaign and device. This fixture uses its
    # own envelope and the same authenticated cut, terminal, charge and transition operations.
    def reconcile(root, binding, folder, replay):
        assert (root, folder, replay) == (tmp_path, session, sealed)
        with store.lease():
            old = helper.original_state(root, store, replay)
            for cut in replay["outputs"].values():
                result = checked_file(root, cut["result"])
                for ref in load(result)["metrics"].values():
                    if isinstance(ref, dict):
                        checked_file(result.parent, ref)
            for cp in replay["training_checkpoints"].values():
                for ref in cp["files"]:
                    checked_file(root, ref)
            tail, _ = helper.charge_state(journal, replay)
            helper.terminal(target, bundle)
            if len(tail) == 1:
                journal.commit(helper.KEY, boot="fixture", monotonic_us=3, termination_proven=True)
            assert len(helper.charge_state(journal, replay)[0]) == 2
            helper.finish(root, store, task, journal, target, bundle, replay, old)

    class HandoffReached(Exception):
        pass

    def stop(root, argv):
        assert root == tmp_path and argv[-1] == "run-research"
        assert RunStore(control, campaign).next_task()["payload"]["id"] == "trial-00-propose"
        raise HandoffReached

    handoff = Mock(side_effect=stop)
    worker = Mock(side_effect=AssertionError("unexpected worker"))
    host = Mock(side_effect=AssertionError("unexpected engine"))
    monkeypatch.setattr(helper, "recover", reconcile)
    monkeypatch.setattr(progressive_workstation, "main", handoff)
    monkeypatch.setattr(progressive_roles, "invoke", worker)
    monkeypatch.setattr(foundation_local, "Host", host)

    # Failed prerequisites on the same stopped cut must leave its transition and charges alone.
    initial = (store.tip, journal.path.read_bytes())
    with store.lease(), pytest.raises(ValueError, match="owns"):
        helper.continue_session(session)
    producer = tmp_path / next(iter(cuts.values()))["result"]["path"]
    sidecar = next(ref for ref in checkpoints[0]["files"] if "optimizer" in ref["path"])
    for path in (
        producer.parent / "evidence.json",
        tmp_path / sidecar["path"],
        target / "supervision/cleanup.json",
    ):
        data = path.read_bytes()
        if path.name == "cleanup.json":
            path.unlink()  # Reservation survived without its supervised completion.
        else:
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with pytest.raises((FileNotFoundError, ValueError)):
            helper.continue_session(session)
        path.write_bytes(data)
        assert (RunStore(control, campaign).tip, journal.path.read_bytes()) == initial
        handoff.assert_not_called()

    for _ in range(2):
        with pytest.raises(HandoffReached):
            helper.continue_session(session)
        after = RunStore(control, campaign)
        assert {k: v for k, v in after.state["outputs"].items() if k != helper.NAME} == cuts
        assert len(after.state["decisions"]) == 1 and len(after.state["outputs"]) == 48
        assert after.state["tasks"] == dict(tasks, **{helper.NAME: "COMPLETE"})
        if handoff.call_count == 1:
            first = (after.tip, journal.path.read_bytes(), after.ordinal)
        else:
            assert (after.tip, journal.path.read_bytes(), after.ordinal) == first
    history = [load(path)["state"] for path in sorted(control.glob("transition-*.json"))]
    assert (
        sum(
            helper.NAME in new["outputs"] and helper.NAME not in old["outputs"]
            for old, new in itertools.pairwise(history)
        )
        == 1
    )
    assert sum(state["tasks"].get(helper.NAME) == "COMPLETE" for state in history) == 1
    assert handoff.call_count == 2
    assert not (session / "run-research/execution-0001").exists()
    assert set(load(session / "journal.json")["steps"]) == {"prepare-research/acceptance"}
    tail, _ = helper.charge_state(journal, sealed)
    assert [e["operation"] for e in tail] == [helper.KEY, helper.KEY]
    assert tail[1]["cost"] == dict(zero, elapsed_us=3)
    worker.assert_not_called()
    host.assert_not_called()


def test_next_proposal_on_separate_cpu_ledger(tmp_path):
    from oeis_learn.generalization.progressive.trials import TrialJournal
    from tests.helpers.progressive_engine import LocalProcessEngine

    request = {
        "study": digest("separate review study"),
        "ordinal": 0,
        "history": [],
        "selected_learning": None,
    }
    search = artifact(tmp_path, "search.json", "search_space", "proposal", request)
    project(
        tmp_path,
        "projected",
        "propose-trial",
        {"search": [search]},
        registry=progressive_roles.REGISTRY,
    )
    output = tmp_path / "worker"
    # The local engine starts the same worker used by the supervisor in a separate CPU process.
    command = LocalProcessEngine(tmp_path).command(
        argv=["propose-trial"],
        environment={},
        mounts=[
            (tmp_path / "projected/search", "/inputs/search", True),
            (str(output), "/output/discovery", False),
        ],
    )
    subprocess.run(command, check=True, timeout=45)
    result = load(output / "result.json")
    assert result["reason"] == "PROPOSAL_DURABLE_BEFORE_DISPATCH"
    assert result["metrics"]["proposal"] == reference(output, "proposal.json")
    saved = load(output / "proposal.json")
    assert saved["ordinal"] == 0
    journal = TrialJournal(output / "journal", study=request["study"])
    assert journal.trials["0"]["proposal"] == saved
    assert journal.trials["0"]["status"] == "PROPOSED" and journal.trials["0"]["result"] is None
