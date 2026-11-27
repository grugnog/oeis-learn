"""Bounded CPU recovery and next-proposal checks. All state lives under tmp_path."""

import io
import itertools
import json
import shutil
import subprocess
import sys
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


def publication_request(root, campaign, gate, template):
    """Freeze the eight typed producer slots and the comparison template in the CPU campaign."""
    producers = []
    public = ["comparison-template.json"]
    for p in gate["producers"]:
        name = f"{p['arm']}-{p['seed']}"
        producers.append(
            {
                "arm": p["arm"],
                "seed": p["seed"],
                "generation_task": name + "-generate",
                "scoring_task": name + "-score",
            }
        )
        for op, output in (("generate", "evaluation"), ("score", "scores")):
            public.append(
                {
                    "task": name + "-" + op,
                    "operation": op,
                    "output": output,
                    "kind": "public_evidence",
                }
            )
    channel = artifact(
        root,
        public[0],
        "public_evidence",
        "verifier",
        dict(template, version="progressive-decision-template/v1", producers=producers),
    )
    atomic_write(
        root / "publish-request.json",
        {
            "operation": "publish",
            "reservation": dict.fromkeys(KEYS, 0) | {"native_work": 1_000_000},
            "artifacts": {"public": public},
            "seconds": 240,
        },
    )
    policy = reference(root, "publish-request.json")
    campaign["payload"]["references"].extend(
        [
            {
                "name": "comparison-template",
                "role": "verifier",
                "kind": "policy",
                "artifact": channel.ref,
            },
            {
                "name": "requests/diversity-comparison",
                "role": "custodian",
                "kind": "policy",
                "artifact": policy,
            },
        ]
    )
    return policy["sha256"]


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
    graph["edges"][0]["condition"] = "COMPLETE"
    fields = {
        k: data[k] for k in ("source_slot", "target_slot", "rows", "roster", "seeds", "design")
    }
    kwargs = {"target_id": "gain", "action": "RETAIN_BASELINE"}
    template = {k: v for k, v in fields.items() if k != "rows"}
    compare["policy"] = publication_request(
        root, campaign, gate, dict(template, policy=policy, kwargs=kwargs, remaining=remaining)
    )
    # Fit the actual historical reservation and supervisor overhead within this fixture's partitions.
    compare["caps"] = dict(compare["caps"], elapsed_us=300_000_000, native_work=1_000_000)
    graph["nodes"][0]["caps"] = {k: sum(p["caps"][k] for p in [*previous, compare]) for k in KEYS}
    campaign["payload"]["caps"]["native_work"] = sum(
        n["caps"]["native_work"] for n in graph["nodes"]
    )
    plans = campaign["payload"]["tasks"]
    storage = campaign["payload"]["storage"]
    storage.update(
        task_store_bytes=sum(p["caps"]["disk_bytes"] for p in plans),
        projection_bytes=1 << 20,
        journal_bytes=128 << 20,
        checkpoint_inodes=64,
        available_inodes=65536,
    )
    storage["required_inodes"] = 16384 + len(plans) * (
        256 + storage["checkpoint_inodes"] * (2 + storage["milestone_pins"])
    )
    storage["required_bytes"] = sum(
        storage[k]
        for k in (
            "task_store_bytes",
            "projection_bytes",
            "journal_bytes",
            "shard_bytes",
            "package_bytes",
        )
    )
    storage["available_bytes"] = storage["required_bytes"]
    campaign = seal(campaign)
    indexed = {(r["arm"], r["seed"], r["task"]): r for r in data["rows"]}
    ordered = [
        indexed[p["arm"], p["seed"], member["task"]]
        for p in gate["producers"]
        for member in fields["roster"]
    ]
    fields["rows"] = [
        dict(r, complete=r.get("complete", r["outcome"] != "unresolved")) for r in ordered
    ]
    decision = publish(
        fields["rows"],
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
                key = "evaluation" if op == "generate" else "scores"
                metrics = {key: reference(output, "evidence.json")}
                assert metrics[key]["sha256"] == ref["sha256"]
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

    # The pinned host scripts prepend their checkout paths during import.
    monkeypatch.setattr(sys, "path", list(sys.path))
    audit = adapter("replay-round027-comparison")
    helper = adapter("continue-round027-publication")
    import foundation_local

    campaign, kinds, gate, task, bundle, table, models = comparison(tmp_path)
    cid = campaign["identity"]
    session = tmp_path / helper.SESSION
    control = tmp_path / "run-state/discovery-control" / cid[7:]
    account = tmp_path / "run-state/discovery-accounting" / cid[7:]
    store = RunStore(control, campaign)
    journal = BudgetJournal(
        account,
        cid,
        campaign["payload"]["caps"],
        journal_bytes=campaign["payload"]["storage"]["journal_bytes"],
    )
    atomic_write(session / "prepare-research/frozen/campaign.json", campaign)
    cuts = completed_outputs(tmp_path, session, campaign, kinds, gate, models)
    tasks = dict.fromkeys(cuts, "COMPLETE")
    tasks[helper.NAME] = "FAILED"
    with store.lease():
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
        dict(zero, elapsed_us=5, native_work=1_000_000),
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
    helper.POLICY = task["payload"]["policy"]
    accepted = session / "prepare-research/acceptance"
    receipt = seal(
        {
            "version": "discovery-acceptance/v1",
            "campaign": cid,
            "runtime": campaign["payload"]["runtime"],
            "status": "COMPLETE",
            "qualified": True,
            "scope": "progressive-lifecycle",
            "fixture": "CPU review",
        }
    )
    atomic_write(accepted / "acceptance.json", receipt)
    atomic_write(
        session / "run-research/execution-0000/run.json",
        seal(
            {
                "campaign": cid,
                "campaign_path": str(
                    (session / "prepare-research/frozen/campaign.json").relative_to(tmp_path)
                ),
                "acceptance_path": str(accepted.relative_to(tmp_path)),
            }
        ),
    )

    # Only the private hardware qualification envelope is replaced by the fixed CPU fixture.
    def accept_fixture(path, value):
        assert path == accepted and value == receipt

    monkeypatch.setattr(helper, "consume_receipt", accept_fixture)
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
    # corrected_table reads the reviewed adapter with git show from the disposable root.
    git_dir = subprocess.check_output(
        ["git", "rev-parse", "--absolute-git-dir"], cwd=checkout, text=True
    ).strip()
    monkeypatch.setenv("GIT_DIR", git_dir)
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
    _, amendment = helper.charge_state(journal, sealed)
    journal.reserve(
        helper.NAME, "publish", helper.ORDINAL, amendment, boot="fixture", monotonic_us=0
    )

    class HandoffReached(Exception):
        pass

    def stop(root, argv):
        assert root == tmp_path and argv[-1] == "run-research"
        assert RunStore(control, campaign).next_task()["payload"]["id"] == "trial-00-propose"
        raise HandoffReached

    handoff = Mock(side_effect=stop)
    worker = Mock(side_effect=AssertionError("unexpected worker"))
    host = Mock(side_effect=AssertionError("unexpected engine"))
    monkeypatch.setattr(progressive_workstation, "main", handoff)
    monkeypatch.setattr(progressive_roles, "invoke", worker)
    monkeypatch.setattr(foundation_local, "Host", host)

    # Failed prerequisites on the same stopped cut must leave its transition and charges alone.
    initial = (store.tip, journal.path.read_bytes())
    with store.lease(), pytest.raises(ValueError, match="owns"):
        helper.continue_session(session)
    producer = tmp_path / next(iter(cuts.values()))["result"]["path"]
    sidecar = next(ref for ref in checkpoints[0]["files"] if "optimizer" in ref["path"])
    for path, reason in (
        (producer.parent / "evidence.json", "original artifact identity changed"),
        (tmp_path / sidecar["path"], "original artifact identity changed"),
        (target / "supervision/cleanup.json", "reserved but has no complete receipts"),
    ):
        data = path.read_bytes()
        if path.name == "cleanup.json":
            path.unlink()  # Reservation survived without its supervised completion.
        else:
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with pytest.raises(ValueError, match=reason):
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
    assert [(e["event"], e["operation"]) for e in tail] == [
        ("RESERVE", helper.KEY),
        ("COMMIT", helper.KEY),
    ]
    elapsed_only = dict(zero, elapsed_us=240_000_000)
    assert tail[0]["cost"] == tail[1]["cost"] == elapsed_only
    assert tail[1]["end_us"] is None
    assert journal.snapshot()["consumed"] == {
        k: v + elapsed_only[k] for k, v in sealed["budget"]["consumed"].items()
    }
    assert load(target / "role/gain-cost.json")["costs"][helper.NAME] == 240_000_005
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


def test_locked_assessment_cli(tmp_path, monkeypatch, capsys):
    from oeis_learn.pipeline_validation import progressive_cli
    from scripts.foundation_local import parser
    from tests.contract.test_progressive_assessment import fixture as locked_selection

    campaign, selection, truth = locked_selection(tmp_path)
    atomic_write(tmp_path / "campaign.json", campaign)
    atomic_write(tmp_path / "locked.json", selection)
    atomic_write(tmp_path / "qual/producer.json", seal({"version": "diagnostic"}))
    atomic_write(
        tmp_path / "runtime.json",
        {
            "payload": {
                "qualification": [reference(tmp_path, "qual/producer.json")],
                "scope": {"images": ["fixture"]},
            }
        },
    )
    receipt = {
        "version": "discovery-acceptance/v1",
        "scope": "progressive-lifecycle",
        "campaign": campaign["identity"],
        "runtime": campaign["payload"]["runtime"],
        "qualified": True,
        "status": "COMPLETE",
    }
    atomic_write(tmp_path / "accepted/acceptance.json", seal(receipt))
    atomic_write(
        tmp_path / "foreign/acceptance.json", seal(dict(receipt, campaign=digest("other")))
    )
    changed = dict(selection["checkpoint"], sha256=digest("substitution"))
    atomic_write(tmp_path / "changed.json", seal(dict(selection, checkpoint=changed)))
    control = tmp_path / "run-state/discovery-control" / campaign["identity"][7:]
    exposure = control / "assessment-exposure.json"

    def native_boundary(host, image, root, operation, channels, **kwargs):
        assert operation == "assess"
        assert set(channels) == {"candidates", "selection", "assessment_truth"}
        assert channels["candidates"][0].ref == selection["candidates"]
        assert channels["assessment_truth"][0].ref == truth.ref
        assert load(root / channels["selection"][0].ref["path"])["data"] == selection
        project(root, "native-inputs", operation, channels, registry=progressive_roles.REGISTRY)
        assert exposure.is_file()
        return seal(
            {"mechanical_status": "PARTIAL", "reason": "fixture stopped before native scoring"}
        )

    native = Mock(side_effect=native_boundary)
    monkeypatch.setattr(progressive_roles, "invoke", native)
    monkeypatch.setattr(progressive_roles, "current_runtime", lambda *args, **kwargs: object())

    def run(output, *, selected="locked.json", accepted="accepted"):
        args = parser().parse_args(
            [
                "discovery",
                "assess",
                "--campaign",
                "campaign.json",
                "--selection",
                selected,
                "--acceptance",
                accepted,
                "--output",
                output,
            ]
        )
        code = progressive_cli.main(args, tmp_path, lambda args: object())
        return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    for options, message in (
        ({"accepted": "foreign"}, "compatible final assessment acceptance"),
        ({"selected": "changed.json"}, "artifact hash/size mismatch"),
    ):
        code, result = run("rejected", **options)
        assert code != 0 and message in result["producer_error"]["message"]
        assert not exposure.exists()
        native.assert_not_called()
    code, result = run("assessment")
    assert code == 2 and result["mechanical_status"] == "PARTIAL"
    original = exposure.read_bytes()
    code, result = run("again")
    assert code != 0 and "already exposed" in result["producer_error"]["message"]
    native.assert_called_once()
    assert exposure.read_bytes() == original
    with pytest.raises(ValueError, match="assessment exposed"):
        RunStore(control, campaign).next_task()
