"""A disposable, one-task workstation handoff with a real local CPU census worker."""

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, load, reference, seal
from oeis_learn.generalization.progressive.freeze import freeze, learning_compatibility
from oeis_learn.generalization.qualification import FACTS
from oeis_learn.pipeline_validation import (
    generalization_qualification_io,
    progressive_cli,
    supervision,
)
from oeis_learn.pipeline_validation import progressive_workstation as workstation
from scripts import foundation_local
from tests.contract.test_progressive_roles import artifact
from tests.helpers.progressive_engine import LocalProcessEngine
from tests.unit import test_progressive_freeze as freeze_inputs
from tests.unit.test_progressive_cohorts import fixture as cohort_fixture
from tests.unit.test_progressive_streamlined import roster


def freeze_census(root, frozen, monkeypatch):
    freeze_inputs.test_complete_diagnostic_freeze_binds_original_producer_and_task_roster(
        root, monkeypatch
    )
    blueprint = load(root / "test-blueprint.json")
    prepared = load(root / "test-prepared.json")["payload"]
    runtime = load(root / "runtime.json")

    # Extend the synthetic original and rebind every reference before freezing.
    qroot = root / "qualification"
    qualification = load(qroot / "qualification.json")
    producer = load(qroot / "producer.json")
    producer["capabilities"].append("progressive-lifecycle")
    workstation.write(qroot / "producer.json", seal(producer))
    qualification["covered_capabilities"] = producer["capabilities"]
    qualification["producer"]["receipt"] = reference(qroot, "producer.json")
    workstation.write(qroot / "qualification.json", seal(qualification))
    qref = reference(root, qroot / "qualification.json")
    runtime["payload"]["qualification"] = [qref]
    workstation.write(root / "runtime.json", seal(runtime))
    prepared["runtime"] = reference(root, "runtime.json")
    refs = prepared["references"]
    next(r for r in refs if r["kind"] == "qualification")["artifact"] = qref

    cohort, exposure = cohort_fixture(root, role="development", panel="first-census")
    inputs = [
        artifact(root, kind + "-channel.json", kind, "custodian", doc).ref
        for kind, doc in (("cohort", cohort), ("exposure", exposure))
    ]
    for item in inputs:
        refs.append(
            {"name": item["path"], "role": "custodian", "kind": "prepared", "artifact": item}
        )
    plan = prepared["task_plans"][0]
    plan["caps"]["disk_bytes"] = 8 << 20
    next(n for n in prepared["graph"]["nodes"] if n["id"] == "capability")["caps"]["disk_bytes"] = (
        8 << 20
    )
    first = roster()["tasks"][0]
    assert (first["id"], first["operation"]) == ("capability-census", "census")
    request = {
        "operation": first["operation"],
        "artifacts": {"sources": [r["path"] for r in inputs]},
        "image": qualification["images"][0],
        "seconds": 20,
        "reservation": dict.fromkeys(plan["caps"], 0),
    }
    atomic_write(root / "census-request.json", request)
    policy = reference(root, "census-request.json")
    refs.append(
        {"name": "census-request", "role": "custodian", "kind": "policy", "artifact": policy}
    )
    plan.update(
        id=first["id"],
        stage=first["stage"],
        slot=first["stage"],
        seed=first["seed"],
        kind="evaluation",
        token_target=0,
        policy=policy["sha256"],
    )
    plan["initializer"]["compatibility"] = learning_compatibility(
        load(root / "recipe.json"), refs, runtime["payload"]["scope"], blueprint["track"]
    )
    campaign, runtime, tasks = freeze(
        root,
        blueprint,
        seal(dict(load(root / "test-prepared.json"), payload=prepared)),
        qualification_root="qualification",
    )
    assert [t["payload"]["id"] for t in tasks] == ["capability-census"]
    frozen_refs = {entry["name"]: entry["artifact"] for entry in campaign["payload"]["references"]}
    assert [frozen_refs[ref["path"]] for ref in inputs] == inputs
    atomic_write(frozen / "campaign.json", campaign)
    atomic_write(frozen / "runtime.json", runtime)
    return campaign, inputs, {k: qualification[k] for k in FACTS}


@pytest.mark.parametrize("case", ["accepted", "missing", "changed", "source", "image", "device"])
def test_first_workstation_dispatch(tmp_path, monkeypatch, case):
    session_dir = "run-state/fresh-fixture"
    base = tmp_path / session_dir / "prepare-research"
    campaign, inputs, facts = freeze_census(tmp_path, base / "frozen", monkeypatch)
    if case == "source":
        facts["source_closure"] = reference(tmp_path, "recipe.json")
    elif case == "image":
        facts["images"] = list(reversed(facts["images"]))
    elif case == "device":
        facts["hardware"] = dict(facts["hardware"], device="other-fixture-device")
    monkeypatch.setattr(generalization_qualification_io, "current_facts", lambda *a, **kw: facts)
    dispatches = []
    worker_commands = []

    class CensusOnly(LocalProcessEngine):
        def command(self, **kw):
            assert not dispatches and kw["argv"][-1] == "census" and kw["gpu"] is False
            sources = next(Path(src) for src, dest, _ in kw["mounts"] if dest == "/inputs/sources")
            received = sorted(
                "sha256:" + sha256(f.read_bytes()).hexdigest() for f in sources.iterdir()
            )
            assert received == sorted(ref["sha256"] for ref in inputs)
            dispatches.append(received)
            command = super().command(**kw)
            worker_commands.append(command)
            return command

    host = SimpleNamespace(engine=CensusOnly(tmp_path), policy={"cpus": 1, "memory_bytes": 1 << 30})
    real_launch = supervision.launch

    def launch(argv, *args, **kwargs):
        if len(argv) > 2 and argv[2] == str(tmp_path / "scripts/foundation_run.py"):
            parsed = foundation_local.parser().parse_args(argv[3:])
            assert (parsed.command, parsed.operation) == ("discovery", "run")
            return progressive_cli.main(parsed, tmp_path, lambda _: host)
        assert worker_commands == [argv], "unexpected process at the fixture boundary"
        worker_commands.pop()
        return real_launch(argv, *args, **kwargs)

    monkeypatch.setattr(supervision, "launch", launch)
    # Publication has its own fixture. End after the finite run and its task receipt.
    monkeypatch.setattr(workstation.Workflow, "collect_execution", lambda *args: None)

    def record_preparation(workflow):
        session = workflow.s
        frozen = base / "frozen/campaign.json"
        session.step("freeze", lambda: (campaign["identity"], [frozen]))
        receipt = base / "acceptance/acceptance.json"
        if case == "missing":
            session.step("acceptance", lambda: None, dependencies=("prepare-research/absent",))
        else:
            atomic_write(
                receipt,
                seal(
                    {
                        "version": "discovery-acceptance/v1",
                        "scope": "progressive-lifecycle",
                        "qualified": True,
                        "status": "COMPLETE",
                        "campaign": campaign["identity"],
                        "runtime": campaign["payload"]["runtime"],
                    }
                ),
            )
            session.step(
                "acceptance",
                lambda: ("diagnostic fixture", [receipt]),
                dependencies=("prepare-research/freeze",),
            )
        if case == "changed":
            workstation.write(receipt, {"changed": True})

    monkeypatch.setattr(workstation.Workflow, "prepare_research", record_preparation)
    monkeypatch.setattr(workstation, "telemetry", lambda _: {"fixture": True})
    code = workstation.main(
        tmp_path,
        [
            "--engine",
            "docker",
            "--session",
            session_dir,
            "--reference-campaign",
            "synthetic",
            "--run-after-prepare",
            "prepare-research",
        ],
    )
    assert code == {"accepted": 0, "missing": 2}.get(case, 1)
    execution = tmp_path / session_dir / "run-research/execution-0000"
    assert not worker_commands
    assert len(dispatches) == (1 if case == "accepted" else 0)
    worker_result = None
    if case == "accepted":
        [result] = execution.glob("task-*/role/result.json")
        assert load(result)["reason"] == "OUTCOME_BLIND_CENSUS"
        assert set(load(execution / "result.json")["metrics"]["tasks"]) == {"capability-census"}
        worker_result = reference(tmp_path, result)
    elif case in {"source", "image", "device"}:
        assert "qualification authority differs" in (execution / "result.json").read_text()
    else:
        assert not execution.exists()
        if case == "changed":
            assert any(
                "changed" in error["message"]
                for error in load(base.parent / "journal.json")["errors"]
            )
    atomic_write(
        tmp_path / "dispatch-observation.json",
        {
            "case": case,
            "task": "capability-census",
            "campaign": campaign["identity"],
            "frozen_inputs": inputs,
            "projected_hashes": dispatches,
            "dispatch_count": len(dispatches),
            "worker_receipt": worker_result,
        },
    )
