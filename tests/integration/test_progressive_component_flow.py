"""Newly mined owned outputs cross the frozen component task graph."""

import copy
import itertools
import time
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    record,
    reference,
    seal,
)


def fixture(tmp_path, fitting_client):
    from oeis_learn.generalization.progressive.component_distillation import RULE
    from oeis_learn.generalization.progressive.component_mining import policy
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    from oeis_learn.pipeline_validation.progressive_learner import compatibility
    from tests.unit.test_progressive_component_mining import sources

    c, _, _, _ = fixture(tmp_path / "base", interrupt=False)
    package = load(tmp_path / "base/training-input.json")
    package = seal(dict(package, component_rules=RULE))
    atomic_write(tmp_path / "package.json", package)
    scope = compatibility(package)

    def channel(name, kind, role, data):
        atomic_write(
            tmp_path / name,
            seal({"version": "discovery-channel/v1", "kind": kind, "role": role, "data": data}),
        )
        return name

    training = channel(
        "training.json",
        "training_request",
        "learner",
        {
            "membership": "training",
            "artifact": reference(tmp_path, "package.json"),
            "profile": "bigint-body/v1",
            "compatibility": scope,
        },
    )
    source_refs = sources(tmp_path, fitting_client)
    archives = [
        channel(
            "origin-" + str(i) + ".json",
            "typed_training_source",
            "component",
            {
                "membership": "training",
                "artifact": r,
                "profile": "bigint-body/v1",
                "compatibility": scope,
            },
        )
        for i, r in enumerate(source_refs)
    ]
    archives.append(
        channel(
            "mining.json",
            "component_policy",
            "component",
            {"policy": policy(maximum_work=8192), "previous": None},
        )
    )
    atomic_write(tmp_path / "profile.json", package["profile"])
    atomic_write(tmp_path / "random.json", {})
    model = [
        channel(
            "component-recipe-channel.json",
            "recipe",
            "generator",
            {
                "artifact": reference(tmp_path, "profile.json"),
                "profile": "bigint-body/v1",
                "compatibility": scope,
            },
        ),
        channel(
            "component-checkpoint-channel.json",
            "checkpoint",
            "generator",
            {
                "artifact": reference(tmp_path, "random.json"),
                "profile": "bigint-body/v1",
                "compatibility": scope,
                "control": "random",
            },
        ),
    ]
    member = {
        "id": "training-new",
        "group": "training-group",
        "role": "training",
        "source_snapshot": digest("diagnostic"),
        "indices": list(map(str, range(100))),
        "values": [str(2 * i + 1) for i in range(100)],
    }
    prompt = channel(
        "prompt.json",
        "visible_prompt",
        "generator",
        {
            k: (member[k][:20] if k in {"indices", "values"} else member[k])
            for k in ("id", "indices", "values")
        },
    )
    truth = channel("truth.json", "training_continuation", "admission", member)
    scoretruth = channel(
        "score-truth.json",
        "confirmation_truth",
        "scorer",
        {k: member[k] for k in ("id", "indices", "values")},
    )

    def slot(task, op, out, kind, **extra):
        return dict(task=task, operation=op, output=out, kind=kind, **extra)

    library = slot("mine", "extract-components", "library", "library")
    requests = [
        ("mine", "extract-components", {"archive": archives}),
        ("generate", "generate", {"visible": [prompt], "model": [*model, library]}),
        (
            "admit",
            "admit-training",
            {
                "candidates": [
                    slot(
                        "generate",
                        "generate",
                        "component_proposals",
                        "component_proposal",
                        member=member["id"],
                    )
                ],
                "admission_truth": [truth, library],
            },
        ),
        (
            "train",
            "train",
            {
                "training": [
                    training,
                    library,
                    slot("admit", "admit-training", "target", "admitted_target"),
                ]
            },
        ),
        (
            "score",
            "score",
            {
                "candidates": [slot("generate", "generate", "candidates", "candidate")],
                "score_truth": [scoretruth],
            },
        ),
    ]
    plans = []
    refs = []
    for name, op, artifacts in requests:
        request = {
            "operation": op,
            "artifacts": artifacts,
            "image": digest("cpu-fixture"),
            "seconds": 180,
            "reservation": {k: 0 for k in c["payload"]["caps"]},
        }
        path = name + "-request.json"
        atomic_write(tmp_path / path, request)
        plan = copy.deepcopy(c["payload"]["tasks"][0])
        plan.update(
            id=name,
            slot="components",
            stage="components",
            kind="training" if op == "train" else "evaluation",
            arm="library",
            token_target=1,
            output=name,
            policy=reference(tmp_path, path)["sha256"],
        )
        plan["initializer"]["compatibility"] = scope
        plan["caps"].update(
            tokens=100000,
            draws=10000,
            attempts=100000,
            rejections=100000,
            native_work=10000000,
            elapsed_us=180000000,
            disk_bytes=1 << 30,
            archive_bytes=128 << 20,
        )
        plans.append(plan)
        refs.append(
            {
                "name": path,
                "kind": "policy",
                "role": "custodian",
                "artifact": reference(tmp_path, path),
            }
        )
    c["payload"]["tasks"] = plans
    c["payload"]["references"] = refs
    zero = {k: 0 for k in plans[0]["caps"]}
    stages = ["capability", "diversity", "tuning", "scaling", "components"]
    c["payload"]["graph"]["selected_stages"] = stages
    c["payload"]["graph"]["nodes"] = [
        {
            "id": stage,
            "stage": stage,
            "kind": "routine",
            "dependencies": [] if i == 0 else [stages[i - 1]],
            "maximum_visits": 1,
            "caps": {k: sum(p["caps"][k] for p in plans) for k in zero}
            if stage == "components"
            else zero,
        }
        for i, stage in enumerate(stages)
    ]
    c["payload"]["graph"]["edges"] = [
        {
            "source": a,
            "target": b,
            "condition": "CONFIRMED_GAIN" if b == "scaling" else "COMPLETE",
            "action": "ADVANCE_REGISTERED",
        }
        for a, b in itertools.pairwise(stages)
    ]
    c["payload"]["caps"] = {k: 2 * sum(p["caps"][k] for p in plans) for k in plans[0]["caps"]}
    # This is a diagnostic capability acceptance, covering the component roles.
    for k in ("task_store_bytes", "projection_bytes", "journal_bytes", "checkpoint_inodes"):
        c["payload"]["storage"].pop(k, None)
    s = c["payload"]["storage"]
    s["required_bytes"] = s["checkpoint_bytes"] * 2 + s["shard_bytes"] + s["package_bytes"]
    s["required_inodes"] = 32768
    s["available_inodes"] = 65536
    for _, _, artifacts in requests:
        for paths in artifacts.values():
            for path in paths:
                if isinstance(path, str) and not any(r["artifact"]["path"] == path for r in refs):
                    value = load(tmp_path / path)
                    refs.append(
                        {
                            "name": "channel-" + path,
                            "kind": "prepared",
                            "role": value["role"],
                            "artifact": reference(tmp_path, path),
                        }
                    )
    c = seal(c)
    return c, requests, package


def test_installed_component_outputs_feed_admission_and_learning_without_future_library(
    tmp_path, fitting_client
):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, role_work_requirements
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.generalization.progressive.projections import project
    from oeis_learn.pipeline_validation.progressive_outputs import (
        commit,
        resolve,
        validate_requests,
    )
    from oeis_learn.pipeline_validation.progressive_role import execute
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY

    c, requests, package = fixture(tmp_path, fitting_client)
    from oeis_learn.pipeline_validation.progressive_learner import compatibility

    compatibility(package)
    plans = c["payload"]["tasks"]
    refs = c["payload"]["references"]
    validate_requests(tmp_path, plans, refs, c["payload"]["graph"])
    store = RunStore(tmp_path / "control", c)
    j = BudgetJournal(tmp_path / "journal", c["identity"], c["payload"]["caps"])
    for name, op, _ in requests:
        p = next(p for p in plans if p["id"] == name)
        task = record(
            "task",
            dict(p, campaign=c["identity"]),
            created_at=c["created_at"],
            purpose=c["purpose"],
            track=c["track"],
        )
        request = load(tmp_path / (name + "-request.json"))
        artifacts, _ = resolve(tmp_path, store, task, request, tmp_path / (name + "-resolved"))
        target = tmp_path / name
        project(
            tmp_path,
            target / "inputs",
            op,
            artifacts,
            registry=REGISTRY,
            authorized=[a.ref for v in artifacts.values() for a in v],
        )
        values = [load(tmp_path / a.ref["path"]) for v in artifacts.values() for a in v]
        cost = role_work_requirements(op, values) if op != "train" else {k: 0 for k in p["caps"]}
        cost["elapsed_us"] = 180000000
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        start = time.monotonic_ns() // 1000
        event = j.reserve(name, op, 0, cost, boot=boot, monotonic_us=start)
        if op == "train":
            authority = seal(
                {
                    "version": "progressive-supervisor/v1",
                    "campaign": c,
                    "task": task,
                    "clock": {
                        "reservation_key": event["key"],
                        "boot": boot,
                        "start_us": start,
                        "end_us": start + cost["elapsed_us"],
                    },
                    "invocation": name,
                }
            )
        else:
            authority = seal(
                {
                    "version": "progressive-role-budget/v1",
                    "campaign": c,
                    "task": task,
                    "operation": op,
                    "inputs": digest(sorted(values, key=lambda v: v["identity"])),
                    "reservation": j._read()[0][-1],
                }
            )
        atomic_write(target / "authority.json", authority)
        result = execute(
            op,
            inputs=target / "inputs",
            output=target / "role",
            authority=target / "authority.json",
            accounting=j.root,
            control=tmp_path / "control",
            component_state=tmp_path / "component-state",
            state_root=tmp_path / "state",
        )
        assert result["mechanical_status"] == "COMPLETE", result
        j.commit(
            event["key"],
            boot=boot,
            monotonic_us=time.monotonic_ns() // 1000,
            termination_proven=True,
        )
        commit(tmp_path, store, task, target / "role", op)
        store.boundary(task, "COMPLETE", "diagnostic")
    result = load(tmp_path / "train/role/result.json")
    assert result["metrics"]["useful_tokens"] > 0
    assert load(tmp_path / "train/role/checkpoint.json")["payload"]["cuts"]["library"] != digest(
        None
    )


def test_canonical_freeze_component_slots_then_resume_without_remining(
    tmp_path, monkeypatch, fitting_client, capsys
):
    from types import SimpleNamespace

    from oeis_learn.pipeline_validation import progressive_roles as roles
    from oeis_learn.pipeline_validation.progressive_commands import main
    from oeis_learn.pipeline_validation.progressive_oeis import Cycle
    from tests.helpers.progressive_host_cycle import accept_prior_stages, freeze_inputs

    prefix = freeze_inputs(
        tmp_path, monkeypatch, build=lambda root: fixture(root, fitting_client)[0]
    )
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
    calls = []

    def invoke(
        host,
        image,
        root,
        operation,
        artifacts,
        *,
        output,
        campaign,
        task,
        journal,
        ordinal,
        seconds,
        **kwargs,
    ):
        cycle = Cycle.__new__(Cycle)
        cycle.root = root
        cycle.campaign = campaign
        cycle.journal = journal
        cycle.ordinal = ordinal
        cycle.pythonpath = None
        calls.append(operation)
        return cycle.cpu_phase(operation, artifacts, root / output, task, seconds)

    monkeypatch.setattr(roles, "invoke", invoke)
    assert (
        main(
            SimpleNamespace(
                operation="run",
                campaign="frozen-cycle/campaign.json",
                acceptance="accepted",
                output="component-run",
            ),
            tmp_path,
            lambda a: object(),
        )
        == 0
    )
    assert calls == ["extract-components", "generate", "admit-training", "train", "score"]
    before = list(calls)
    assert (
        main(
            SimpleNamespace(operation="resume", run="component-run", output="component-resume"),
            tmp_path,
            lambda a: object(),
        )
        == 0
    )
    assert calls == before
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.generalization.progressive.recovery import Recovery

    store = RunStore(tmp_path / "run-state/discovery-control" / c["identity"][7:], c)
    plan = next(p for p in c["payload"]["tasks"] if p["id"] == "train")
    task = record(
        "task",
        dict(plan, campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    j = BudgetJournal(
        tmp_path / "run-state/discovery-accounting" / c["identity"][7:],
        c["identity"],
        c["payload"]["caps"],
    )
    with Recovery(
        j,
        task,
        plan["initializer"]["compatibility"],
        state_root=tmp_path
        / "run-state/discovery-learning"
        / c["identity"][7:]
        / task["identity"][7:],
    ) as owner:
        assert owner.restore()["cuts"]["library"] != digest(None)
    from oeis_learn.pipeline_validation.progressive_outputs import resolve

    request = load(tmp_path / "train-request.json")
    artifacts, _ = resolve(tmp_path, store, task, request, tmp_path / "clean-resolved")
    from oeis_learn.pipeline_validation.progressive_quarantine_control import require_clean_inputs

    require_clean_inputs(tmp_path, j, artifacts)
    from oeis_learn.generalization.progressive.quarantine import Journal

    q = Journal(j.root)
    library = next(n["id"] for n in q.read()[0] if n["kind"] == "library")
    q.refute([library], reason="diagnostic independent refuted-library overlay")
    import pytest

    with pytest.raises(ValueError, match="quarantined"):
        require_clean_inputs(tmp_path, j, artifacts)
    original = store.state["outputs"]["mine"]
    store.state["outputs"]["mine"] = dict(original, task=digest("foreign-task"))
    with pytest.raises(ValueError):
        resolve(tmp_path, store, task, request, tmp_path / "foreign-resolved")
