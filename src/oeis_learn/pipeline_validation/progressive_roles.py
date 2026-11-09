"""Explicit discovery operation registry and supervised installed-role adapter."""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from oeis_learn.generalization.progressive.projections import project, remove_projection
from oeis_learn.pipeline_validation.identity import confined, digest


@dataclass(frozen=True)
class Operation:
    role: str
    channels: tuple[str, ...]
    gpu: bool = False


REGISTRY = MappingProxyType(
    {
        "census": Operation("custodian", ("sources",)),
        "prepare-stream": Operation("custodian", ("sources",)),
        "audit-language": Operation("custodian", ("sources",)),
        "propose-trial": Operation("proposal", ("search",)),
        "train": Operation("learner", ("training",), True),
        "generate": Operation("generator", ("visible", "model"), True),
        "admit-training": Operation("admission", ("candidates", "admission_truth")),
        "score": Operation("scorer", ("candidates", "score_truth")),
        "assess": Operation("assessment", ("candidates", "assessment_truth", "selection")),
        "extract-components": Operation("component", ("archive",)),
        "verify-components": Operation("component", ("archive",)),
        "source-pilot": Operation("custodian", ("sources",)),
        "review-claim": Operation("review", ("dossier",)),
        "publish": Operation("verifier", ("public",)),
        "verify": Operation("verifier", ("public",)),
    }
)
ROLE_POLICY = digest(
    {
        name: {"role": v.role, "channels": list(v.channels), "gpu": v.gpu}
        for name, v in REGISTRY.items()
    }
)
IMPLEMENTED = frozenset(
    {
        "source-pilot",
        "verify",
        "census",
        "audit-language",
        "publish",
        "generate",
        "score",
        "prepare-stream",
        "propose-trial",
        "admit-training",
        "train",
        "extract-components",
        "verify-components",
        "review-claim",
        "assess",
    }
)


def deadline_for(task, *, remaining_us, seconds, runtime=None, operation):
    """Only validated discovery authority can extend the generic six-hour cap."""
    from oeis_learn.generalization.budgets import integer
    from oeis_learn.generalization.progressive.campaign import VerifiedRuntime, validate_task

    t = validate_task(task)
    p = t["payload"]
    integer(remaining_us, 1)
    if type(seconds) not in (int, float) or not 0 < seconds <= 86400:
        raise ValueError("bounded discovery role duration")
    if int(seconds * 1_000_000) > remaining_us or remaining_us > p["caps"]["elapsed_us"]:
        raise ValueError("enclosing remaining deadline exceeded")
    if remaining_us <= p["startup_us"] + p["finalization_us"]:
        raise ValueError("protected startup/finalization exhausted")
    if seconds > 21600 and (
        not isinstance(runtime, VerifiedRuntime)
        or not runtime.execution_ready
        or operation not in runtime.payload["operations"]
        or runtime.purpose != t["purpose"]
        or runtime.track != t["track"]
        or not runtime.producers
    ):
        raise ValueError("verified discovery task/runtime required beyond six hours")
    return time.monotonic() + seconds


def _invoke_reserved(
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
    reservation,
    seconds,
    runtime=None,
    settings=None,
    forbidden=(),
    derived=None,
):
    if operation not in IMPLEMENTED:
        raise NotImplementedError("discovery operation not implemented: " + operation)
    if settings:
        raise ValueError("no arbitrary role settings accepted")
    if runtime is None or not runtime.execution_ready:
        raise ValueError("freshly inspected original qualified runtime required")
    from oeis_learn.generalization.progressive.budgets import vector
    from oeis_learn.generalization.progressive.campaign import validate_campaign, validate_task

    c = validate_campaign(campaign)
    t = validate_task(task, campaign=c)
    exposure_path = Path(root)/"run-state/discovery-control"/c["identity"].split(":")[1]/"assessment-exposure.json"
    if operation != "assess" and exposure_path.exists():
        raise ValueError("assessment exposed; adaptive operations forbidden")
    if (
        (not _trial_runtime_matches(root,c,t,runtime) if 'trial_runtime' in t['payload'] else c["payload"]["runtime"] != runtime.identity)
        or runtime.track != t["track"]
        or runtime.purpose != t["purpose"]
    ):
        raise ValueError("campaign/runtime/task mismatch")
    if (
        journal.binding["campaign"] != c["identity"]
        or journal.binding["caps"] != c["payload"]["caps"]
    ):
        raise ValueError("unrelated campaign journal")
    if operation == "train" and any(reservation[k] for k in reservation if k != "elapsed_us"):
        raise ValueError(
            "learner reserves updates separately; enclosing role may reserve elapsed only"
        )
    remaining = journal.task_remaining(t)
    vector(reservation)
    if reservation["elapsed_us"] != int(seconds * 1_000_000) or any(
        reservation[k] > remaining[k] for k in remaining
    ):
        raise ValueError("unreserved role work")
    if (
        operation not in runtime.payload["operations"]
        or image not in runtime.payload["scope"]["images"]
    ):
        raise ValueError("unqualified operation/image")
    deadline = deadline_for(
        task,
        remaining_us=remaining["elapsed_us"],
        seconds=seconds,
        runtime=runtime,
        operation=operation,
    )
    from oeis_learn.generalization.progressive.budgets import WORK_ROLES, role_work_requirements
    from oeis_learn.generalization.progressive.contracts import (
        load,
        verify_refs,
        digest,
        atomic_write,
        seal,
    )

    work_values = []
    if operation in WORK_ROLES:
        for items in artifacts.values():
            for item in items:
                verify_refs(root, [item.ref])
                work_values.append(load(confined(root, item.ref["path"]), maximum=512 << 20))
        need = role_work_requirements(operation, work_values)
        if any(reservation[k] < need[k] for k in need):
            raise ValueError("insufficient enclosing role work reservation")
    root = Path(root)
    target = confined(root, output, exists=False)
    if target.exists():
        raise ValueError("fresh role output required")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    started_us = time.monotonic_ns() // 1000
    charge = journal.reserve(
        t["payload"]["id"], operation, ordinal, reservation, boot=boot, monotonic_us=started_us
    )
    if charge.get("committed") or "reserved" in charge:
        raise ValueError("existing operation requires reconciliation, never duplicate dispatch")
    from oeis_learn.generalization.progressive.storage import (
        register_invocation,
        owned_guard,
        usage,
    )

    storage_proof = register_invocation(root, c, t, journal.root, target)
    own_state = (
        root
        / (
            "run-state/discovery-components"
            if operation in {"extract-components", "verify-components"}
            else "run-state/discovery-learning"
        )
        / c["identity"].split(":")[1]
        / t["identity"].split(":")[1]
    )
    projection_guard = owned_guard(
        dict(campaign=c, task=t, storage=storage_proof), output=target, state_root=own_state
    )
    import json

    metadata_bytes = len(json.dumps(dict(campaign=c, task=t), indent=2).encode()) + (64 << 10)
    projection_guard.reserve_bytes += (2 << 20) + metadata_bytes
    projection_guard.check(0, 64)
    target.mkdir(parents=True)
    cleanup_seconds=min(30,seconds/5)
    work_end_us=min(started_us+reservation['elapsed_us'],int(deadline*1000000))-int(cleanup_seconds*1000000)
    spec = REGISTRY[operation]
    views = target / "private-input"
    authorized = [r["artifact"] for r in c["payload"]["references"] if r["role"] == spec.role]
    if derived is not None:
        if operation == "assess":
            from oeis_learn.generalization.progressive.assessment import check_derived
        else:
            from .progressive_oeis import check_derived

        authorized.extend(check_derived(derived, campaign=c, task=t, operation=operation, **({"root":root} if operation=="assess" else {})))
    with projection_guard.activate():
        channels = project(
            root,
            views,
            operation,
            artifacts,
            registry=REGISTRY,
            forbidden=forbidden,
            authorized=authorized,
        )
    projected_bytes, projected_inodes = usage([target])
    storage_proof["prior_bytes"] += projected_bytes + (2 << 20) + metadata_bytes
    storage_proof["prior_inodes"] += projected_inodes + 64
    from oeis_learn.container_runtime import private_runtime_environment

    from .generalization_roles import supervise_owned
    from .supervision import diagnostic_program

    name = "discovery-" + uuid.uuid4().hex
    role_output = target / "role"
    role_output.mkdir()
    mounted = [(str(path), "/inputs/" + channel, False) for channel, path in channels.items()] + [
        (str(role_output), "/output/discovery", True)
    ]
    gpu = spec.gpu and t["purpose"] != "acceptance"
    if operation == "train":
        from oeis_learn.generalization.progressive.contracts import atomic_write, seal

        authority_dir = target / "authority"
        authority_dir.mkdir()
        invocation = uuid.uuid4().hex
        # Host campaign invocation identifies exactly one pause namespace.
        from .progressive_commands import _INVOCATION

        invocation = _INVOCATION.get() or invocation
        authority = seal(
            {
                "version": "progressive-supervisor/v1",
                "campaign": c,
                "task": t,
                "clock": {
                    "reservation_key": charge["key"],
                    "boot": boot,
                    "start_us": started_us,
                    "end_us": started_us + reservation["elapsed_us"],
                    "work_end_us":work_end_us,
                },
                "invocation": invocation,
                "storage": storage_proof,
            }
        )
        atomic_write(authority_dir / "supervisor.json", authority)
        control = Path(root) / "run-state" / "discovery-control" / c["identity"].split(":")[1]
        control.mkdir(parents=True, exist_ok=True)
        states = Path(root) / "run-state" / "discovery-learning" / c["identity"].split(":")[1]
        state = states / t["identity"].split(":")[1]
        state.mkdir(parents=True, exist_ok=True)
        mounted.append((str(state), "/learning-state", True))
        parent_slot = t["payload"]["initializer"].get("parent_slot")
        parent_task = t["payload"]["initializer"].get("parent_task")
        if parent_slot or parent_task:
            from oeis_learn.generalization.progressive.contracts import record

            plan = next(
                p
                for p in c["payload"]["tasks"]
                if (p["id"] == parent_task if parent_task else p["slot"] == parent_slot)
                and p["arm"] == t["payload"]["arm"]
                and p["seed"] == t["payload"]["seed"]
            )
            if 'trial_choices' in plan:
                from oeis_learn.generalization.progressive.trial_scopes import resolve
                plan=resolve(plan,t['payload']['trial_choice'])
            parent = record(
                "task",
                dict(plan, campaign=c["identity"]),
                created_at=c["created_at"],
                purpose=c["purpose"],
                track=c["track"],
            )
            parent_state = confined(states, parent["identity"].split(":")[1])
            mounted.append((str(parent_state), "/predecessor-state", False))
        mounted += [
            (str(authority_dir), "/authority", False),
            (str(journal.root), "/accounting", True),
            (str(control), "/control", False),
        ]
    if operation in {"extract-components", "verify-components"}:
        own_state.mkdir(parents=True, exist_ok=True)
        mounted.append((str(own_state), "/component-state", True))
    if operation in WORK_ROLES:
        authority_dir = target / "authority"
        authority_dir.mkdir()
        with journal.lock():
            reserved = next(
                e
                for e in journal._read()[0]
                if e["operation"] == charge["key"] and e["event"] == "RESERVE"
            )
        atomic_write(
            authority_dir / "supervisor.json",
            seal(
                dict(
                    version="progressive-role-budget/v1",
                    campaign=c,
                    task=t,
                    operation=operation,
                    inputs=digest(sorted(work_values, key=lambda v: v["identity"])),
                    reservation=reserved,
                    work_end_us=work_end_us,
                    storage=storage_proof,
                )
            ),
        )
        mounted.append((str(authority_dir), "/authority", False))
    argv = [
        "python",
        "-u",
        "-c",
        diagnostic_program(
            "import runpy;runpy.run_module('oeis_learn.pipeline_validation.progressive_role',run_name='__main__')",
            "discovery STARTUP; exposure unavailable until checkpoint verification; remaining deadline externally supervised",
        ),
        operation,
    ]
    environment = private_runtime_environment(role_output, destination="/output/discovery")
    if gpu:
        from .generalization_execution import startup_environment
        environment.update(startup_environment())
    environment["OEIS_DISCOVERY_PURPOSE"] = t["purpose"]
    environment['OEIS_DISCOVERY_TRACK']=t['track']
    command = host.engine.command(
        image=image,
        name=name,
        mounts=mounted,
        argv=argv,
        gpu=gpu,
        render_node=host.policy.get("render_node") if gpu else None,
        memory_bytes=host.policy["memory_bytes"],
        cpus=host.policy["cpus"],
        network=False,
        tmp_bytes=2 << 30,
        environment=environment,
    )
    try:
        supervise_owned(
            host,
            name,
            command,
            target,
            deadline=deadline,
            cleanup_seconds=cleanup_seconds,
            gpu=gpu,
            allowed_returncodes=(0, 1, 2),
        )
        journal.commit(
            charge["key"],
            boot=boot,
            monotonic_us=time.monotonic_ns() // 1000,
            termination_proven=True,
        )
        from oeis_learn.generalization.progressive.contracts import load

        result = load(role_output / "result.json")
        from oeis_learn.pipeline_validation.identity import seal

        if (
            seal(result) != result
            or result.get("version") != "discovery-result/v1"
            or result.get("operation") != operation
        ):
            raise ValueError("role terminal identity/operation differs")
        expected = {"COMPLETE": 0, "PARTIAL": 2, "BLOCKED": 2, "FAILED": 1, "INVALID": 1}
        import json

        process = json.loads((target / "supervision" / "process.json").read_text())
        if (
            result.get("mechanical_status") not in expected
            or process.get("returncode") != expected[result["mechanical_status"]]
        ):
            raise ValueError("role process/result disagree")
        return result
    finally:
        import json

        cleanup = target / "supervision" / "cleanup.json"
        process = target / "supervision" / "process.json"
        if cleanup.is_file() and process.is_file():
            q = json.loads(cleanup.read_text())
            p = json.loads(process.read_text())
            if (
                q.get("within_bound") is True
                and q.get("acknowledged") is True
                and p.get("cleanup_failure") is None
            ):
                journal.commit(
                    charge["key"],
                    boot=boot,
                    monotonic_us=time.monotonic_ns() // 1000,
                    termination_proven=True,
                )
        remove_projection(views)


def current_runtime(host, root, output, runtime, *, producer_root):
    """Reinspect original producers on the selected engine before execution."""
    from oeis_learn.generalization.progressive.campaign import validate_runtime
    from oeis_learn.generalization.progressive.contracts import load, validate

    from .generalization_qualification_io import current_facts

    r = validate("runtime", runtime)
    q = load(Path(producer_root) / "qualification.json")
    facts = current_facts(
        host,
        root,
        output,
        images=q["images"],
        source_ref=q["source_closure"],
        build_refs=q["build_dependencies"],
        producer_root=producer_root,
    )
    if not r['payload']['fixture']:
        from .progressive_handoff import stage, validate_device_proofs
        from oeis_learn.generalization.progressive.contracts import digest
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        request,_=validate_device_proofs(producer_root,q,source=digest(collect_snapshot(root)),
                                        image=q['images'][1],engine=host.engine.name)
        probe=Path(output)/'selected-device'
        code="from pathlib import Path;from oeis_learn.pipeline_validation.progressive_qualification import hardware_facts;from oeis_learn.generalization.progressive.contracts import atomic_write;atomic_write(Path('/work/output/hardware.json'),hardware_facts())"
        stage(host,root,q['images'][1],probe,['python','-c',code],gpu=True,seconds=120)
        if load(probe/'hardware.json')!=request['hardware']:
            raise ValueError('current exact selected HIP device differs')
    return validate_runtime(r, root=root, current_facts=facts, inspected=True)


def invoke(host, image, root, operation, artifacts, **kwargs):
    """Take the stable task lease before any reservation or projection work."""
    from oeis_learn.generalization.progressive.budgets import TaskLease

    campaign = kwargs["campaign"]
    expected = Path(root) / "run-state/discovery-accounting" / campaign["identity"].split(":")[1]
    if kwargs["journal"].root.resolve() != expected.resolve():
        raise ValueError("canonical campaign journal required; output cannot reset allowance")
    with TaskLease(Path(root) / "run-state/discovery-leases", kwargs["task"]):
        return _invoke_reserved(host, image, root, operation, artifacts, **kwargs)


def _trial_runtime_matches(root,campaign,task,runtime):
    from oeis_learn.generalization.progressive.contracts import verify_refs
    import json
    ref=task['payload']['trial_runtime']
    if not any(r['artifact']==ref and r['kind']=='qualification' for r in campaign['payload']['references']):return False
    value=json.loads(verify_refs(root,[ref])[0])
    return value['identity']==runtime.identity
