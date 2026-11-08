"""Explicit discovery operation registry and supervised installed-role adapter."""

from dataclasses import dataclass
from pathlib import Path
import time
from types import MappingProxyType
import uuid

from oeis_learn.pipeline_validation.identity import digest, confined
from oeis_learn.generalization.progressive.projections import project, remove_projection


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
    {name: dict(role=v.role, channels=list(v.channels), gpu=v.gpu) for name, v in REGISTRY.items()}
)
IMPLEMENTED = frozenset({"census", "audit-language", "publish"})


def deadline_for(task, *, remaining_us, seconds, runtime=None, operation):
    """Only validated discovery authority can extend the generic six-hour cap."""
    from oeis_learn.generalization.progressive.campaign import validate_task, VerifiedRuntime
    from oeis_learn.generalization.budgets import integer

    t = validate_task(task)
    p = t["payload"]
    integer(remaining_us, 1)
    if type(seconds) not in (int, float) or not 0 < seconds <= 86400:
        raise ValueError("bounded discovery role duration")
    if int(seconds * 1_000_000) > remaining_us or remaining_us > p["caps"]["elapsed_us"]:
        raise ValueError("enclosing remaining deadline exceeded")
    if remaining_us <= p["startup_us"] + p["finalization_us"]:
        raise ValueError("protected startup/finalization exhausted")
    if seconds > 21600:
        if (
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
):
    if operation not in IMPLEMENTED:
        raise NotImplementedError("discovery operation not implemented: " + operation)
    if settings:
        raise ValueError("no arbitrary role settings accepted")
    if runtime is None or not runtime.execution_ready:
        raise ValueError("freshly inspected original qualified runtime required")
    from oeis_learn.generalization.progressive.campaign import validate_task, validate_campaign
    from oeis_learn.generalization.progressive.budgets import vector

    c = validate_campaign(campaign)
    t = validate_task(task, campaign=c)
    if (
        c["payload"]["runtime"] != runtime.identity
        or runtime.track != t["track"]
        or runtime.purpose != t["purpose"]
    ):
        raise ValueError("campaign/runtime/task mismatch")
    if (
        journal.binding["campaign"] != c["identity"]
        or journal.binding["caps"] != c["payload"]["caps"]
    ):
        raise ValueError("unrelated campaign journal")
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
    target.mkdir(parents=True)
    spec = REGISTRY[operation]
    views = target / "private-input"
    authorized = [r["artifact"] for r in c["payload"]["references"] if r["role"] == spec.role]
    channels = project(
        root,
        views,
        operation,
        artifacts,
        registry=REGISTRY,
        forbidden=forbidden,
        authorized=authorized,
    )
    from .generalization_roles import supervise_owned
    from .supervision import diagnostic_program
    from oeis_learn.container_runtime import private_runtime_environment

    name = "discovery-" + uuid.uuid4().hex
    role_output = target / "role"
    role_output.mkdir()
    mounted = [(str(path), "/inputs/" + channel, False) for channel, path in channels.items()] + [
        (str(role_output), "/output/discovery", True)
    ]
    argv = [
        "python",
        "-u",
        "-c",
        diagnostic_program(
            "import runpy;runpy.run_module('oeis_learn.pipeline_validation.progressive_role',run_name='__main__')",
            "discovery",
        ),
        operation,
    ]
    environment = private_runtime_environment(role_output, destination="/output/discovery")
    command = host.engine.command(
        image=image,
        name=name,
        mounts=mounted,
        argv=argv,
        gpu=spec.gpu,
        render_node=host.policy.get("render_node") if spec.gpu else None,
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
            cleanup_seconds=min(30, seconds / 5),
            gpu=spec.gpu,
        )
        journal.commit(
            charge["key"],
            boot=boot,
            monotonic_us=time.monotonic_ns() // 1000,
            termination_proven=True,
        )
        from oeis_learn.generalization.progressive.contracts import load

        result = load(role_output / "result.json")
        if result.get("mechanical_status") not in {"COMPLETE", "PARTIAL", "FAILED", "BLOCKED"}:
            raise ValueError("role missing terminal result")
        return result
    finally:
        remove_projection(views)


def current_runtime(host, root, output, runtime, *, producer_root):
    """Reinspect original producers on the selected engine before execution."""
    from .generalization_qualification_io import current_facts
    from oeis_learn.generalization.progressive.contracts import validate, load
    from oeis_learn.generalization.progressive.campaign import validate_runtime

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
