"""Stdlib command plumbing; unfinished research commands fail explicitly."""

import json
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import atomic_write, load, record
from oeis_learn.pipeline_validation.identity import confined, seal

from .progressive_evidence import publish

_INVOCATION = ContextVar("discovery_invocation", default=None)


def _output(root, name):
    path = confined(root, name, exists=False)
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("fresh empty output required")
    else:
        path.mkdir(parents=True)
    atomic_write(
        path / "owner.json",
        seal(
            {
                "version": "discovery-owner/v1",
                "output": str(path.relative_to(root)),
                "invocation": _INVOCATION.get(),
            }
        ),
    )
    return path


def init(args, root):
    from oeis_learn.generalization.progressive.budgets import KEYS
    from oeis_learn.generalization.progressive.freeze import inventory

    audit = inventory(root, args.reference_campaign)
    output = _output(root, args.output)
    blueprint = record(
        "blueprint",
        {
            "executable": False,
            "selected_stages": ["capability", "diversity"],
            "references": audit["references"],
            "unresolved": audit["unresolved"],
            "proposed_caps": dict.fromkeys(KEYS, 0),
            "reference_campaign": args.reference_campaign,
        },
        created_at=datetime.now(UTC).isoformat(),
        purpose="research",
        track="synthetic_strict",
    )
    atomic_write(output / "blueprint.json", blueprint)
    atomic_write(output / "inputs.json", audit)
    return publish(
        output,
        operation="init",
        mechanical_status="COMPLETE",
        reason="BLUEPRINT_NON_EXECUTABLE",
        missing=audit["unresolved"],
    ), 0


def freeze_command(args, root):
    from oeis_learn.generalization.progressive.freeze import IncompleteFreeze, freeze

    b = load(confined(root, args.blueprint))
    p = load(confined(root, args.prepared) / "prepared.json")
    output = _output(root, args.output)
    try:
        campaign, runtime, tasks = freeze(root, b, p, qualification_root=args.qualification)
    except IncompleteFreeze as exc:
        return publish(
            output,
            operation="freeze",
            mechanical_status="BLOCKED",
            scientific_status="INCOMPLETE",
            reason="FREEZE_DEFICIT",
            missing=exc.deficits,
        ), 2
    atomic_write(output / "campaign.json", campaign)
    atomic_write(output / "runtime.json", runtime)
    for t in tasks:
        atomic_write(output / "tasks" / (t["identity"].split(":")[1] + ".json"), t)
    return publish(
        output,
        operation="freeze",
        mechanical_status="COMPLETE",
        reason="FROZEN_EXECUTION_STILL_REQUIRES_CURRENT_ACCEPTANCE",
        campaign=campaign["identity"],
    ), 0


def status(args, root):
    folder = confined(root, args.run)
    from oeis_learn.pipeline_validation.identity import seal

    if (folder / "run.json").is_file():
        manifest = load(folder / "run.json")
        if seal(manifest) != manifest:
            raise ValueError("run manifest changed")
        from oeis_learn.generalization.progressive.lifecycle import RunStore

        campaign = load(confined(root, manifest["campaign_path"]))
        store = object.__new__(RunStore)
        store.root = confined(
            root,
            "run-state/discovery-control/" + campaign["identity"].split(":")[1],
            exists=False,
        )
        store.campaign = campaign
        observed = store.read()
        value = seal(
            {
                "version": "discovery-result/v1",
                "operation": "status",
                "campaign": campaign["identity"],
                "mechanical_status": observed["status"],
                "tasks": observed["tasks"],
                "scientific_status": "UNMEASURED",
            }
        )
    else:
        value = load(folder / "result.json")
    if value.get("version") != "discovery-result/v1" or seal(value) != value:
        raise ValueError("invalid terminal record")
    # Read-only: status never allocates authority or invents a task's remainder.
    accounting_root = folder
    campaign = value.get("campaign")
    if isinstance(campaign, str) and len(campaign) == 71 and campaign.startswith("sha256:"):
        accounting_root = confined(
            root, "run-state/discovery-accounting/" + campaign.split(":")[1], exists=False
        )
    budget = accounting_root / "budget-binding.json"
    if budget.is_file():
        from oeis_learn.generalization.progressive.budgets import BudgetJournal

        binding = load(budget)
        if seal(binding) != binding:
            raise ValueError("budget binding identity differs")
        # Do not open a mutating journal/lock for status; replay directly.
        journal = object.__new__(BudgetJournal)
        journal.root = accounting_root
        journal.path = accounting_root / "budget.jsonl"
        journal.binding = {k: binding[k] for k in ("campaign", "caps")}
        _, pending, _finished, totals, _ = journal._read()
        value = seal(
            dict(
                value,
                observed_result=value["identity"],
                accounting={
                    "consumed": totals,
                    "remaining": {k: binding["caps"][k] - totals[k] for k in totals},
                    "pending": list(pending),
                },
            )
        )
    return value, 0


def publish_capability_inputs(root, output, *, cohort, exposure, coverage, evaluations):
    """Bind custodian/audit/generator/scorer outputs for read-only reconciliation."""
    from oeis_learn.generalization.progressive.readiness import from_request

    request = seal(
        {
            "version": "capability-readiness-request/v1",
            "cohort": cohort,
            "exposure": exposure,
            "coverage": coverage,
            "evaluations": evaluations,
        }
    )
    result = from_request(root, request)
    folder = confined(root, output, exists=False)
    if folder.exists():
        raise ValueError("fresh capability publication required")
    atomic_write(folder / "capability-inputs.json", request)
    atomic_write(folder / "readiness.json", result)
    return result


def capability_report(args, root):
    """Bounded public-only capability export; full bundle verification is T080."""
    import zipfile

    from oeis_learn.generalization.progressive.readiness import from_request

    request = load(confined(root, args.run) / "capability-inputs.json")
    readiness = from_request(root, request)
    result = seal(
        {
            "version": "discovery-result/v1",
            "operation": "report",
            "mechanical_status": readiness["status"],
            "scientific_status": "UNMEASURED",
            "qualification_status": "PENDING",
            "claim_status": "UNREVIEWED",
            "next_action": "STOP_AFFECTED",
            "reason": "CAPABILITY_READINESS_ONLY",
            "missing_evidence": readiness["missing"],
            "evidence": "readiness.json",
        }
    )
    target = confined(root, args.output, exists=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        target.open("xb") as stream,
        zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        for name, value in (("readiness.json", readiness), ("result.json", result)):
            blob = json.dumps(value, sort_keys=True, allow_nan=False).encode()
            if len(blob) > 32 << 20:
                raise ValueError("bounded capability export required")
            archive.writestr(name, blob)
    return result, 0 if readiness["status"] == "COMPLETE" else 2


def main(args, root, host_factory):
    root = Path(root).resolve()
    invocation = uuid.uuid4().hex
    token = _INVOCATION.set(invocation)
    caught = None
    try:
        if args.operation == "init":
            result, code = init(args, root)
        elif args.operation == "freeze":
            result, code = freeze_command(args, root)
        elif (
            args.operation == "report"
            and (confined(root, args.run) / "capability-inputs.json").is_file()
        ):
            result, code = capability_report(args, root)
        elif args.operation in {"run", "resume", "pause"}:
            result, code = lifecycle_command(args, root, host_factory)
        elif args.operation == "status":
            result, code = status(args, root)
        else:
            # Later task owners must replace these explicit blocked endpoints;
            # no engine is contacted or research launched by a placeholder.
            result = seal(
                {
                    "version": "discovery-result/v1",
                    "operation": args.operation,
                    "campaign": None,
                    "stage": None,
                    "task": None,
                    "mechanical_status": "BLOCKED",
                    "scientific_status": "UNMEASURED",
                    "qualification_status": "PENDING",
                    "claim_status": "UNREVIEWED",
                    "next_action": "STOP_AFFECTED",
                    "target_policy": None,
                    "reason": "NOT_IMPLEMENTED:" + args.operation,
                    "evidence": None,
                    "missing_evidence": ["implementation"],
                }
            )
            code = 2
    except KeyboardInterrupt:
        result = seal(
            {
                "version": "discovery-result/v1",
                "operation": args.operation,
                "mechanical_status": "INTERRUPTED",
                "scientific_status": "UNMEASURED",
                "next_action": "STOP_AFFECTED",
                "reason": "NO_DURABLE_RESUME_CONFIRMED",
                "missing_evidence": ["recovery"],
            }
        )
        code = 130
    except (ValueError, KeyError, TypeError) as exc:
        caught = exc
        from .progressive_evidence import safe_failure

        result = seal(
            {
                "version": "discovery-result/v1",
                "operation": args.operation,
                "mechanical_status": "INVALID",
                "scientific_status": "INVALID",
                "next_action": "STOP_AFFECTED",
                "reason": "CONTRACT_MISMATCH",
                "producer_error": safe_failure(exc, producer="host"),
                "missing_evidence": [],
            }
        )
        code = 3
    except (OSError, RuntimeError) as exc:
        caught = exc
        from .progressive_evidence import safe_failure

        result = seal(
            {
                "version": "discovery-result/v1",
                "operation": args.operation,
                "mechanical_status": "FAILED",
                "scientific_status": "UNMEASURED",
                "next_action": "STOP_AFFECTED",
                "reason": "HOST_FAILURE",
                "producer_error": safe_failure(exc, producer="host"),
                "missing_evidence": [],
            }
        )
        code = 1
    if caught is not None and getattr(args, "output", None):
        # Only a newly owned output may receive terminal diagnostics. A failed
        # attempt to reuse an earlier output never mutates that earlier run.
        try:
            output = confined(root, args.output, exists=False)
            if (
                (output / "owner.json").is_file()
                and load(output / "owner.json").get("invocation") == invocation
                and not (output / "result.json").exists()
            ):
                import traceback

                from oeis_learn.generalization.progressive.contracts import reference

                from .progressive_evidence import safe_failure

                private = output / "private"
                private.mkdir(exist_ok=True)
                private.chmod(0o700)
                trace = private / "host-traceback.txt"
                with trace.open("x") as stream:
                    stream.write("".join(traceback.format_exception(caught)))
                trace.chmod(0o600)
                result = seal(
                    dict(
                        result,
                        producer_error=safe_failure(
                            caught, producer="host", traceback_ref=reference(output, trace)
                        ),
                    )
                )
                atomic_write(output / "result.json", result)
        except (ValueError, OSError) as collector:
            result = seal(
                dict(
                    result,
                    collector_errors=[
                        {"type": type(collector).__name__, "message": str(collector)[:2048]}
                    ],
                )
            )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    _INVOCATION.reset(token)
    return code


def lifecycle_command(args, root, host_factory):
    """Finite invocation dispatch, retaining the original campaign accounting root."""
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.campaign import validate_campaign
    from oeis_learn.generalization.progressive.contracts import verify_refs
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.generalization.progressive.projections import describe

    from .progressive_roles import current_runtime, invoke

    if args.operation == "run":
        campaign_path = args.campaign
        acceptance_path = args.acceptance
    else:
        manifest = load(confined(root, args.run) / "run.json")
        if seal(manifest) != manifest:
            raise ValueError("run manifest seal differs")
        campaign_path = manifest["campaign_path"]
        acceptance_path = manifest["acceptance_path"]
    campaign = validate_campaign(load(confined(root, campaign_path)))
    folder = confined(
        root, "run-state/discovery-accounting/" + campaign["identity"].split(":")[1], exists=False
    )
    control = confined(
        root, "run-state/discovery-control/" + campaign["identity"].split(":")[1], exists=False
    )
    if args.operation == "pause":
        if not (control / "campaign.json").is_file():
            raise ValueError("owned campaign run required")
        store = RunStore(control, campaign)
        state = store.pause()
        return seal(
            {
                "version": "discovery-result/v1",
                "operation": "pause",
                "mechanical_status": state,
                "campaign": campaign["identity"],
                "scientific_status": "UNMEASURED",
                "reason": "OWNED_BOUNDARY_REQUEST",
            }
        ), 2 if state != "PAUSED" else 0
    output = _output(root, args.output)
    atomic_write(
        output / "run.json",
        seal(
            {
                "version": "discovery-run/v1",
                "campaign_path": campaign_path,
                "acceptance_path": acceptance_path,
                "campaign": campaign["identity"],
                "previous": getattr(args, "run", None),
                "invocation": _INVOCATION.get(),
            }
        ),
    )
    receipt_path = confined(root, acceptance_path) / "acceptance.json"
    if not receipt_path.is_file():
        return publish(
            output,
            operation=args.operation,
            mechanical_status="BLOCKED",
            campaign=campaign["identity"],
            reason="CURRENT_FEATURE_ACCEPTANCE_REQUIRED",
            missing=["acceptance"],
        ), 2
    receipt = load(receipt_path)
    if (
        seal(receipt) != receipt
        or receipt.get("campaign") != campaign["identity"]
        or receipt.get("runtime") != campaign["payload"]["runtime"]
        or receipt.get("status") != "COMPLETE"
    ):
        raise ValueError("acceptance campaign/runtime/status differs")
    runtime = load(confined(root, campaign_path).parent / "runtime.json")
    producers = verify_refs(root, runtime["payload"]["qualification"])
    if not producers or any(
        "progressive-lifecycle" not in json.loads(p).get("covered_capabilities", [])
        for p in producers
    ):
        return publish(
            output,
            operation=args.operation,
            mechanical_status="BLOCKED",
            campaign=campaign["identity"],
            reason="LIFECYCLE_PRODUCER_REQUIRED",
            missing=["progressive-lifecycle"],
        ), 2
    host = host_factory(args)
    producer_root = confined(root, runtime["payload"]["qualification"][0]["path"]).parent
    checked = current_runtime(host, root, output, runtime, producer_root=producer_root)
    store = RunStore(control, campaign)
    journal = BudgetJournal(folder, campaign["identity"], campaign["payload"]["caps"])
    with store.lease():
        store.append(
            {"tasks": store.state["tasks"], "status": "RUNNING", "invocation": _INVOCATION.get()}
        )
        for _ in range(len(campaign["payload"]["tasks"])):
            task = store.next_task()
            if task is None:
                break
            request_refs = [
                r["artifact"]
                for r in campaign["payload"]["references"]
                if r["kind"] == "policy" and r["artifact"]["sha256"] == task["payload"]["policy"]
            ]
            if len(request_refs) != 1:
                return publish(
                    output,
                    operation=args.operation,
                    mechanical_status="PARTIAL",
                    campaign=campaign["identity"],
                    reason="TASK_REQUEST_DEFICIT",
                    missing=["frozen_task_request"],
                ), 2
            request = json.loads(verify_refs(root, request_refs)[0])
            if set(request) != {"operation", "artifacts", "image", "seconds", "reservation"}:
                raise ValueError("closed installed task request required")
            from .progressive_roles import IMPLEMENTED, REGISTRY

            if request["operation"] not in IMPLEMENTED:
                return publish(
                    output,
                    operation=args.operation,
                    mechanical_status="BLOCKED",
                    reason="REGISTERED_OPERATION_PENDING",
                    missing=[request["operation"]],
                    campaign=campaign["identity"],
                ), 2
            if not isinstance(request["artifacts"], dict) or set(request["artifacts"]) != set(
                REGISTRY[request["operation"]].channels
            ):
                raise ValueError("frozen exact role channel map required")
            artifacts = {
                channel: [describe(root, p) for p in paths]
                for channel, paths in request["artifacts"].items()
            }
            remaining = journal.task_remaining(task)
            seconds = request["seconds"]
            reservation = dict(request["reservation"])
            if request["operation"] == "train":
                if (
                    not remaining["tokens"]
                    or remaining["elapsed_us"]
                    <= task["payload"]["startup_us"] + task["payload"]["finalization_us"]
                ):
                    from oeis_learn.generalization.progressive.recovery import Recovery

                    state_root = confined(
                        root,
                        "run-state/discovery-learning/"
                        + campaign["identity"].split(":")[1]
                        + "/"
                        + task["identity"].split(":")[1],
                        exists=False,
                    )
                    with Recovery(
                        journal,
                        task,
                        task["payload"]["initializer"]["compatibility"],
                        state_root=state_root,
                    ) as recovery:
                        recovery.append(status="PARTIAL")
                    store.boundary(task, "PARTIAL", _INVOCATION.get())
                    continue
                seconds = min(seconds, remaining["elapsed_us"] / 1000000)
                reservation["elapsed_us"] = int(seconds * 1000000)
            result = invoke(
                host,
                request["image"],
                root,
                request["operation"],
                artifacts,
                output=str((output / ("task-" + task["identity"].split(":")[1])).relative_to(root)),
                campaign=campaign,
                task=task,
                journal=journal,
                ordinal=store.ordinal,
                reservation=reservation,
                seconds=seconds,
                runtime=checked,
            )
            task_status = result.get("metrics", {}).get("task_status", result["mechanical_status"])
            if task_status in {"INVALID", "BLOCKED"}:
                task_status = "FAILED"
            store.boundary(task, task_status, _INVOCATION.get())
            if store.state["status"] == "PAUSED" or task_status == "FAILED":
                break
        done = all(
            store.state["tasks"].get(t["id"]) == "COMPLETE" for t in campaign["payload"]["tasks"]
        )
        return publish(
            output,
            operation=args.operation,
            campaign=campaign["identity"],
            mechanical_status="COMPLETE" if done else "PARTIAL",
            reason="FINITE_DAG_COMPLETE" if done else "NO_ELIGIBLE_RESERVED_NEXT_TASK",
            missing=[] if done else ["eligible_reserved_next_slot"],
        ), 0 if done else 2
