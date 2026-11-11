"""Stdlib command plumbing; unfinished research commands fail explicitly."""

from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES

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
    selected=[r['artifact'] for r in audit['references'] if r['name']=='preflight_plan']
    if selected:
        from oeis_learn.generalization.progressive.contracts import verify_refs
        from oeis_learn.generalization.progressive.preflight import prepare as preflight
        if len(selected)!=1:
            raise ValueError('one original outcome-blind preflight plan required')
        plan=json.loads(verify_refs(root,selected,max_bytes=4<<20)[0])
        if not isinstance(plan,dict) or set(plan)!={'version','identity','cohorts','exposure','sensitivity'} or plan['version']!='progressive-preflight-plan/v1' or seal(plan)!=plan:
            raise ValueError('closed original preflight plan required')
        design=json.loads(verify_refs(root,[plan['sensitivity']],max_bytes=4<<20)[0])
        atomic_write(output/'preparation-preflight.json',preflight(root,cohorts=plan['cohorts'],exposure=plan['exposure'],sensitivity_design=design))
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
    p = load(confined(root, args.prepared) / "prepared.json",maximum=CAMPAIGN_DOCUMENT_BYTES)
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

        campaign = load(confined(root, manifest["campaign_path"]),maximum=CAMPAIGN_DOCUMENT_BYTES)
        store = object.__new__(RunStore)
        store.root = confined(
            root,
            "run-state/discovery-control/" + campaign["identity"].split(":")[1],
            exists=False,
        )
        store.campaign = campaign
        observed = store.read()
        if store.ordinal == 0 and (folder / "result.json").is_file():
            original = load(folder / "result.json")
            if seal(original) != original or original.get("version") != "discovery-result/v1":
                raise ValueError("invalid original terminal record")
            observed["status"] = original["mechanical_status"]
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
        elif args.operation == 'qualify':
            from .progressive_handoff import qualify_command
            result,code=qualify_command(args,root,host_factory)
        elif args.operation == 'acceptance':
            from .progressive_handoff import acceptance_command
            result,code=acceptance_command(args,root,host_factory)
        elif args.operation == 'prepare':
            from .progressive_capability import prepare
            result,code=prepare(args,root,host_factory)
        elif args.operation == "freeze":
            result, code = freeze_command(args, root)
        elif (
            args.operation == "report"
            and (confined(root, args.run) / "capability-inputs.json").is_file()
        ):
            result, code = capability_report(args, root)
        elif args.operation in {"run", "resume", "pause"}:
            result, code = lifecycle_command(args, root, host_factory)
        elif args.operation == "assess":
            result, code = assess_command(args, root, host_factory)
        elif args.operation == "status":
            result, code = status(args, root)
        elif args.operation == "report":
            from oeis_learn.generalization.progressive.report import build

            folder = confined(root, args.run)
            if (folder/'public-report.json').is_file():request=load(folder/'public-report.json',maximum=32<<20)
            else:
                from .progressive_report_collect import collect
                request=collect(root,folder,confined(root,args.output+'.metadata',exists=False))
            manifest = build(root, confined(root, args.output, exists=False), request)
            complete = manifest['status']=='COMPLETE'
            result=seal(dict(version='discovery-result/v1',operation='report',campaign=manifest['campaign'],
                mechanical_status='COMPLETE' if complete else 'PARTIAL',scientific_status='UNMEASURED',
                qualification_status='PENDING',claim_status='UNREVIEWED',next_action='STOP_AFFECTED',
                reason=manifest['status'],evidence=args.output,missing_evidence=manifest['missing'],
                producer_error=manifest['producer_error'],collector_errors=manifest['collector_errors'],
                metrics=dict(report=manifest['identity'],counts=manifest['summary'])))
            code=0 if complete else 2
        elif args.operation == "verify-bundle":
            from oeis_learn.generalization.progressive.verify import verify

            output = _output(root, args.output)
            result = verify(confined(root, args.archive))
            atomic_write(output / "verification.json", result)
            verified=result['status']=='VERIFIED'
            original=result
            result=publish(output,operation='verify-bundle',mechanical_status='COMPLETE' if verified else 'PARTIAL',
                scientific_status=original['decision'],reason=original['status'],missing=original['missing'],
                producer_error=original['producer_error'],collector_errors=original['collector_errors'],
                metrics=dict(verification=original['identity'],recomputed=original['metrics']))
            code=0 if verified else 2
        else:
            raise ValueError('unknown discovery operation: '+args.operation)
    except KeyboardInterrupt as exc:
        caught = exc
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
    campaign = validate_campaign(load(confined(root, campaign_path),maximum=CAMPAIGN_DOCUMENT_BYTES))
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
        or receipt.get("version") != "discovery-acceptance/v1"
        or receipt.get("scope") != "progressive-lifecycle"
        or receipt.get("qualified") is not True
        or receipt.get("campaign") != campaign["identity"]
        or receipt.get("runtime") != campaign["payload"]["runtime"]
        or receipt.get("status") != "COMPLETE"
    ):
        raise ValueError("acceptance campaign/runtime/status differs")
    runtime = load(confined(root, campaign_path).parent / "runtime.json")
    feature_verified=False
    if {'software','device','request'}<=set(receipt):
        from .progressive_device_acceptance import consume_receipt
        consume_receipt(receipt_path.parent,receipt)
        feature_verified=True
    elif campaign['purpose']=='research':
        raise ValueError('original installed feature producers required')
    producers = verify_refs(root, runtime["payload"]["qualification"])
    if not feature_verified and (not producers or any(
        "progressive-lifecycle" not in json.loads(p).get("covered_capabilities", [])
        for p in producers
    )):
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
            if not request_refs:
                store.append(dict(store.state, status="PARTIAL"))
                return publish(
                    output,
                    operation=args.operation,
                    mechanical_status="PARTIAL",
                    campaign=campaign["identity"],
                    reason="TASK_REQUEST_DEFICIT",
                    missing=["frozen_task_request"],
                ), 2
            request_bytes = verify_refs(root, request_refs)
            if len(set(request_bytes)) != 1:
                raise ValueError("conflicting frozen request bytes")
            request = json.loads(request_bytes[0])
            if set(request) != {"operation", "artifacts", "image", "seconds", "reservation"}:
                raise ValueError("closed installed task request required")
            from .progressive_roles import IMPLEMENTED, REGISTRY

            if request["operation"] == "assess":
                raise ValueError("assessment requires explicit locked assess command")
            if request['operation'] in {'oeis-cycle','source-cycle'}:
                if request['operation']=='source-cycle':
                    from .progressive_source_cycle import run_cycle as selected_cycle
                else:selected_cycle=oeis_cycle
                result = selected_cycle(
                    root,
                    output / ("task-" + task["identity"].split(":")[1]),
                    campaign,
                    task,
                    request,
                    journal,
                    host=host,
                    runtime=checked,
                )
                from .progressive_outputs import commit_host_training
                commit_host_training(root,store,request,journal,result['metrics']['completed_tasks'])
                for task_id in result["metrics"]["completed_tasks"]:
                    plan = next(p for p in campaign["payload"]["tasks"] if p["id"] == task_id)
                    child = record(
                        "task",
                        dict(plan, campaign=campaign["identity"]),
                        created_at=campaign["created_at"],
                        purpose=campaign["purpose"],
                        track=campaign["track"],
                    )
                    store.boundary(child, "COMPLETE", _INVOCATION.get())
                store.boundary(task, result["metrics"]["task_status"], _INVOCATION.get())
                # A second cycle is a new study; do not let an unvisited paired
                # slot silently rerun a low-yield first cycle.
                if result['reason'] in {'OEIS_YIELD_DEFICIT','SOURCE_YIELD_DEFICIT'}:
                    for pending in campaign['payload']['tasks']:
                        if pending['policy']==task['payload']['policy'] and pending['id']!=task['payload']['id'] and pending['id'] not in result['metrics']['completed_tasks']:
                            child=record('task',dict(pending,campaign=campaign['identity']),created_at=campaign['created_at'],purpose=campaign['purpose'],track=campaign['track'])
                            store.boundary(child,'PARTIAL',_INVOCATION.get())
                    store.append(dict(store.state,status='PARTIAL'))
                    from oeis_learn.generalization.progressive.contracts import reference
                    return publish(output,operation=args.operation,mechanical_status='PARTIAL',scientific_status=result['scientific_status'],campaign=campaign['identity'],reason=result['reason'],metrics=dict(result['metrics'],cycle_result=reference(root,confined(root,load(folder/(request['operation']+'.json'))['output'])/'result.json'))),2
                continue
            if request["operation"] not in IMPLEMENTED:
                store.append(dict(store.state, status="PARTIAL"))
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
            from . import progressive_study as study
            trial=task['payload'].get('trial')
            if trial:
                artifacts,derived=study.prepare_inputs(root,store,task,request,output/('task-input-'+str(store.ordinal)),journal)
            else:
                from .progressive_outputs import resolve as resolve_outputs
                artifacts,derived=resolve_outputs(root,store,task,request,output/('task-input-'+task['identity'].split(':')[1]))
            from .progressive_quarantine_control import require_clean_inputs
            require_clean_inputs(root,journal,artifacts)
            remaining = journal.task_remaining(task)
            seconds = min(request['seconds'],remaining['elapsed_us']/1000000) if trial else request['seconds']
            reservation = dict(request['reservation'])
            reservation['elapsed_us']=int(seconds*1000000)
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
                    study.after(root,store,task,dict(mechanical_status='PARTIAL',metrics={'task_status':'PARTIAL'}),journal)
                    continue
                seconds = min(seconds, remaining["elapsed_us"] / 1000000)
                reservation["elapsed_us"] = int(seconds * 1000000)
            role_output = output / ("task-" + task["identity"].split(":")[1])
            task_runtime=checked
            if 'trial_runtime' in task['payload']:
                selected=json.loads(verify_refs(root,[task['payload']['trial_runtime']])[0])
                task_runtime=current_runtime(host,root,output/('trial-runtime-'+str(store.ordinal)),selected,producer_root=confined(root,selected['payload']['qualification'][0]['path']).parent)
            try:
                result = invoke(
                    host,
                    request["image"],
                    root,
                    request["operation"],
                    artifacts,
                    output=str(role_output.relative_to(root)),
                    campaign=campaign,
                    task=task,
                    journal=journal,
                    ordinal=store.ordinal,
                    reservation=reservation,
                    seconds=seconds,
                    runtime=task_runtime,
                    derived=derived,
                )
            except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
                from .progressive_interruption import recover
                recovered=recover(root,campaign,task,journal,role_output,exc) if request['operation']=='train' else None
                store.boundary(task, recovered or "FAILED", _INVOCATION.get())
                if recovered:
                    study.after(root,store,task,dict(mechanical_status='PARTIAL',metrics={'task_status':recovered}),journal)
                    store.append(dict(store.state,status='PAUSED' if recovered=='PAUSED' else 'PARTIAL'))
                    return publish(output,operation=args.operation,mechanical_status='PARTIAL',campaign=campaign['identity'],
                        reason='OWNED_INTERRUPTION_'+recovered,metrics={'task_status':recovered}),2
                raise
            if request["operation"] == "train":
                from .progressive_evidence import finalize_learning

                state_root = (
                    root
                    / "run-state/discovery-learning"
                    / campaign["identity"].split(":")[1]
                    / task["identity"].split(":")[1]
                )
                if (state_root / "binding.json").is_file():
                    finalize_learning(journal, task, state_root, role_output / "role")
            if request['operation']=='review-claim':
                from .progressive_quarantine_control import reconcile as reconcile_quarantine
                reconcile_quarantine(root,role_output/'role',result,journal)
            task_status = result.get("metrics", {}).get("task_status", result["mechanical_status"])
            if task_status in {"INVALID", "BLOCKED"}:
                task_status = "FAILED"
            from .progressive_decision_control import reconcile_decision

            if request['operation']=='census' and task_status=='PARTIAL':
                from .progressive_outputs import commit as commit_output
                commit_output(root,store,task,role_output/'role',request['operation'])
            if task_status == "COMPLETE":
                from .progressive_outputs import commit as commit_output
                commit_output(root,store,task,role_output/"role",request["operation"])
                reconcile_decision(root, role_output / "role", result, store, journal, task)
            store.boundary(task, task_status, _INVOCATION.get())
            study.after(root,store,task,result,journal)
            if trial and trial['phase']=='selection' and task_status=='COMPLETE':study.commit_selection(root,store,task,result)
            if store.state["status"] == "PAUSED" or (task_status == "FAILED" and not trial):
                break
        done = all(
            store.state["tasks"].get(t['id']) in {'COMPLETE','SKIPPED'} for t in campaign['payload']['tasks']
        )
        failed = store.state["status"] == "FAILED"
        paused = store.state["status"] == "PAUSED"
        terminal = "FAILED" if failed else "COMPLETE" if done else "PARTIAL"
        store.append(dict(store.state, status="PAUSED" if paused else terminal))
        return publish(
            output,
            operation=args.operation,
            campaign=campaign["identity"],
            mechanical_status=terminal,
            scientific_status="INCOMPLETE" if not done else "UNMEASURED",
            reason="OWNED_TASK_FAILED"
            if failed
            else "PAUSE_ACKNOWLEDGED"
            if paused
            else "FINITE_DAG_COMPLETE"
            if done
            else "NO_ELIGIBLE_RESERVED_NEXT_TASK",
            producer_error=result.get("producer_error") if failed else None,
            metrics={
                "tasks": store.state["tasks"],
                "decisions": store.state.get("decisions", {}),
                "paused": paused,
            },
            missing=[] if done else ["eligible_reserved_next_slot"],
        ), 1 if failed else 0 if done else 2


def oeis_cycle(root, output, campaign, manager, request, journal, **kwargs):
    """Explicit fixed-cycle orchestration; no user callbacks or recursive repeats."""
    from .progressive_oeis import run_cycle

    return run_cycle(root, output, campaign, manager, request, journal, **kwargs)


def prepare_source_pilot(sources, roles):
    """Outcome-blind custodian feasibility gate; never buys new source support."""
    from oeis_learn.generalization.progressive.sources import pilot_support

    return pilot_support(sources, roles)


def source_pilot_cycle(
    worker,
    prompts,
    truth,
    *,
    client,
    seed,
    base_targets,
    arm,
    feasibility,
    sources,
    roles,
    transaction,
    seconds=4,
    max_tokens=64,
):
    """Installed pilot adapter using shared sampler, admission, learner and scorer.

    Invoke only inside the frozen assisted task. The generator receives prompts
    alone; nine-term truth is passed solely to admission/scoring. Lack of accepted
    yield is an explicit bounded outcome, never grounds for expanding the source.
    """
    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.generalization.progressive.short_profile import admit
    from oeis_learn.pipeline_validation.progressive_evaluation import generate_short, score_short
    from oeis_learn.pipeline_validation.progressive_training import short_update
    from oeis_learn.generalization.progressive.sources import pilot_support

    if (
        feasibility != pilot_support(sources, roles)
        or feasibility.get("status") != "ELIGIBLE"
        or arm not in {"with_source", "without_source"}
        or len(base_targets) != 16
        or worker.recovery.task["track"] != "external_assisted"
    ):
        raise ValueError("eligible frozen assisted pilot and matched support required")
    if len(prompts) != len(truth) or len(prompts) > 8:
        raise ValueError("frozen short roster required")
    permitted = [
        dict(indices=list(map(str, range(9))), values=s["values"])
        for s, r in zip(sources, roles)
        if r == "training"
    ]
    from oeis_learn.generalization.progressive.short_profile import visible

    if any(t not in permitted or p != visible(t) for p, t in zip(prompts, truth)):
        raise ValueError("pilot generation/admission restricted to frozen training sources")
    from oeis_learn.generalization.progressive.budgets import KEYS

    charge = dict.fromkeys(KEYS, 0)
    charge.update(
        tokens=len(prompts) * max_tokens,
        draws=len(prompts),
        attempts=2 * len(prompts),
        rejections=len(prompts),
        native_work=2 * len(prompts),
    )
    remaining = worker.recovery.journal.task_remaining(worker.recovery.task)
    if any(charge[k] > remaining[k] for k in KEYS):
        raise TimeoutError("source pilot work exhausted")
    event = worker.recovery.journal.reserve(
        worker.recovery.p["id"],
        "source-pilot",
        worker.recovery.sequence(),
        charge,
        boot=transaction["boot"],
        monotonic_us=transaction["monotonic_us"],
    )
    worker.recovery.journal.commit(event["key"])
    generated = generate_short(
        worker.runner.encoder,
        worker.runner.decoder,
        prompts,
        seed=seed,
        seconds=seconds,
        max_tokens=max_tokens,
    )
    accepted = []
    scores = []
    for i, c in enumerate(generated):
        if c["complete"] and c["source"]:
            target = admit(
                c["source"],
                truth[i],
                member=digest(prompts[i]),
                client=client,
                purpose=worker.recovery.task["purpose"],
            )
            scores.append(score_short(c["source"], truth[i], client=client))
            if target:
                accepted.append(target)
    if arm == "with_source" and not accepted:
        return seal(
            dict(
                version="progressive-source-cycle/v1",
                status="YIELD_DEFICIT",
                candidates=generated,
                scores=scores,
                accepted=0,
                primary_scaling_eligible=False,
            )
        )
    targets = list(base_targets)
    if arm == "with_source":
        for count in range(1, 17):
            chosen = [accepted[i % len(accepted)] for i in range(count)]
            targets = chosen + list(base_targets[count:])
            if 4 * sum(len(t["tokens"]) for t in chosen) >= sum(len(t["tokens"]) for t in targets):
                break
    result, _ = short_update(worker, targets, **transaction)
    return seal(
        dict(
            version="progressive-source-cycle/v1",
            status="COMPLETE",
            candidates=generated,
            scores=scores,
            accepted=len(accepted),
            update=result,
            primary_scaling_eligible=False,
        )
    )


def assess_command(args, root, host_factory):
    from oeis_learn.generalization.progressive.assessment import validate_selection, expose
    from oeis_learn.generalization.progressive.contracts import verify_refs, reference
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, role_work_requirements
    from oeis_learn.generalization.progressive.projections import describe
    from .progressive_roles import invoke, current_runtime

    campaign = load(confined(root, args.campaign),maximum=CAMPAIGN_DOCUMENT_BYTES)
    selection = validate_selection(root, load(confined(root, args.selection)), campaign)
    acceptance = load(confined(root, args.acceptance) / "acceptance.json")
    if (
        seal(acceptance) != acceptance
        or acceptance.get("version") != "discovery-acceptance/v1"
        or acceptance.get("scope") != "progressive-lifecycle"
        or acceptance.get("campaign") != campaign["identity"]
        or acceptance.get("runtime") != campaign["payload"]["runtime"]
        or not acceptance.get("qualified")
        or acceptance.get("status") != "COMPLETE"
    ):
        raise ValueError("compatible final assessment acceptance required")
    if {'software','device','request'}<=set(acceptance):
        from .progressive_device_acceptance import consume_receipt
        consume_receipt(confined(root,args.acceptance),acceptance)
    elif campaign['purpose']=='research':
        raise ValueError('original final-assessment feature producers required')
    runtime = load(confined(root, args.campaign).parent / "runtime.json")
    verify_refs(root, runtime["payload"]["qualification"])
    output = _output(root, args.output)
    host = host_factory(args)
    checked = current_runtime(
        host,
        root,
        output,
        runtime,
        producer_root=confined(root, runtime["payload"]["qualification"][0]["path"]).parent,
    )
    expose(root, campaign, selection)
    channel = seal(
        dict(
            version="discovery-channel/v1",
            kind="locked_selection",
            role="assessment",
            data=selection,
        )
    )
    atomic_write(output / "selection-channel.json", channel)
    artifacts = dict(
        candidates=[describe(root, selection["candidates"]["path"])],
        assessment_truth=[describe(root, r["path"]) for r in selection["panel"]],
        selection=[describe(root, reference(root, output / "selection-channel.json")["path"])],
    )
    values = [load(confined(root, a.ref["path"])) for items in artifacts.values() for a in items]
    cost = role_work_requirements("assess", values)
    task = selection["task"]
    seconds = min(3600, task["payload"]["caps"]["elapsed_us"] / 1e6)
    cost["elapsed_us"] = int(seconds * 1e6)
    journal = BudgetJournal(
        root / "run-state/discovery-accounting" / campaign["identity"].split(":")[1],
        campaign["identity"],
        campaign["payload"]["caps"],
    )
    derived = seal(
        dict(
            version="progressive-assessment-derived/v1",
            selection=selection,
            artifacts=[a.ref for items in artifacts.values() for a in items],
        )
    )
    image = runtime["payload"]["scope"]["images"][0]
    result = invoke(
        host,
        image,
        root,
        "assess",
        artifacts,
        output=str((output / "assessment").relative_to(root)),
        campaign=campaign,
        task=task,
        journal=journal,
        ordinal=0,
        reservation=cost,
        seconds=seconds,
        runtime=checked,
        derived=derived,
    )
    atomic_write(
        output / "assessment-complete.json",
        seal(dict(selection=selection["identity"], result=result)),
    )
    return result, 0 if result["mechanical_status"] == "COMPLETE" else 2
