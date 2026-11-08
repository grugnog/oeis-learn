"""Stdlib command plumbing; unfinished research commands fail explicitly."""

from datetime import datetime, timezone
import json
from pathlib import Path
from contextvars import ContextVar
import uuid
from oeis_learn.generalization.progressive.contracts import load, record, atomic_write
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
            dict(
                version="discovery-owner/v1",
                output=str(path.relative_to(root)),
                invocation=_INVOCATION.get(),
            )
        ),
    )
    return path


def init(args, root):
    from oeis_learn.generalization.progressive.freeze import inventory
    from oeis_learn.generalization.progressive.budgets import KEYS

    audit = inventory(root, args.reference_campaign)
    output = _output(root, args.output)
    blueprint = record(
        "blueprint",
        dict(
            executable=False,
            selected_stages=["capability", "diversity"],
            references=audit["references"],
            unresolved=audit["unresolved"],
            proposed_caps=dict.fromkeys(KEYS, 0),
            reference_campaign=args.reference_campaign,
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
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
    from oeis_learn.generalization.progressive.freeze import freeze, IncompleteFreeze

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
        _, pending, finished, totals, _ = journal._read()
        value = seal(
            dict(
                value,
                observed_result=value["identity"],
                accounting=dict(
                    consumed=totals,
                    remaining={k: binding["caps"][k] - totals[k] for k in totals},
                    pending=list(pending),
                ),
            )
        )
    return value, 0


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
        elif args.operation == "status":
            result, code = status(args, root)
        else:
            # Later task owners must replace these explicit blocked endpoints;
            # no engine is contacted or research launched by a placeholder.
            result = seal(
                dict(
                    version="discovery-result/v1",
                    operation=args.operation,
                    campaign=None,
                    stage=None,
                    task=None,
                    mechanical_status="BLOCKED",
                    scientific_status="UNMEASURED",
                    qualification_status="PENDING",
                    claim_status="UNREVIEWED",
                    next_action="STOP_AFFECTED",
                    target_policy=None,
                    reason="NOT_IMPLEMENTED:" + args.operation,
                    evidence=None,
                    missing_evidence=["implementation"],
                )
            )
            code = 2
    except KeyboardInterrupt:
        result = seal(
            dict(
                version="discovery-result/v1",
                operation=args.operation,
                mechanical_status="INTERRUPTED",
                scientific_status="UNMEASURED",
                next_action="STOP_AFFECTED",
                reason="NO_DURABLE_RESUME_CONFIRMED",
                missing_evidence=["recovery"],
            )
        )
        code = 130
    except (ValueError, KeyError, TypeError) as exc:
        caught = exc
        from .progressive_evidence import safe_failure

        result = seal(
            dict(
                version="discovery-result/v1",
                operation=args.operation,
                mechanical_status="INVALID",
                scientific_status="INVALID",
                next_action="STOP_AFFECTED",
                reason="CONTRACT_MISMATCH",
                producer_error=safe_failure(exc, producer="host"),
                missing_evidence=[],
            )
        )
        code = 3
    except (OSError, RuntimeError) as exc:
        caught = exc
        from .progressive_evidence import safe_failure

        result = seal(
            dict(
                version="discovery-result/v1",
                operation=args.operation,
                mechanical_status="FAILED",
                scientific_status="UNMEASURED",
                next_action="STOP_AFFECTED",
                reason="HOST_FAILURE",
                producer_error=safe_failure(exc, producer="host"),
                missing_evidence=[],
            )
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
                from .progressive_evidence import safe_failure
                from oeis_learn.generalization.progressive.contracts import reference

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
                        dict(type=type(collector).__name__, message=str(collector)[:2048])
                    ],
                )
            )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    _INVOCATION.reset(token)
    return code
