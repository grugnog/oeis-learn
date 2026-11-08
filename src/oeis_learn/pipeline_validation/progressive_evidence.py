"""Initial terminal evidence with independent statuses and safe producer failures."""

import traceback
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import atomic_write, reference
from oeis_learn.pipeline_validation.identity import seal


def safe_failure(exc, *, producer, traceback_ref=None, private_values=()):
    message = str(exc)
    for value in sorted({str(v) for v in private_values if str(v)}, key=len, reverse=True):
        message = message.replace(value, "[redacted]")
    return {
        "type": type(exc).__name__,
        "message": message[:2048],
        "code": getattr(exc, "code", "PRODUCER_FAILURE"),
        "producer": producer,
        "local_traceback": traceback_ref,
    }


def publish(
    output,
    *,
    operation,
    mechanical_status,
    scientific_status="UNMEASURED",
    qualification_status="PENDING",
    claim_status="UNREVIEWED",
    next_action="STOP_AFFECTED",
    reason,
    missing=(),
    producer_error=None,
    collector_errors=(),
    campaign=None,
    stage=None,
    task=None,
    target_policy=None,
    metrics=None,
):
    output = Path(output)
    if mechanical_status not in {
        "COMPLETE",
        "PARTIAL",
        "BLOCKED",
        "FAILED",
        "INVALID",
        "INTERRUPTED",
    }:
        raise ValueError("terminal mechanical state required")
    if scientific_status not in {
        "UNMEASURED",
        "CONFIRMED_GAIN",
        "CONFIRMED_EFFICIENCY",
        "NO_MATERIAL_GAIN",
        "REGRESSION",
        "INCONCLUSIVE",
        "INCOMPLETE",
        "INVALID",
    }:
        raise ValueError("scientific state required")
    result = seal(
        {
            "version": "discovery-result/v1",
            "operation": operation,
            "campaign": campaign,
            "stage": stage,
            "task": task,
            "mechanical_status": mechanical_status,
            "scientific_status": scientific_status,
            "qualification_status": qualification_status,
            "claim_status": claim_status,
            "next_action": next_action,
            "target_policy": target_policy,
            "reason": reason,
            "evidence": "result.json",
            "missing_evidence": list(missing),
            "producer_error": producer_error,
            "collector_errors": list(collector_errors),
            "metrics": metrics
            or {"device": None, "timing": None, "unavailable_reason": "not_measured"},
        }
    )
    if output.exists() and (output / "result.json").exists():
        raise ValueError("terminal output already completed")
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / "result.json", result)
    return result


def failure(output, operation, exc, *, producer, private_values=(), collector_errors=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Detailed original traceback stays local/private; exports retain its reference.
    private = output / "private"
    private.mkdir(exist_ok=True)
    private.chmod(0o700)
    trace = private / "producer-traceback.txt"
    with trace.open("x") as stream:
        stream.write("".join(traceback.format_exception(exc)))
    trace.chmod(0o600)
    error = safe_failure(
        exc,
        producer=producer,
        traceback_ref=reference(output, trace),
        private_values=private_values,
    )
    return publish(
        output,
        operation=operation,
        mechanical_status="FAILED",
        reason=error["code"],
        producer_error=error,
        collector_errors=collector_errors,
        missing=["scientific_outcomes", "qualification"],
    )


def private_values(value):
    """Extract actual protected strings for diagnostic redaction, never fixed masks."""
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"source", "values", "indices", "witness"}:
                if isinstance(child, str):
                    result.append(child)
                elif isinstance(child, list):
                    result.extend(x for x in child if isinstance(x, str))
            else:
                result.extend(private_values(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(private_values(child))
    return result


def diversity_report(static, refreshed, *, decision=None):
    """Preserve matched costs and negative decisions without inferring gain from loss."""
    from oeis_learn.generalization.progressive.contracts import validate
    from oeis_learn.pipeline_validation.identity import seal

    for result, arm in ((static, "static"), (refreshed, "refreshed")):
        if (
            seal(result) != result
            or result.get("version") != "progressive-diversity-acceptance/v1"
            or result["arm"] != arm
        ):
            raise ValueError("sealed matched pilot results required")
    if any(
        static[k] != refreshed[k]
        for k in ("initial_shard", "initial_weights", "matched_policy", "seed")
    ):
        raise ValueError("unmatched data/model/optimizer/allocation comparison")
    reported = None
    action = "RETAIN_BASELINE"
    if decision is not None:
        decision = validate("decision", decision)
        reported = decision["payload"]["evidence_verdict"]
        if decision["payload"]["next_action"] != "RETAIN_BASELINE":
            action = "STOP_AFFECTED"
    return seal(
        {
            "version": "progressive-diversity-report/v1",
            "kind": "DIAGNOSTIC_ONLY",
            "scientific_status": "UNMEASURED",
            "qualified": False,
            "static": static["identity"],
            "refreshed": refreshed["identity"],
            "comparable_initialization": True,
            "reported_verdict": reported,
            "decision": decision["identity"] if decision else None,
            "decision_reconciliation": "verify against outcomes before research use",
            "next_action": action,
            "training_loss_is_not_capability": True,
            "arms": {
                arm: {
                    k: r[k]
                    for k in (
                        "updates",
                        "charged_tokens",
                        "useful_tokens",
                        "discarded_tokens",
                        "elapsed_us",
                        "equal_wall_tokens_per_second",
                        "preparation_work_units",
                        "preparation_elapsed_us",
                    )
                }
                for arm, r in (("static", static), ("refreshed", refreshed))
            },
            "preparation_accounting": "shared preallocated shard production; do not add shared work twice",
        }
    )


def tuning_report(rows):
    """Public development-only ranking; screening is never confirmation."""
    from oeis_learn.generalization.progressive.selection import rank
    from oeis_learn.pipeline_validation.identity import seal

    ranking = rank(rows)
    return seal(
        {
            "version": "progressive-tuning-report/v1",
            "scientific_status": "UNMEASURED",
            "qualification_status": "PENDING",
            "ranking": ranking,
            "trials": [
                dict(
                    r,
                    learning_trial=i < 12,
                    size_trial=i >= 12,
                    performance_pruning_allowed=i not in (0, 1) and i < 12,
                    grace_tokens=1000000,
                )
                for i, r in enumerate(rows)
            ],
            "missing": ["actual_HIP_selected_shape_dropout_matrix", "independent_confirmation"],
        }
    )


def finalize_learning(journal, task, state_root, output):
    """Stdlib host receipt after owned process and cleanup have terminated.

    Worker progress is a lower bound; this receipt includes process startup and
    cleanup and all previous invocations. It never changes committed model state.
    """
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.milestones import milestone

    state_root = Path(state_root)
    binding = load(state_root / "binding.json")
    if seal(binding) != binding or binding["task"] != task:
        raise ValueError("original recovery binding required")
    lineage = load(state_root / "lineage.json") if (state_root / "lineage.json").is_file() else None
    if lineage and (seal(lineage) != lineage or lineage["task"] != task["identity"]):
        raise ValueError("changed learning ancestry")
    base = lineage["base"] if lineage else {"tokens": 0, "ancestor_cost_us": 0}
    remaining = journal.task_remaining(task)
    charged = base["tokens"] + task["payload"]["caps"]["tokens"] - remaining["tokens"]
    elapsed = task["payload"]["caps"]["elapsed_us"] - remaining["elapsed_us"]
    report = seal(
        {
            "version": "progressive-final-accounting/v1",
            "task": task["identity"],
            "milestone": milestone(
                charged,
                ancestor_tokens=base["tokens"],
                ancestor_cost_us=base["ancestor_cost_us"],
                new_cost_us=elapsed,
            ),
            "ancestor_checkpoint": lineage["parent"] if lineage else None,
            "remaining": remaining,
            "journal": journal.snapshot()["watermark"],
            "ancestor_cost_available": lineage.get("ancestor_cost_available", True)
            if lineage
            else task["payload"]["initializer"]["mode"] != "weights_only",
            "cost_scope": "all task invocations including startup and proven supervisor cleanup",
            "pending": journal.snapshot()["pending"],
        }
    )
    atomic_write(Path(output) / "accounting-final.json", report)
    return report


def generation_timing(value):
    """Collect only measured producer phase clocks, without adding overlap to wall time."""
    import math

    if value.get("version") != "discovery-candidates/v1" or seal(value) != value:
        raise ValueError("original sealed generation producer required")
    generated, queue, fit, capped, attempts = [], [], [], 0, 0

    def micros(seconds):
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid producer phase clock")
        return int(seconds * 1000000)

    for row in value["records"].values():
        for mode, cost in row["costs"].items():
            if mode not in {"correct", "shuffled", "absent", "template"}:
                continue
            for detail in cost.get("details", []):
                attempts += 1
                if "generation_seconds" in detail:
                    generated.append(micros(detail["generation_seconds"]))
                capped += detail.get("complete") is False
                clock = detail.get("fitting", {}).get("clock", {})
                if {"admitted", "dispatched", "completed"} <= clock.keys():
                    queue.append(micros(clock["dispatched"] - clock["admitted"]))
                    fit.append(micros(clock["completed"] - clock["dispatched"]))

    def summary(values):
        return {
            "observed": len(values),
            "unavailable": attempts - len(values),
            "total_us": sum(values),
            "maximum_us": max(values) if values else None,
            "unavailable_reason": None
            if len(values) == attempts and attempts
            else "producer phase clock absent",
        }

    tail = summary(generated)
    tail.update(
        attempts=attempts,
        token_cap_or_incomplete_events=capped,
        p95_us=sorted(generated)[max(0, (95 * len(generated) + 99) // 100 - 1)]
        if generated
        else None,
    )
    return seal(
        {
            "version": "progressive-phase-timing/v1",
            "producer": value["identity"],
            "generation_tail": tail,
            "queue": summary(queue),
            "fit": summary(fit),
            "wall_elapsed_us": value["elapsed_us"],
            "accounting": "phase clocks may overlap; do not add their totals to wall time",
        }
    )
