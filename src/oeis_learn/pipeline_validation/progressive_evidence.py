"""Initial terminal evidence with independent statuses and safe producer failures."""

from pathlib import Path
import traceback
from oeis_learn.generalization.progressive.contracts import atomic_write, reference
from oeis_learn.pipeline_validation.identity import seal


def safe_failure(exc, *, producer, traceback_ref=None, private_values=()):
    message = str(exc)
    for value in sorted({str(v) for v in private_values if str(v)}, key=len, reverse=True):
        message = message.replace(value, "[redacted]")
    return dict(
        type=type(exc).__name__,
        message=message[:2048],
        code=getattr(exc, "code", "PRODUCER_FAILURE"),
        producer=producer,
        local_traceback=traceback_ref,
    )


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
        dict(
            version="discovery-result/v1",
            operation=operation,
            campaign=campaign,
            stage=stage,
            task=task,
            mechanical_status=mechanical_status,
            scientific_status=scientific_status,
            qualification_status=qualification_status,
            claim_status=claim_status,
            next_action=next_action,
            target_policy=target_policy,
            reason=reason,
            evidence="result.json",
            missing_evidence=list(missing),
            producer_error=producer_error,
            collector_errors=list(collector_errors),
            metrics=metrics
            or {"device": None, "timing": None, "unavailable_reason": "not_measured"},
        )
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
