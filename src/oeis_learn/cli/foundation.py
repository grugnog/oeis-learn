"""Strict foundation CLI boundary; later pipeline commands fail closed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from oeis_learn.experiments.artifacts import load_json
from oeis_learn.experiments.models import (
    FoundationValidationError,
    validate_artifact,
    validate_shape,
)

_SUPPORTED_COMMANDS = frozenset({"preflight", "conformance"})
_UNSUPPORTED_COMMANDS = {
    "freeze-cohort": "requires cohort isolation and grouping (T020–T022/T028)",
    "build-pool": "requires generic generation and independent admission (T029–T036)",
    "train": "requires strict training and resource gates (T044–T047)",
    "resume": "requires checkpoint continuation (T041–T045)",
    "inspect": "requires checkpoint/run inspection (T041/T045)",
    "evaluate": "requires prefix-only selection and sealing (T024–T028)",
    "finalize": "requires immutable finalization (T026/T028)",
}


def _reject_unsupported(command: str) -> int:
    reason = _UNSUPPORTED_COMMANDS.get(command, "not implemented in this slice")
    print(f"ERROR: foundation {command!r}: {reason}", file=sys.stderr)
    return 1


def cmd_preflight(config: str, output: str, as_json: bool = False) -> int:
    from oeis_learn.cli.foundation_preflight import main

    return main(["--config", config, "--output", output])


def cmd_conformance(
    as_json: bool = False, schema: str | None = None, artifacts: list[str] | None = None
) -> int:
    """Validate diagnostic artifact shape/semantics, never claim execution success."""
    errors, count = [], 0
    try:
        extra_schema = json.loads(Path(schema).read_text()) if schema else None
        if not artifacts:
            raise FoundationValidationError("at least one artifact file is required")
        for name in artifacts:
            try:
                raw = load_json(Path(name).read_text(encoding="utf-8"))
                items = raw if isinstance(raw, list) else [raw]
                if not items:
                    raise FoundationValidationError("empty artifact list")
                for item in items:
                    validate_artifact(item)
                    if extra_schema is not None:
                        validate_shape(item, extra_schema)
                    count += 1
            except (OSError, ValueError) as exc:
                errors.append(f"{name}: {exc}")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    result = {
        "status": "FAIL" if errors else "PASS",
        "purpose": "diagnostic",
        "validated_artifacts": count,
        "executed_evidence": False,
        "errors": errors,
    }
    print(
        json.dumps(result)
        if as_json
        else f"{result['status']}: diagnostic validation; {count} artifacts"
        + (": " + "; ".join(errors) if errors else "")
    )
    return 2 if errors else 0


def dispatch(command: str, **kwargs) -> int:
    if command not in _SUPPORTED_COMMANDS:
        return _reject_unsupported(command)
    return {"preflight": cmd_preflight, "conformance": cmd_conformance}[command](
        **{k: v for k, v in kwargs.items() if v is not None}
    )
