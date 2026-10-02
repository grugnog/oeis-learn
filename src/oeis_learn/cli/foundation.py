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

_SUPPORTED_COMMANDS = frozenset(
    {
        "preflight",
        "conformance",
        "freeze-cohort",
        "evaluate",
        "finalize",
        "build-pool",
        "train",
        "resume",
        "inspect",
    }
)
_UNSUPPORTED_COMMANDS = {}


def _reject_unsupported(command: str) -> int:
    reason = _UNSUPPORTED_COMMANDS.get(command, "not implemented in this slice")
    print(f"ERROR: foundation {command!r}: {reason}", file=sys.stderr)
    return 1


def cmd_preflight(config: str, output: str, as_json: bool = False) -> int:
    from oeis_learn.cli.foundation_preflight import main

    return main(["--config", config, "--output", output])


def cmd_conformance(
    as_json: bool = False,
    schema: str | None = None,
    artifacts: list[str] | None = None,
    profile: str | None = None,
    output: str | None = None,
) -> int:
    """Run G1 with profile/output, or explicitly diagnostic artifact validation."""
    if profile is not None or output is not None:
        try:
            if not profile or not output or schema or artifacts:
                raise ValueError(
                    "--profile and --output are required together; cannot mix artifact diagnostics"
                )
            from oeis_learn.sandbox.conformance_runner import run_conformance

            result = run_conformance(Path(profile), Path(output))
            code = 0 if result["status"] == "PASS" else 5 if result["interruptions"] else 4
        except (OSError, ValueError) as exc:
            result = {"command": "conformance", "status": "FAIL", "errors": [str(exc)]}
            code = 2 if isinstance(exc, ValueError) else 5
        except (RuntimeError, EOFError, TimeoutError) as exc:
            result = {"command": "conformance", "status": "FAIL", "errors": [str(exc)]}
            code = 5
        print(json.dumps(result) if as_json else f"{result['status']}: {result}")
        return code
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
    return {
        "preflight": cmd_preflight,
        "conformance": cmd_conformance,
        "freeze-cohort": cmd_freeze_cohort,
        "evaluate": cmd_evaluate,
        "finalize": cmd_finalize,
        "build-pool": cmd_build_pool,
        "train": cmd_train,
        "resume": cmd_resume,
        "inspect": cmd_inspect,
    }[command](**{k: v for k, v in kwargs.items() if v is not None})


def _phase4_command(command, operation, as_json):
    from oeis_learn.evaluation.foundation_synthesis import EvaluationGateError, HardwareUnavailable

    try:
        report = operation()
        result = {"command": command, "status": "complete", **report}
        code = 0
    except HardwareUnavailable as exc:
        result, code = (
            {
                "command": command,
                "status": "hardware_unavailable",
                "error": str(exc),
                "qualified": False,
            },
            3,
        )
    except EvaluationGateError as exc:
        result, code = (
            {"command": command, "status": "gate_failed", "error": str(exc), "qualified": False},
            4,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        result, code = (
            {"command": command, "status": "invalid", "error": str(exc), "qualified": False},
            2,
        )
    except (OSError, RuntimeError, EOFError, TimeoutError) as exc:
        result, code = (
            {"command": command, "status": "incomplete", "error": str(exc), "qualified": False},
            5,
        )
    print(json.dumps(result) if as_json else f"{command}: {result}")
    return code


def cmd_freeze_cohort(source, config, output, as_json=False):
    from oeis_learn.evaluation.foundation_cohort import freeze_cohort

    def run():
        record = freeze_cohort(Path(source), Path(config), Path(output))
        return {
            "cohort_id": record["cohort_id"],
            "census": record["census"],
            "result_path": str(Path(output) / "manifest.json"),
        }

    return _phase4_command("freeze-cohort", run, as_json)


def cmd_evaluate(
    output,
    split,
    checkpoint=None,
    protocol=None,
    cohort=None,
    finalization=None,
    device="cpu",
    as_json=False,
):
    from oeis_learn.evaluation.foundation_synthesis import evaluate_foundation

    def run():
        if finalization:
            if checkpoint or protocol or cohort or split != "final":
                raise ValueError(
                    "finalization cannot be combined with overriding checkpoint/protocol/cohort or development split"
                )
            lock = load_json(Path(finalization).read_bytes())
            inputs = (lock["checkpoint_path"], lock["cohort_path"], lock["protocol_path"])
        else:
            if not checkpoint or not protocol or not cohort or split != "development":
                raise ValueError(
                    "development requires checkpoint/cohort/protocol; final requires a decision lock"
                )
            inputs = (checkpoint, cohort, protocol)
        return evaluate_foundation(
            *inputs, output, split=split, finalization=finalization, device=device
        )

    return _phase4_command("evaluate", run, as_json)


def cmd_finalize(run_dir, checkpoint, protocol, cohort, stopping_record, output, as_json=False):
    from oeis_learn.evaluation.finalization import create_finalization, final_output

    def run():
        lock = create_finalization(run_dir, checkpoint, protocol, cohort, stopping_record, output)
        return {
            "finalization_id": lock["finalization_id"],
            "result_path": str(output),
            "evaluation_output": str(final_output(lock)),
        }

    return _phase4_command("finalize", run, as_json)


def cmd_build_pool(config, cohort, output, as_json=False):
    import yaml
    from oeis_learn.data.program_pool import build_pool
    from oeis_learn.data.program_admission import AdmissionGateError

    try:
        result = build_pool(config, cohort, output)
        code = (
            0
            if result["status"] == "complete"
            else 5
            if result["reason"] == "wall_budget_exhausted"
            else 4
        )
    except AdmissionGateError as exc:
        result, code = (
            {
                "command": "build-pool",
                "status": "gate_failed",
                "error": str(exc),
                "qualified": False,
            },
            4,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        result, code = (
            {
                "command": "build-pool",
                "status": "invalid_input",
                "error": str(exc),
                "qualified": False,
            },
            2,
        )
    except (OSError, RuntimeError, EOFError, TimeoutError) as exc:
        result, code = (
            {
                "command": "build-pool",
                "status": "interrupted",
                "error": str(exc),
                "qualified": False,
            },
            5,
        )
    print(json.dumps(result) if as_json else f"{result['status']}: {result}")
    return code


def cmd_train(
    config,
    pool,
    run_dir,
    device,
    diagnostic=False,
    diagnostic_small_host=False,
    prepare_only=False,
    seed=20260913,
    stop_after_update=None,
    as_json=False,
):
    from oeis_learn.rl.foundation_sft import prepare_run
    from oeis_learn.tracking.foundation_controller import run_training
    from oeis_learn.evaluation.foundation_synthesis import HardwareUnavailable

    def run():
        if device == "cuda":
            import torch

            if not torch.cuda.is_available() or not torch.version.hip:
                raise HardwareUnavailable("HIP GPU unavailable; no CPU fallback")
        record = prepare_run(
            config,
            pool,
            run_dir,
            device=device,
            diagnostic=diagnostic,
            seed=seed,
            diagnostic_small_host=diagnostic_small_host,
        )
        if prepare_only:
            return dict(
                status="prepared",
                qualified=False,
                run_id=record["run_id"],
                result_path=str(Path(run_dir) / "run.json"),
            )
        return run_training(run_dir, stop_after=stop_after_update)

    return _phase4_command("train", run, as_json)


def cmd_resume(run_dir, checkpoint=None, stop_after_update=None, as_json=False):
    from oeis_learn.tracking.foundation_controller import run_training
    from oeis_learn.tracking.run_manager import inspect_foundation_run

    def run():
        if checkpoint is not None:
            selected = inspect_foundation_run(run_dir)["checkpoint"]
            if selected is None or Path(checkpoint).resolve() != Path(selected).resolve():
                raise ValueError("resume must select newest valid checkpoint; override forbidden")
        return run_training(run_dir, resume=True, stop_after=stop_after_update)

    return _phase4_command("resume", run, as_json)


def cmd_inspect(run_dir, as_json=False):
    from oeis_learn.tracking.run_manager import inspect_foundation_run

    return _phase4_command("inspect", lambda: inspect_foundation_run(run_dir), as_json)
