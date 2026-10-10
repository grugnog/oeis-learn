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
        "qualify-repairs",
        "activate-qualified",
        "qualify-learning",
        "qualify-discovery",
        "discover",
        "check-certificate",
        "learning-train",
        "learning-search",
        "conformance",
        "freeze-cohort",
        "evaluate",
        "finalize",
        "build-pool",
        "train",
        "resume",
        "inspect",
        "measure",
        "census",
        "prepare-release",
        "prepare-measurement",
        "qualify-baseline",
        "diagnose-overflow",
        "coordinated-smoke",
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
        "qualify-repairs": cmd_qualify_repairs,
        "activate-qualified": cmd_activate_qualified,
        "qualify-learning": cmd_qualify_learning,
        "qualify-discovery": cmd_qualify_discovery,
        "discover": cmd_discover,
        "check-certificate": cmd_check_certificate,
        "learning-train": cmd_learning_train,
        "learning-search": cmd_learning_search,
        "conformance": cmd_conformance,
        "freeze-cohort": cmd_freeze_cohort,
        "evaluate": cmd_evaluate,
        "finalize": cmd_finalize,
        "build-pool": cmd_build_pool,
        "train": cmd_train,
        "resume": cmd_resume,
        "inspect": cmd_inspect,
        "measure": cmd_measure,
        "census": cmd_census,
        "prepare-release": cmd_prepare_release,
        "prepare-measurement": cmd_prepare_measurement,
        "qualify-baseline": cmd_qualify_baseline,
        "diagnose-overflow": cmd_diagnose_overflow,
        "coordinated-smoke": cmd_coordinated_smoke,
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


def cmd_qualify_repairs(output, as_json=False):
    from oeis_learn.evaluation.repair_gate import run_qualification

    return _phase4_command("qualify-repairs", lambda: run_qualification(output), as_json)


def cmd_activate_qualified(
    config, gate_report, profiles, as_json=False, learning_gate_report=None, diagnostic=False
):
    from oeis_learn.experiments.config import load_qualified_config

    def operation():
        def resolve(digest):
            return load_json(
                (Path(profiles) / (digest.removeprefix("sha256:") + ".json")).read_bytes()
            )

        services = load_qualified_config(
            config,
            g7_report=load_json(Path(gate_report).read_bytes()),
            evidence_root=Path(gate_report).parent,
            resolve=resolve,
            g8_report=None
            if learning_gate_report is None
            else load_json(Path(learning_gate_report).read_bytes()),
            g8_evidence_root=None
            if learning_gate_report is None
            else Path(learning_gate_report).parent,
            diagnostic=diagnostic,
        )
        return {
            "g8_sha256": services.g8_sha256,
            "diagnostic": services.diagnostic,
            "contract_sha256": services.contract_sha256,
            "g7_sha256": services.g7_sha256,
            "capabilities": list(services.capabilities),
        }

    return _phase4_command("activate-qualified", operation, as_json)


def cmd_qualify_learning(output, hardware=False, as_json=False):
    from oeis_learn.evaluation.learning_gate import run_qualification

    report = run_qualification(output, hardware=hardware)
    print(
        json.dumps(report)
        if as_json
        else f"G8 software={report['software_status']} hardware={report['hardware_status']}"
    )
    return (
        0
        if report["software_status"] == "PASS"
        and (not hardware or report["hardware_status"] == "PASS")
        else 4
    )


def _learning_services(config, gate_report, learning_gate_report, profiles, diagnostic):
    from oeis_learn.experiments.config import load_qualified_config

    root = Path(profiles)

    def resolve(key):
        return load_json((root / (key[7:] + ".json")).read_bytes())

    return load_qualified_config(
        config,
        g7_report=load_json(Path(gate_report).read_bytes()),
        evidence_root=Path(gate_report).parent,
        g8_report=load_json(Path(learning_gate_report).read_bytes()),
        g8_evidence_root=Path(learning_gate_report).parent,
        resolve=resolve,
        diagnostic=diagnostic,
    ), resolve


def cmd_learning_train(
    config,
    gate_report,
    learning_gate_report,
    profiles,
    archive,
    output,
    updates=1,
    seed=20260913,
    device="cuda",
    diagnostic=False,
    resume=None,
    as_json=False,
):
    """Bounded real training. Diagnostic CPU runs never advertise qualification."""

    def operation():
        import torch
        from oeis_learn.discovery.program_archive import ProgramArchive
        from oeis_learn.experiments.learning_profiles import load_holdouts
        from oeis_learn.rl.replay import FrozenReplay
        from oeis_learn.rl.sft_trainer import replay_update
        from oeis_learn.tracking.training_checkpoint import (
            QualifiedCheckpointStore,
            capture_learning_state,
            restore_learning_state,
        )
        from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
        from oeis_learn.sandbox.pipeline import runtime_identity

        services, resolve = _learning_services(
            config, gate_report, learning_gate_report, profiles, diagnostic
        )
        contract = load_json(Path(config).read_bytes())
        if (
            type(updates) is not int
            or not 1 <= updates <= 100
            or type(seed) is not int
            or not 0 <= seed < 2**63
        ):
            raise ValueError("bounded updates and named seed required")
        if "replay" not in services.capabilities:
            raise ValueError("explicit replay capability required")
        holdouts = load_holdouts(resolve(contract["profiles"]["admission"]))
        program_archive = ProgramArchive(archive, holdouts=holdouts)
        torch.manual_seed(seed)
        encoder, decoder, counts = services.models(device)
        model_profile = services._settings("learning")["model_profile"]
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()), lr=0.0003, weight_decay=0.01
        )
        replay = FrozenReplay(program_archive)
        generic = [
            key
            for key, value in program_archive.training_view(replay.snapshot)
            if value["entry"]["origin"] == "generic_sample"
        ]
        if not generic:
            raise ValueError("verified generic training complement is empty")
        runtime = digest(runtime_identity("rust"))
        store = QualifiedCheckpointStore(output, free_floor_bytes=0 if diagnostic else 100 << 30)
        from oeis_learn.tracking.budget_ledger import BudgetLedger
        from oeis_learn.experiments.artifacts import atomic_write, ArtifactPath, canonical_bytes

        run_inputs = {
            "contract": services.contract_sha256,
            "profile": model_profile["identity_sha256"],
            "archive_root": replay.snapshot["root_sha256"],
            "seed": seed,
            "budget_ns": 300_000_000_000,
        }
        atomic_write(
            ArtifactPath(Path(output), "learning-inputs.json"), canonical_bytes(run_inputs)
        )
        ledger = BudgetLedger(output, digest(run_inputs), run_inputs["budget_ns"])
        ledger.transition("recovery" if resume else "training")
        sampler_state = {"seed": seed, "cursor": 0}
        previous = None
        completed = 0
        policy = None
        objective = model_profile["objective"]
        runner = services.runner(Path(output) / "runtime")
        try:
            if objective != "sft":
                from oeis_learn.rl.trainer import EgcaGrpoTrainer
                from oeis_learn.curriculum.scheduler import CurriculumScheduler

                policy = EgcaGrpoTrainer(
                    encoder,
                    decoder,
                    CurriculumScheduler(),
                    wasm_runner=runner,
                    objective=objective,
                    seed=seed,
                    max_program_length=1023,
                    device=torch.device(device),
                )
                optimizer = policy.optimizer
            if resume:
                manifest, base, state = store.load_learning(
                    resume,
                    expected_contract=services.contract_sha256,
                    expected_profile=model_profile["identity_sha256"],
                    expected_runtime=runtime,
                )
                sampler_state, _ = restore_learning_state(
                    state,
                    encoder,
                    decoder,
                    optimizer,
                    replay=replay,
                    reference=None if policy is None else policy.reference,
                    curriculum=None if policy is None else policy.scheduler,
                )
                if (
                    base["ledger_sequence"] > ledger.state["sequence"]
                    or base["charged_budget_ns"] > ledger.state["charged_budget_ns"]
                ):
                    raise ValueError("checkpoint exceeds the durable budget ledger")
                previous = manifest["qualified"]["identity_sha256"]
                completed = base["completed_update"]
                if policy is not None:
                    policy.rollout_cursor = sampler_state["policy_cursor"]
            metrics = []
            ledger.transition("training")
            for _ in range(updates):
                if ledger.remaining_ns() <= 0:
                    raise RuntimeError("bounded training session deadline exhausted")
                cursor = sampler_state["cursor"]
                generic_batch = [generic[(cursor * 3 + i) % len(generic)] for i in range(3)]
                result = replay_update(
                    encoder, decoder, optimizer, replay, generic=generic_batch, device=device
                )
                if policy is not None:
                    from oeis_learn.rl.trainer import TrainingPrompt

                    value = program_archive.get(generic[cursor % len(generic)])
                    result["policy"] = policy.train_step_for_prompt(
                        TrainingPrompt(
                            prompt_id="training-" + generic[cursor % len(generic)][7:23],
                            terms=tuple(int(v) for v in value["entry"]["conditioning_values"]),
                            curriculum_stage=1,
                        ),
                        group_size=4,
                    )
                    sampler_state["policy_cursor"] = policy.rollout_cursor
                sampler_state["cursor"] += 1
                completed += 1
                metrics.append(result)
                state = capture_learning_state(
                    encoder,
                    decoder,
                    optimizer,
                    sampler=sampler_state,
                    prefetch={"archive_root_sha256": replay.snapshot["root_sha256"], "next": []},
                    replay=replay,
                    reference=None if policy is None else policy.reference,
                    curriculum=None if policy is None else policy.scheduler,
                )
                budget = ledger.transition("checkpoint")
                path = store.save_learning(
                    state,
                    contract_sha256=services.contract_sha256,
                    profile_sha256=model_profile["identity_sha256"],
                    runtime_sha256=runtime,
                    ledger_sequence=budget["sequence"],
                    charged_budget_ns=budget["charged_budget_ns"],
                    completed_update=completed,
                    previous=previous,
                )
                previous = load_json(path.read_bytes())["qualified"]["identity_sha256"]
                ledger.transition("training")
            ledger.transition("paused")
            return {
                "command": "learning-train",
                "status": "complete",
                "qualified": not diagnostic,
                "objective": objective,
                "parameters": counts,
                "updates": updates,
                "checkpoint": str(path),
                "metrics": metrics,
            }
        except BaseException:
            ledger.transition("failed")
            raise
        finally:
            runner.close()

    return _phase4_command("learning-train", operation, as_json)


def cmd_learning_search(
    config,
    gate_report,
    learning_gate_report,
    profiles,
    archive,
    checkpoint,
    prompt,
    output,
    portfolio="portfolio32",
    device="cuda",
    diagnostic=False,
    as_json=False,
):
    def operation():
        from oeis_learn.discovery.program_archive import ProgramArchive
        from oeis_learn.experiments.learning_profiles import load_holdouts
        from oeis_learn.evaluation.search import SearchSession, ModelProposer
        from oeis_learn.tracking.training_checkpoint import QualifiedCheckpointStore
        from oeis_learn.experiments.artifacts import (
            compute_canonical_digest as digest,
            canonical_bytes,
            atomic_write,
            ArtifactPath,
        )
        from oeis_learn.sandbox.pipeline import runtime_identity

        services, resolve = _learning_services(
            config, gate_report, learning_gate_report, profiles, diagnostic
        )
        if "search" not in services.capabilities:
            raise ValueError("search capability disabled")
        contract = load_json(Path(config).read_bytes())
        model = services._settings("learning")["model_profile"]
        encoder, decoder, counts = services.models(device)
        store = QualifiedCheckpointStore(Path(checkpoint).parent, free_floor_bytes=0)
        _, _, state = store.load_learning(
            checkpoint,
            expected_contract=services.contract_sha256,
            expected_profile=model["identity_sha256"],
            expected_runtime=digest(runtime_identity("rust")),
        )
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        visible = load_json(Path(prompt).read_bytes())
        if set(visible) != {"values"}:
            raise ValueError("proposer accepts visible values only")
        program_archive = ProgramArchive(
            archive, holdouts=load_holdouts(resolve(contract["profiles"]["admission"]))
        )
        snapshot = state["replay"]["snapshot"]
        session = SearchSession.with_archive(
            visible["values"], program_archive, snapshot, portfolio=portfolio
        )
        with services.runner(Path(output) / "runtime") as runner:
            report = session.run(ModelProposer(encoder, decoder), runner)
        report["qualified"] = not diagnostic
        report["parameters"] = counts
        atomic_write(ArtifactPath(Path(output), "search.json"), canonical_bytes(report))
        return report

    return _phase4_command("learning-search", operation, as_json)


def cmd_qualify_discovery(output, native=False, as_json=False):
    from oeis_learn.evaluation.discovery_gate import run_qualification

    result = run_qualification(output, native=native)
    print(json.dumps(result) if as_json else str(result))
    return 0 if result["status"] == "PASS" else 3 if result["status"] == "PENDING" else 4


def cmd_check_certificate(certificate, profile, output, as_json=False):
    from oeis_learn.discovery.certificate_checker import check_certificate
    from oeis_learn.sandbox.logical_ir import parse_logical, profile_identity
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
    from oeis_learn.tracking.foundation_metrics import replace_json

    def action():
        data = load_json(Path(certificate).read_bytes())
        selected = load_json(Path(profile).read_bytes())
        modes = {parse_logical(p["source"]).profile for p in data["statement"]["programs"]}
        if len(modes) != 1 or digest(selected) != digest(profile_identity(next(iter(modes)))):
            raise ValueError("certificate/profile mismatch")
        result = check_certificate(data)
        replace_json(Path(output) / "certificate-check.json", result)
        if result["result"] != "verified":
            from oeis_learn.evaluation.foundation_synthesis import EvaluationGateError

            raise EvaluationGateError(
                result["result"] + ": " + result.get("reason", "certificate not verified")
            )
        return result

    return _phase4_command("check-certificate", action, as_json)


def cmd_discover(archive, discovery_gate_report, output, holdouts=None, as_json=False):
    from oeis_learn.evaluation.readiness import require_discovery_readiness
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.discovery.pipeline import run_program_discovery
    from oeis_learn.tracking.foundation_metrics import replace_json

    def action():
        report = load_json(Path(discovery_gate_report).read_bytes())
        require_discovery_readiness(report, evidence_root=Path(discovery_gate_report).parent)
        from oeis_learn.experiments.learning_profiles import load_holdouts

        selected = (
            None if holdouts is None else load_holdouts(load_json(Path(holdouts).read_bytes()))
        )
        result = run_program_discovery(ProgramArchive(archive, holdouts=selected))
        replace_json(Path(output) / "discovery.json", result)
        return result

    return _phase4_command("discover", action, as_json)


def cmd_measure(manifest, output, dry_run=False, as_json=False):
    from oeis_learn.evaluation.experiment_runner import measure

    try:
        report = measure(
            load_json(Path(manifest).read_bytes()),
            Path(output),
            dry_run=dry_run,
            source_root=Path(manifest).parent,
        )
        code = (
            3
            if report["status"] == "hardware_unavailable"
            else 5
            if report["status"] in ("interrupted", "failed_gate")
            else 0
        )
    except (ValueError, OSError) as exc:
        report = dict(command="measure", status="invalid_input", qualified=False, reason=str(exc))
        code = 2
    print(json.dumps(report) if as_json else f"{report['status']}: {report}")
    return code


def cmd_census(source, output, as_json=False):
    from oeis_learn.evaluation.source_policy import analyze_sequence_cohort
    from oeis_learn.evaluation.foundation_cohort import write_json
    from oeis_learn.experiments.artifacts import compute_file_hash

    try:
        path = Path(source).resolve(strict=True)
        rows = load_json(path.read_bytes())
        if not isinstance(rows, list):
            raise ValueError("source census requires a record array")
        report = dict(
            command="census",
            status="complete",
            source_path=str(path),
            source_sha256=compute_file_hash(path),
            qualified=False,
            **analyze_sequence_cohort(rows),
        )
        write_json(Path(output), "census.json", report)
        code = 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = dict(command="census", status="invalid_input", qualified=False, reason=str(exc))
        code = 2
    print(json.dumps(report) if as_json else f"{report['status']}: {report}")
    return code


def cmd_prepare_release(manifest, config, output, as_json=False):
    from oeis_learn.evaluation.release_source import prepare_release

    return _phase4_command(
        "prepare-release", lambda: prepare_release(manifest, config, output), as_json
    )


def cmd_prepare_measurement(
    archive,
    cohort,
    gate_report,
    learning_gate_report,
    image,
    output,
    baseline_report,
    baseline_checkpoint,
    holdouts=None,
    as_json=False,
):
    from oeis_learn.evaluation.measurement_inputs import prepare_measurement
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.experiments.learning_profiles import load_holdouts

    selected = None if holdouts is None else load_holdouts(load_json(Path(holdouts).read_bytes()))
    return _phase4_command(
        "prepare-measurement",
        lambda: prepare_measurement(
            ProgramArchive(archive, holdouts=selected),
            cohort,
            gate_report,
            learning_gate_report,
            Path(gate_report).parent,
            image,
            output,
            baseline_report=baseline_report,
            baseline_checkpoint=baseline_checkpoint,
        ),
        as_json,
    )


def cmd_qualify_baseline(checkpoint, run_dir, preflight, suites, output, as_json=False):
    from oeis_learn.evaluation.baseline_gate import assemble_baseline_bundle

    return _phase4_command(
        "qualify-baseline",
        lambda: assemble_baseline_bundle(
            checkpoint, run_dir, preflight, load_json(Path(suites).read_bytes()), output
        ),
        as_json,
    )


def cmd_diagnose_overflow(
    strata, output, provenance=None, archive=None, holdouts=None, development=None, as_json=False
):
    from oeis_learn.evaluation.overflow_diagnostics import diagnose_overflow
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.experiments.learning_profiles import load_holdouts

    def action():
        policy = None if holdouts is None else load_holdouts(load_json(Path(holdouts).read_bytes()))
        selected = None if archive is None else ProgramArchive(archive, holdouts=policy)
        return diagnose_overflow(
            load_json(Path(strata).read_bytes()),
            output,
            archive=selected,
            development=development,
            provenance_packets=None
            if provenance is None
            else load_json(Path(provenance).read_bytes()),
        )

    return _phase4_command("diagnose-overflow", action, as_json)


def cmd_coordinated_smoke(config, pool, image, output, seconds=600, resume_run=None, as_json=False):
    from oeis_learn.evaluation.coordinated_smoke import coordinated_smoke

    return _phase4_command(
        "coordinated-smoke",
        lambda: coordinated_smoke(
            config, pool, image, output, seconds=seconds, resume_run=resume_run
        ),
        as_json,
    )
