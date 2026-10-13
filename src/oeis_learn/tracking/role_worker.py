"""Fixed internal role entry points. Inputs are role-specific data, never commands."""

from pathlib import Path
import sys
from oeis_learn.experiments.artifacts import (
    load_json,
    canonical_bytes,
    atomic_write,
    ArtifactPath,
    compute_canonical_digest as digest,
)


def candidate(inputs, output):
    from oeis_learn.sandbox.pipeline import Runtime, Limits

    request = load_json((inputs / "request.json").read_bytes())
    if set(request) != {"source", "indices", "limits", "request_id"}:
        raise ValueError("closed worker request required")
    return (
        Runtime(backend="rust")
        .evaluate(request["source"], request["indices"], Limits(**request["limits"]))
        .to_dict()
    )


def serve(inputs, output):
    """Persistent native runtime/cache; the host owns every candidate kill deadline."""
    import time
    from oeis_learn.sandbox.pipeline import Runtime, Limits

    runtime = Runtime(backend="rust")
    atomic_write(ArtifactPath(output, "ready.json"), canonical_bytes({"ready": True}))
    seen = set()
    while True:
        for path in sorted(inputs.glob("*.json")):
            if path.name in seen:
                continue
            if path.is_symlink() or path.stat().st_size > 1 << 20:
                raise ValueError("bounded worker request required")
            request = load_json(path.read_bytes())
            if set(request) != {"source", "indices", "limits", "request_id"}:
                raise ValueError("closed worker request required")
            seen.add(path.name)
            value = runtime.evaluate(
                request["source"], request["indices"], Limits(**request["limits"])
            )
            atomic_write(ArtifactPath(output, path.name), canonical_bytes(value.to_dict()))
        seen.intersection_update(p.name for p in inputs.glob("*.json"))
        time.sleep(0.001)


def load_models(inputs, settings):
    import torch
    from oeis_learn.experiments.pilot_models import construct_pilot_models

    torch.set_num_threads(4)
    device = settings["device"]
    if device == "cuda" and (not torch.cuda.is_available() or not torch.version.hip):
        raise RuntimeError("actual HIP device required; no CPU fallback")
    torch.manual_seed(settings["seed"])
    encoder, decoder = construct_pilot_models(settings["profile"], device)
    if (inputs / "state.pt").exists():
        state = load_state(inputs, device)
        if state["profile"] != settings["profile"] or state["seed"] != settings["seed"]:
            raise ValueError("pilot model profile/seed mismatch on resume")
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
    return encoder, decoder


def probe(inputs, output):
    import torch
    from oeis_learn.rl.sft_trainer import supervised_step
    from oeis_learn.decoder.program_codec import BOS_ID, PAD_ID
    from oeis_learn.evaluation.engineering import check_device_resources

    settings = load_json((inputs / "settings.json").read_bytes())
    encoder, decoder = load_models(inputs, settings)
    if (inputs / "baseline.json").exists():
        from oeis_learn.evaluation.baseline_gate import require_baseline
        from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint

        name = load_json((inputs / "baseline-checkpoint.json").read_bytes())["name"]
        if Path(name).name != name:
            raise ValueError("confined checkpoint filename required")
        checkpoint = load_foundation_checkpoint(
            inputs / "gates/baseline/checkpoints" / name, device="cpu", include_payload=True
        )
        require_baseline(
            load_json((inputs / "baseline.json").read_bytes()),
            checkpoint,
            evidence_root=inputs / "gates/baseline",
        )
    if (inputs / "gate7.json").exists():
        from oeis_learn.evaluation.repair_gate import require_g7
        from oeis_learn.evaluation.learning_gate import require_g8

        require_g7(
            load_json((inputs / "gate7.json").read_bytes()), evidence_root=inputs / "gates/g7"
        )
        require_g8(
            load_json((inputs / "gate8.json").read_bytes()), evidence_root=inputs / "gates/g8"
        )
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=0.0003
    )
    rows = load_json((inputs / "examples.json").read_bytes())["rows"]
    records = []
    for index in range(3):
        row = rows[index]
        records.append(
            supervised_step(
                encoder,
                decoder,
                optimizer,
                [[int(v) for v in row["entry"]["conditioning_values"]]],
                torch.tensor([[BOS_ID] + row["tokens"]], device=settings["device"]),
                device=settings["device"],
                pad_id=PAD_ID,
            )
        )
        check_device_resources(settings["device"])
    return dict(
        status="PASS",
        device=settings["device"],
        hip=torch.version.hip,
        updates=records,
        actual_parameter_device=str(next(decoder.parameters()).device),
        code_sha256=__import__(
            "oeis_learn.evaluation.experiment_runner", fromlist=["implementation_identity"]
        ).implementation_identity(),
    )


def train(inputs, output):
    import torch
    import time
    from oeis_learn.rl.sft_trainer import supervised_step
    from oeis_learn.decoder.program_codec import BOS_ID, PAD_ID
    from oeis_learn.tracking.foundation_metrics import EventLog
    from oeis_learn.evaluation.engineering import check_device_resources

    settings = load_json((inputs / "settings.json").read_bytes())
    encoder, decoder = load_models(inputs, settings)
    examples = load_json((inputs / "examples.json").read_bytes())
    rows = examples.get("training_rows", examples["rows"])
    validation = examples.get("validation_rows", [])
    if not rows:
        raise ValueError("empty verified training fixture")
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=0.0003, weight_decay=0.01
    )
    runner = __import__(
        "oeis_learn.tracking.role_coordinator", fromlist=["QueueRunner"]
    ).QueueRunner(output / "runtime-queue")
    log = EventLog(output / "updates")
    start, completed = time.monotonic(), 0
    if (inputs / "state.pt").exists():
        state = load_state(inputs, settings["device"])
        optimizer.load_state_dict(state["optimizer"])
        completed = state["completed"]
        import random, numpy as np

        random.setstate(state["python_rng"])
        n = state["numpy_rng"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(state["torch_rng"].cpu())
        if settings["device"] == "cuda":
            torch.cuda.set_rng_state_all(state["device_rng"])
    policy = None
    if settings["profile"]["objective"] == "grpo":
        from oeis_learn.rl.trainer import EgcaGrpoTrainer
        from oeis_learn.curriculum.scheduler import CurriculumScheduler

        policy = EgcaGrpoTrainer(
            encoder,
            decoder,
            CurriculumScheduler(),
            wasm_runner=runner,
            objective="grpo",
            seed=settings["seed"],
            max_program_length=1023,
            device=torch.device(settings["device"]),
        )
        if (inputs / "state.pt").exists():
            policy.optimizer.load_state_dict(state["optimizer"])
            policy.reference.encoder.load_state_dict(state["reference_encoder"])
            policy.reference.decoder.load_state_dict(state["reference_decoder"])
            policy.rollout_cursor = state["policy_cursor"]
            policy.scheduler.load_state_dict(state["curriculum"])
        optimizer = policy.optimizer
    # Checkpointing is inside the same charged segment; reserve measured work time
    # by stopping updates before the segment boundary, never resetting its clock.
    losses, validation_windows = [], []
    if (inputs / "state.pt").exists():
        rows = state.get("replay_rows", rows)
        validation_windows = state.get("validation_windows", [])
    while completed < settings["update_cap"] and time.monotonic() - start < settings["seconds"] - 5:
        row = rows[completed % len(rows)]
        values = [int(v) for v in row["entry"]["conditioning_values"]]
        if policy is None:
            batch = rows[completed % len(rows) : completed % len(rows) + 4]
            tokens = [[BOS_ID] + r["tokens"] for r in batch]
            padded = torch.full(
                (len(tokens), max(map(len, tokens))),
                PAD_ID,
                device=settings["device"],
                dtype=torch.long,
            )
            for i, token in enumerate(tokens):
                padded[i, : len(token)] = torch.tensor(token, device=settings["device"])
            result = supervised_step(
                encoder,
                decoder,
                optimizer,
                [[int(v) for v in r["entry"]["conditioning_values"]] for r in batch],
                padded,
                device=settings["device"],
                pad_id=PAD_ID,
            )
        else:
            from oeis_learn.rl.trainer import TrainingPrompt

            result = policy.train_step_for_prompt(
                TrainingPrompt(
                    prompt_id=f"pilot-{completed}", terms=tuple(values), curriculum_stage=1
                ),
                group_size=4,
            )
        completed += 1
        log.append(str(completed), result)
        losses.append(float(result["loss"]))
        check_device_resources(settings["device"])
        if completed % 100 == 0 and validation:
            validation_windows.append(
                validation_loss(encoder, decoder, validation, settings["device"])
            )
        if settings["profile"]["self_training"] and completed % 100 == 0:
            rows = self_training_round(
                rows,
                encoder,
                decoder,
                runner,
                completed,
                log,
                holdouts=examples.get("holdouts"),
                archive_root=output / "replay-archive",
            )
    import random, numpy as np

    n = np.random.get_state()
    state = dict(
        version="pilot-state/v1",
        seed=settings["seed"],
        scheduler=None,
        scaler=None,
        python_rng=random.getstate(),
        numpy_rng=[n[0], n[1].tolist(), n[2], n[3], n[4]],
        encoder=encoder.state_dict(),
        decoder=decoder.state_dict(),
        optimizer=optimizer.state_dict(),
        completed=completed,
        profile=settings["profile"],
        replay_rows=rows,
        validation_windows=validation_windows,
        torch_rng=torch.get_rng_state(),
        device_rng=torch.cuda.get_rng_state_all() if settings["device"] == "cuda" else [],
    )
    if policy is not None:
        state.update(
            reference_encoder=policy.reference.encoder.state_dict(),
            reference_decoder=policy.reference.decoder.state_dict(),
            policy_cursor=policy.rollout_cursor,
            curriculum=policy.scheduler.state_dict(),
        )
    save_state(output, state)
    from oeis_learn.experiments.artifacts import compute_file_hash

    return dict(
        status="complete",
        completed_updates=completed,
        state_sha256=compute_file_hash(output / "state.pt"),
        last_loss=losses[-1] if losses else None,
        elapsed_seconds=time.monotonic() - start,
        validation_windows=validation_windows,
        training_loss_windows=[sum(losses[i : i + 100]) / 100 for i in (0, 100, 200, 300)]
        if len(losses) >= 400
        else None,
    )


def admit_training_hypothesis(candidate, values, runner, archive_root, *, holdouts, request_id):
    """Count/reuse a search result only after shared, held-out archive admission."""
    from oeis_learn.data.structural_splits import StructuralHoldouts
    from oeis_learn.discovery.program_archive import ProgramArchive

    h = holdouts or {}
    policy = StructuralHoldouts(
        seed=h.get("seed", 20260913),
        reserved_structures=frozenset(h.get("reserved_structures", [])),
        reserved_compositions=tuple(
            frozenset(s) for s in h.get("reserved_compositions", [["i64.xor", "i64.mul"]])
        ),
        reserved_parameter_bits=h.get("reserved_parameter_bits", 128),
        reserved_prefixes=frozenset(h.get("reserved_prefixes", [])),
    )
    source = candidate["source"]
    if not policy.decision(source, values[:20])[0]:
        return None
    packet = candidate["provenance"]["hypothesis"]
    archive = ProgramArchive(archive_root, holdouts=policy)
    try:
        key = archive.admit(
            source,
            values[:20],
            values,
            origin="training_prefix_hypothesis",
            provenance={"search": packet},
            runner=runner,
            request_id=request_id,
            model_sha256=digest([packet["encoder_sha256"], packet["decoder_sha256"]]),
        )
    except ValueError:
        # A visible fit need not match the independent training continuation.
        return None
    item = archive.get(key)
    return dict(key=key, **item) if item["entry"]["training_eligible"] else None


def self_training_round(
    rows, encoder, decoder, runner, counter, log, *, holdouts=None, archive_root
):
    """Search only fixture training prefixes; continuation verification precedes SFT use."""
    from oeis_learn.evaluation.search import SearchSession, ModelProposer

    proposer = ModelProposer(encoder, decoder)
    admitted = []
    seen = {r["source"] for r in rows}
    for row in rows[:4]:
        values = [int(v) for v in row["entry"]["outputs"]]
        report = SearchSession(values[:20], portfolio="sampling32", seed=7001 + counter).run(
            proposer, runner
        )
        for result in report["results"]:
            source = result.get("source")
            if result.get("status") != "visible_fit" or not source or source in seen:
                continue
            item = admit_training_hypothesis(
                result,
                values,
                runner,
                archive_root,
                holdouts=holdouts,
                request_id=f"selftraining-admit-{counter}-{len(seen)}",
            )
            if item is None:
                continue
            seen.add(source)
            admitted.append(item)
    log.append(
        "selftraining-" + str(counter),
        dict(admitted=len(admitted), sources_sha256=digest(admitted), training_only=True),
    )
    # Frozen next-round data, 25% replay at most; never evaluator targets.
    if not admitted:
        return rows
    return rows + admitted[: max(1, len(rows) // 3)]


def encoder_collision_count(encoder, device):
    """Probe the selected encoder's input features before its learned projection."""
    import torch
    from oeis_learn.encoder.integer_bytes import IntegerByteEncoder, encode_sequences
    from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder

    collisions = 0
    with torch.no_grad():
        for value in ((1 << 54), (1 << 100), -(1 << 100)):
            pair = [[value] * 20, [value + 1] * 20]
            if isinstance(encoder, IntegerByteEncoder):
                features = encode_sequences(pair)
                collision = torch.equal(features.signs[0], features.signs[1]) and torch.equal(
                    features.magnitudes[0], features.magnitudes[1]
                )
            elif isinstance(encoder, TriStreamEncoder):
                streams = [
                    encoder.s1_magnitude(pair, device=device),
                    encoder.s2_modulo(pair, device=device),
                    encoder.s3_diff_padic(pair, device=device),
                ]
                collision = all(torch.equal(s[0], s[1]) for s in streams)
            else:
                raise ValueError("unsupported diagnostic encoder")
            collisions += int(collision)
    return collisions


def diagnostic(inputs, output):
    from oeis_learn.experiments.pilot_models import pilot_profile
    from oeis_learn.data.structural_splits import structural_signature

    settings = load_json((inputs / "settings.json").read_bytes())
    settings["profile"] = pilot_profile()
    atomic_write(ArtifactPath(output, "diagnostic-settings.json"), canonical_bytes(settings))
    result = train(inputs, output)
    rows = load_json((inputs / "examples.json").read_bytes())["rows"]
    encoder, decoder = load_models(output, settings)
    collisions = encoder_collision_count(encoder, settings["device"])
    observed = dict(
        encoder_collisions=collisions,
        updates=result["completed_updates"],
        diverse_programs=len(
            {
                structural_signature(r["source"])
                for r in load_json((inputs / "examples.json").read_bytes()).get(
                    "training_rows", rows
                )
            }
        ),
    )
    windows = result["validation_windows"]
    if len(windows) >= 4:
        observed["plateau_improvements"] = [
            (windows[i - 1] - windows[i]) / max(abs(windows[i - 1]), 1e-12) for i in range(1, 4)
        ]
    if result["last_loss"] is not None:
        observed["train_loss"] = result["last_loss"]
    if windows:
        observed["validation_loss"] = windows[-1]
    from oeis_learn.evaluation.search import SearchSession, ModelProposer
    from oeis_learn.tracking.role_coordinator import QueueRunner

    runner = QueueRunner(output / "runtime-queue")
    examples = load_json((inputs / "examples.json").read_bytes())
    attempted, valid, variance, duplicate_failure = 0, 0, [], 0
    original = {structural_signature(r["source"]) for r in examples.get("training_rows", rows)}
    new_structures = set()
    for number, row in enumerate(rows[:4]):
        search = SearchSession(
            [int(v) for v in row["entry"]["conditioning_values"]],
            portfolio="sampling32",
            seed=7001 + number,
        ).run(ModelProposer(encoder, decoder), runner)
        rewards = []
        for candidate in search["results"]:
            attempted += 1
            status = candidate["status"]
            duplicate_failure += int(status in ("duplicate_source", "visible_mismatch"))
            execution = candidate.get("execution", {})
            valid += int(bool(execution) and execution.get("outcome") is None)
            rewards.append(int(status == "visible_fit"))
            if status == "visible_fit" and "hypothesis" in candidate.get("provenance", {}):
                item = admit_training_hypothesis(
                    candidate,
                    [int(v) for v in row["entry"]["outputs"]],
                    runner,
                    output / "diagnostic-archive",
                    holdouts=examples.get("holdouts"),
                    request_id=f"diag-new-{number}-{attempted}",
                )
                if item is not None:
                    signature = structural_signature(candidate["source"])
                    if signature not in original:
                        new_structures.add(signature)
        if rewards:
            variance.append(len(set(rewards)) > 1)
    if attempted:
        observed.update(
            valid_fraction=valid / attempted,
            reward_variance_fraction=sum(variance) / len(variance),
            duplicate_or_prefix_failure_fraction=duplicate_failure / attempted,
            eligible_archive=len(examples.get("training_rows", rows)),
            new_structures=len(new_structures),
        )
    validation = examples.get("validation_rows", [])
    if validation:
        structural_buckets = {}
        for row in validation:
            signature = structural_signature(row["source"])
            structural_buckets.setdefault(signature, set()).add(
                "long" if len(row["tokens"]) >= 512 else "short"
            )
        common = {key for key, value in structural_buckets.items() if value == {"long", "short"}}
        matched = [r for r in validation if structural_signature(r["source"]) in common]
        balanced = [r for r in matched if len(r["tokens"]) < 512][:16] + [
            r for r in matched if len(r["tokens"]) >= 512
        ][:16]
        accuracy = synthetic_accuracy(encoder, decoder, balanced, runner)
        if accuracy["short"] and accuracy["long"]:
            observed.update(
                short_accuracy=sum(accuracy["short"]) / len(accuracy["short"]),
                long_accuracy=sum(accuracy["long"]) / len(accuracy["long"]),
                matched_structural_coverage=True,
            )
    return dict(
        result,
        diagnostic=observed,
        diagnostic_attempts=attempted,
        encoder_probe=dict(profile=settings["profile"], boundary="selected input features", pairs=3),
        missing=["synthetic validation unavailable"] if not validation else [],
    )


def validation_loss(encoder, decoder, rows, device):
    import torch
    from oeis_learn.rl.sft_trainer import teacher_forced_logits
    from oeis_learn.rl.token_loss import token_cross_entropy
    from oeis_learn.rl.policy import scoring_mode
    from oeis_learn.decoder.program_codec import BOS_ID, PAD_ID

    losses = []
    with scoring_mode(encoder, decoder), torch.no_grad():
        for row in rows[:32]:
            tokens = torch.tensor([[BOS_ID] + row["tokens"]], device=device)
            logits = teacher_forced_logits(
                encoder,
                decoder,
                [[int(v) for v in row["entry"]["conditioning_values"]]],
                tokens,
                device=device,
                pad_id=PAD_ID,
            )
            losses.append(float(token_cross_entropy(logits, tokens[:, 1:]).mean()))
    return sum(losses) / len(losses)


def synthetic_accuracy(encoder, decoder, rows, runner):
    from oeis_learn.evaluation.search import ModelProposer, SearchSession

    buckets = dict(short=[], long=[])
    for number, row in enumerate(rows[:32]):
        bucket = "long" if len(row["tokens"]) >= 512 else "short"
        result = SearchSession(
            [int(v) for v in row["entry"]["conditioning_values"]],
            portfolio="greedy",
            seed=7001 + number,
        ).run(ModelProposer(encoder, decoder), runner)
        source = result["results"][0].get("source") if result["results"] else None
        correct = False
        if source is not None:
            evidence = runner.run_single(
                source, range(100), request_id=f"synthetic-validation-{number}"
            ).to_dict()
            correct = (
                evidence["outcome"] is None
                and evidence["outputs"] == row["entry"]["outputs"]
                and evidence["reference_outputs"] == evidence["outputs"]
            )
        buckets[bucket].append(correct)
    return buckets


def generate(inputs, output):
    from oeis_learn.evaluation.search import SearchSession, ModelProposer
    from oeis_learn.tracking.role_coordinator import QueueRunner

    settings = load_json((inputs / "settings.json").read_bytes())
    encoder, decoder = load_models(inputs, settings)
    encoder.eval()
    decoder.eval()
    visible = load_json((inputs / "visible.json").read_bytes())
    if set(visible) != {"groups"} or any(
        set(g) != {"group_id", "values"} or len(g["values"]) != 20 for g in visible["groups"]
    ):
        raise ValueError("generation accepts only visible twenty-term prompts")
    runner = QueueRunner(output / "runtime-queue")
    records = []
    for number, group in enumerate(visible["groups"]):
        report = SearchSession(
            [int(v) for v in group["values"]],
            portfolio=settings["profile"]["portfolio"],
            seed=settings["seed"] + number,
        ).run(ModelProposer(encoder, decoder), runner)
        fitting = [r for r in report["results"] if r.get("status") == "visible_fit"]
        selected = (
            min(
                fitting,
                key=lambda r: (
                    len(
                        __import__(
                            "oeis_learn.decoder.program_codec", fromlist=["encode_body"]
                        ).encode_body(r["source"])
                    ),
                    digest(r["source"]),
                ),
            )
            if fitting
            else None
        )
        records.append(
            dict(
                group_id=group["group_id"],
                selected_sha256=digest(selected["source"]) if selected else None,
                sources=[r["source"] for r in fitting],
                report=report,
            )
        )
    seal = dict(
        version="pilot-candidate-seal/v1",
        profile_sha256=settings["profile"]["identity_sha256"],
        seed=settings["seed"],
        records=records,
    )
    seal["seal_sha256"] = digest(seal)
    return seal


def evaluate(inputs, output):
    from oeis_learn.tracking.role_coordinator import QueueRunner

    seal = load_json((inputs / "seal.json").read_bytes())
    if seal["seal_sha256"] != digest(seal, "seal_sha256"):
        raise ValueError("candidate seal mismatch")
    development = load_json((inputs / "development.json").read_bytes())
    if development["version"] != "development-only/v1":
        raise ValueError("development evaluator only")
    groups = {g["group_id"]: g["values"] for g in development["groups"]}
    if len(seal["records"]) != len(groups) or {r["group_id"] for r in seal["records"]} != set(
        groups
    ):
        raise ValueError("complete fixed evaluation denominator required")
    runner = QueueRunner(output / "runtime-queue")
    solved, top_one, failures = set(), set(), []
    for row in seal["records"]:
        for number, source in enumerate(row["sources"]):
            result = runner.run_single(
                source, range(100), request_id=f"development-{row['group_id'][7:23]}-{number}"
            ).to_dict()
            if (
                result["outcome"] is None
                and result["outputs"] == groups[row["group_id"]]
                and result["reference_outputs"] == result["outputs"]
            ):
                solved.add(row["group_id"])
                if digest(source) == row["selected_sha256"]:
                    top_one.add(row["group_id"])
            elif result["outcome"] is not None:
                failures.append(result["outcome"])
    return dict(
        version="pilot-development-score/v1",
        candidate_seal_sha256=seal["seal_sha256"],
        solved_groups=sorted(solved),
        top_one_groups=sorted(top_one),
        denominator=len(groups),
        failure_classes=failures,
        selection_split="development",
        final_truth_available=False,
    )


def engineering(inputs, output):
    from oeis_learn.evaluation.engineering import run_engineering

    return run_engineering(inputs, output)


def smoke(inputs, output):
    from oeis_learn.rl.foundation_sft import prepare_run
    from oeis_learn.tracking.foundation_controller import run_training
    from oeis_learn.experiments.artifacts import compute_file_hash
    import shutil

    settings = load_json((inputs / "settings.json").read_bytes())
    run = output / "run"
    if settings["resume"]:
        shutil.copytree(inputs / "run", run)
    else:
        prepare_run(
            inputs / "config.yaml",
            inputs / "pool",
            run,
            device="cuda",
            diagnostic=True,
            budget_ns=settings["seconds"] * 10**9,
            run_provenance=settings.get("run_provenance"),
        )
    result = run_training(run, resume=settings["resume"], stop_after=settings.get("stop_after"))
    expected = settings.get("stop_after") or 3
    if result["completed_update"] != expected:
        raise ValueError("strict GPU smoke incomplete")
    return dict(
        status="complete" if expected == 3 else "paused",
        run_id=result["run_id"],
        checkpoint_sha256=result["checkpoint_sha256"],
        checkpoint_name=Path(result["checkpoint"]).name,
        run_root="smoke/run",
        device="cuda",
        completed_updates=result["completed_update"],
        run_sha256=compute_file_hash(run / "run.json"),
        final_truth_available=False,
    )


def main():
    operation = sys.argv[1]
    if operation == "serve":
        serve(Path("/work/input"), Path("/work/output"))
        return
    functions = {
        "candidate": candidate,
        "probe": probe,
        "train": train,
        "diagnostic": diagnostic,
        "generate": generate,
        "evaluate": evaluate,
        "engineering": engineering,
        "smoke": smoke,
    }
    if operation not in functions:
        raise ValueError("unknown internal role operation")
    result = functions[operation](Path("/work/input"), Path("/work/output"))
    atomic_write(ArtifactPath(Path("/work/output"), "result.json"), canonical_bytes(result))


def save_state(root, state):
    import torch, os
    from oeis_learn.experiments.artifacts import compute_file_hash
    from oeis_learn.tracking.foundation_metrics import fsync_directory

    root = Path(root)
    with (root / "state.pending").open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    (root / "state.pending").replace(root / "state.pt")
    fsync_directory(root)
    atomic_write(
        ArtifactPath(root, "state-manifest.json"),
        canonical_bytes(
            dict(
                version="pilot-state-manifest/v1",
                blob_sha256=compute_file_hash(root / "state.pt"),
                profile_sha256=state["profile"]["identity_sha256"],
                seed=state["seed"],
                completed=state["completed"],
            )
        ),
    )


def load_state(root, device):
    import torch
    from oeis_learn.experiments.artifacts import compute_file_hash

    root = Path(root)
    manifest = load_json((root / "state-manifest.json").read_bytes())
    if (
        set(manifest) != {"version", "blob_sha256", "profile_sha256", "seed", "completed"}
        or manifest["version"] != "pilot-state-manifest/v1"
    ):
        raise ValueError("complete external pilot checkpoint manifest required")
    if compute_file_hash(root / "state.pt") != manifest["blob_sha256"]:
        raise ValueError("pilot checkpoint final bytes mismatch")
    state = torch.load(root / "state.pt", map_location=device, weights_only=True)
    required = {
        "version",
        "seed",
        "scheduler",
        "scaler",
        "python_rng",
        "numpy_rng",
        "encoder",
        "decoder",
        "optimizer",
        "completed",
        "profile",
        "replay_rows",
        "validation_windows",
        "torch_rng",
        "device_rng",
    }
    if state.get("profile", {}).get("objective") == "grpo":
        required |= {"reference_encoder", "reference_decoder", "policy_cursor", "curriculum"}
    if (
        set(state) != required
        or type(state["completed"]) is not int
        or not 0 <= state["completed"] <= 1000000
    ):
        raise ValueError("complete active pilot state required")
    if (
        state["version"] != "pilot-state/v1"
        or state["profile"]["identity_sha256"] != manifest["profile_sha256"]
        or state["seed"] != manifest["seed"]
        or state["completed"] != manifest["completed"]
    ):
        raise ValueError("pilot checkpoint component binding mismatch")
    return state


if __name__ == "__main__":
    main()
