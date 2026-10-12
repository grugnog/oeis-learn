"""Private role entry points for fresh SFT and sealed finite evaluation."""

import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

from .common import digest, read, write


def runtime(device):
    from oeis_learn.rl.foundation_sft import runtime_record

    return runtime_record(device)


def models(settings):
    import numpy as np
    import torch

    from oeis_learn.experiments.profiles import construct_qualified_models

    if settings["device"] not in {"cpu", "cuda"}:
        raise ValueError("explicit CPU fixture or HIP device required")
    if settings["device"] == "cuda" and (not torch.cuda.is_available() or not torch.version.hip):
        raise RuntimeError("actual HIP accelerator required; no CPU fallback")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if settings["device"] == "cuda":
        index = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.8, (64 << 30) / total), index)
    return construct_qualified_models(settings["model"], device=settings["device"])


def identities(settings):
    return {
        "expected_contract": settings["contract_sha256"],
        "expected_profile": settings["model"]["identity_sha256"],
        "expected_runtime": digest(runtime(settings["device"])),
    }


def train(inputs, output):
    import torch

    from oeis_learn.decoder.program_codec import BOS_ID, PAD_ID, decode_body, encode_body
    from oeis_learn.rl.sft_trainer import supervised_step
    from oeis_learn.tracking.role_worker import validation_loss
    from oeis_learn.tracking.training_checkpoint import (
        QualifiedCheckpointStore,
        capture_learning_state,
        restore_learning_state,
    )

    inputs, output = Path(inputs), Path(output)
    settings = read(inputs / "settings.json")
    examples = read(inputs / "examples.json")
    if digest(examples) != settings["examples_sha256"]:
        raise ValueError("learner examples do not match frozen run")
    rows = examples["training"]
    if not rows or any(
        len(r["values"]) != 20 or encode_body(decode_body(r["tokens"])) != r["tokens"]
        for r in rows + examples["development"]
    ):
        raise ValueError("complete visible training examples required")
    encoder, decoder, counts = models(settings)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()],
        lr=settings["learning_rate"],
        weight_decay=0.01,
        foreach=False,
    )
    store_root = output / "checkpoints"
    if (inputs / "checkpoints").exists():
        shutil.copytree(inputs / "checkpoints", store_root)
    store = QualifiedCheckpointStore(store_root, free_floor_bytes=settings["free_disk_floor_bytes"])
    binding = identities(settings)
    rng = torch.Generator().manual_seed(settings["seed"] + 1)
    order = {
        "permutation": torch.randperm(len(rows), generator=rng).tolist(),
        "cursor": 0,
        "completed": 0,
        "epoch": 0,
        "rng": rng.get_state(),
        "next_sample_ids": [],
    }
    previous = None
    recovery = None
    if (inputs / "checkpoints").exists():
        path, (manifest, base, state) = store.recover_learning(**binding)
        if base["charged_budget_ns"] > settings["charged_budget_ns"]:
            raise ValueError("checkpoint budget exceeds durable host ledger")
        order, prefetch = restore_learning_state(state, encoder, decoder, optimizer)
        if prefetch != {"examples_sha256": settings["examples_sha256"], "next": []}:
            raise ValueError("checkpoint data identity mismatch")
        if sorted(order["permutation"]) != list(range(len(rows))) or not 0 <= order[
            "cursor"
        ] <= len(rows):
            raise ValueError("checkpoint data cursor mismatch")
        rng.set_state(order["rng"])
        previous = manifest["qualified"]["identity_sha256"]
        recovery = {
            "manifest": path.name,
            "completed": order["completed"],
            "checkpoint_sha256": base["blob_sha256"],
        }
    encoder.train()
    decoder.train()
    start = time.monotonic()
    last_save = start
    checkpoint = None
    remaining = settings["seconds"]

    def save():
        nonlocal previous, checkpoint, last_save
        order["rng"] = rng.get_state()
        cursor = order["cursor"]
        order["next_sample_ids"] = [
            rows[i]["program_id"]
            for i in order["permutation"][cursor : cursor + settings["batch_size"]]
        ]
        state = capture_learning_state(
            encoder,
            decoder,
            optimizer,
            sampler=order,
            prefetch={"examples_sha256": settings["examples_sha256"], "next": []},
        )
        checkpoint = store.save_learning(
            state,
            contract_sha256=binding["expected_contract"],
            profile_sha256=binding["expected_profile"],
            runtime_sha256=binding["expected_runtime"],
            ledger_sequence=settings["ledger_sequence"],
            charged_budget_ns=settings["charged_budget_ns"] + int((time.monotonic() - start) * 1e9),
            completed_update=order["completed"],
            previous=previous,
        )
        previous = read(checkpoint)["qualified"]["identity_sha256"]
        last_save = time.monotonic()
        write(
            output / "progress.json",
            {
                "completed_updates": order["completed"],
                "checkpoint": checkpoint.name,
                "elapsed_seconds": last_save - start,
            },
        )
        print(f"CHECKPOINT update={order['completed']} elapsed={last_save-start:.1f}s", flush=True)

    # A durable random-initialized baseline is itself a complete active state.
    if recovery is None:
        save()
    with (output / "metrics.jsonl").open("ab") as stream:
        os.fchmod(stream.fileno(), 0o644)
        while order["completed"] < settings["update_cap"] and time.monotonic() - start < max(
            0, remaining - settings["checkpoint_reserve_seconds"]
        ):
            update_started = time.monotonic_ns()
            if order["cursor"] == len(rows):
                order.update(
                    permutation=torch.randperm(len(rows), generator=rng).tolist(),
                    cursor=0,
                    epoch=order["epoch"] + 1,
                )
            indices = order["permutation"][
                order["cursor"] : order["cursor"] + settings["batch_size"]
            ]
            batch = [rows[i] for i in indices]
            tokens = [[BOS_ID] + row["tokens"] for row in batch]
            padded = torch.full(
                (len(batch), max(map(len, tokens))),
                PAD_ID,
                device=settings["device"],
                dtype=torch.long,
            )
            for i, row in enumerate(tokens):
                padded[i, : len(row)] = torch.tensor(row, device=settings["device"])
            before = decoder.lm_head.weight.detach().clone()
            result = supervised_step(
                encoder,
                decoder,
                optimizer,
                [[int(v) for v in row["values"]] for row in batch],
                padded,
                device=settings["device"],
                pad_id=PAD_ID,
            )
            if torch.equal(before, decoder.lm_head.weight.detach()):
                raise RuntimeError("optimizer update did not change decoder parameters")
            result.update(
                parameters_changed=True, parameter_device=str(decoder.lm_head.weight.device)
            )
            order["cursor"] += len(batch)
            order["completed"] += 1
            result.update(
                update=order["completed"],
                epoch=order["epoch"],
                sample_ids=[r["program_id"] for r in batch],
                nonpadding_tokens=sum(len(r["tokens"]) for r in batch),
                elapsed_seconds=time.monotonic() - start,
                update_wall_ns=time.monotonic_ns() - update_started,
                observed_utc_ns=time.time_ns(),
            )
            from .common import canonical

            stream.write(canonical(result) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            if time.monotonic() - last_save >= settings["checkpoint_every_seconds"]:
                save()
    save()
    validation = None
    if examples["development"] and time.monotonic() - start < remaining - 2:
        validation = validation_loss(
            encoder,
            decoder,
            [
                {"tokens": r["tokens"], "entry": {"conditioning_values": r["values"]}}
                for r in examples["development"]
            ],
            settings["device"],
        )
    write(output / "runtime.json", runtime(settings["device"]))
    return {
        "version": "fresh-training-result/v1",
        "completed_updates": order["completed"],
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": read(store_root / (checkpoint.stem + "-base.json"))["blob_sha256"],
        "recovery": recovery,
        "validation_loss": validation,
        "parameter_count": counts,
        "runtime_sha256": binding["expected_runtime"],
        "device": settings["device"],
        "qualified": False,
    }


def generate(inputs, output):
    from oeis_learn.evaluation.search import ModelProposer, SearchExhausted, SearchLedger
    from oeis_learn.tracking.training_checkpoint import QualifiedCheckpointStore

    inputs, output = Path(inputs), Path(output)
    settings = read(inputs / "settings.json")
    visible = read(inputs / "visible.json")
    if not isinstance(visible, list) or any(
        set(r) != {"group_id", "values"} or len(r["values"]) != 20 for r in visible
    ):
        raise ValueError("generator receives only twenty visible terms")
    encoder, decoder, _ = models(settings)
    store = QualifiedCheckpointStore(inputs / "checkpoints", free_floor_bytes=0)
    _, base, state = store.load_learning(
        inputs / "checkpoints" / settings["checkpoint"], **identities(settings)
    )
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    encoder.eval()
    decoder.eval()
    proposer = ModelProposer(encoder, decoder, context_limit=settings["generation_tokens"] + 1)
    records = []
    for n, group in enumerate(visible):
        attempts = []
        ledger = SearchLedger(
            max_attempts=settings["attempts"],
            max_tokens=settings["attempts"] * settings["generation_tokens"],
            max_expansions=settings["attempts"] * settings["generation_tokens"],
            budget_ns=settings["per_group_seconds"] * 10**9,
        )
        for attempt in range(settings["attempts"]):
            try:
                ledger.charge(attempts=1)
                source, evidence = proposer.stochastic(
                    [int(v) for v in group["values"]],
                    settings["seed"] + n * 32 + attempt,
                    ledger,
                    greedy=attempt == 0,
                )
                attempts.append(
                    {
                        "source": source,
                        "method": evidence["method"],
                        "complete": evidence["complete"],
                        "failure": evidence["failure"],
                    }
                )
            except SearchExhausted as exc:
                attempts.append(
                    {"source": None, "method": "bounded", "complete": False, "failure": str(exc)}
                )
                break
        # Selection is fixed before any hidden scoring: first complete candidate.
        selected = next(
            (digest(r["source"]) for r in attempts if r["source"] is not None and r["complete"]),
            None,
        )
        records.append(
            {"group_id": group["group_id"], "selected_sha256": selected, "attempts": attempts}
        )
        write(
            output / "progress.json",
            {"completed_groups": len(records), "denominator": len(visible)},
        )
    seal = {
        "version": "fresh-candidate-seal/v1",
        "checkpoint_sha256": base["blob_sha256"],
        "protocol_sha256": settings["protocol_sha256"],
        "visible_sha256": digest(visible),
        "records": records,
    }
    seal["identity"] = digest(seal)
    return seal


def score(inputs, output):
    from oeis_learn.tracking.role_coordinator import QueueRunner

    inputs, output = Path(inputs), Path(output)
    seal, truth, settings = (
        read(inputs / "seal.json"),
        read(inputs / "truth.json"),
        read(inputs / "settings.json"),
    )
    if seal["identity"] != digest({k: v for k, v in seal.items() if k != "identity"}):
        raise ValueError("candidate seal changed")
    groups = {r["group_id"]: r["values"] for r in truth}
    if (
        len(groups) != len(truth)
        or {r["group_id"] for r in seal["records"]} != set(groups)
        or len(seal["records"]) != len(groups)
    ):
        raise ValueError("fixed complete denominator required")
    if settings["split"] == "final":
        final = read(inputs / "finalization.json")
        if (
            final["checkpoint_sha256"] != seal["checkpoint_sha256"]
            or final["protocol_sha256"] != seal["protocol_sha256"]
            or final["truth_sha256"] != digest(truth)
        ):
            raise ValueError("final selection/protocol lock mismatch")
    runner = QueueRunner(output / "runtime-queue")
    solved, top, failures, details = [], [], Counter(), []
    execution_totals = {"calls": 0, "elapsed_ns": 0, "fuel": 0, "reference_steps": 0}
    for row in seal["records"]:
        correct, selected = False, False
        for number, attempt in enumerate(row["attempts"]):
            source = attempt["source"]
            if source is None or not attempt["complete"]:
                failures[attempt["failure"] or "incomplete_program"] += 1
                continue
            result = runner.run_single(
                source, range(100), request_id=f"fresh-{row['group_id'][7:23]}-{number}"
            ).to_dict()
            execution_totals["calls"] += 1
            for name in ("elapsed_ns", "fuel", "reference_steps"):
                value = result.get(name)
                if value is None:
                    execution_totals[name] = None
                elif execution_totals[name] is not None:
                    execution_totals[name] += value
            if result["outcome"] is None and result["outputs"] != result["reference_outputs"]:
                raise RuntimeError("native/reference disagreement during sealed scoring")
            matched = (
                result["outcome"] is None
                and result["outputs"] == groups[row["group_id"]]
                and result["reference_outputs"] == result["outputs"]
            )
            failures["exact_match" if matched else result["outcome"] or "wrong_value"] += 1
            correct |= matched
            selected |= matched and digest(source) == row["selected_sha256"]
            if settings["split"] == "development":
                details.append(
                    {
                        "group_id": row["group_id"],
                        "attempt": number,
                        "source": source,
                        "matched": matched,
                        "outcome": result["outcome"],
                        "reason": result["reason"],
                        "elapsed_ns": result["elapsed_ns"],
                        "fuel": result["fuel"],
                        "reference_steps": result["reference_steps"],
                    }
                )
        if correct:
            solved.append(row["group_id"])
        if selected:
            top.append(row["group_id"])
    report = {
        "version": "fresh-score/v1",
        "split": settings["split"],
        "denominator": len(groups),
        "solved_groups": solved,
        "top_one_groups": top,
        "failure_classes": dict(failures),
        "details": details,
        "candidate_seal_sha256": seal["identity"],
        "checkpoint_sha256": seal["checkpoint_sha256"],
        "protocol_sha256": seal["protocol_sha256"],
        "qualified": False,
        "finite_horizon": 100,
        "execution_telemetry": execution_totals,
    }
    report["identity"] = digest(report)
    return report


def probe(inputs, output):
    import torch

    settings = read(Path(inputs) / "settings.json")
    if (
        Path("/workspace/gpu-project.json").exists()
        or Path("/work/input/truth.json").exists()
        or Path("/var/run/docker.sock").exists()
    ):
        raise RuntimeError("workspace/truth/runtime socket leaked into the role")
    gpu = settings["gpu"]
    if gpu:
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("HIP unavailable in private GPU role")
        value = torch.ones((16, 16), device="cuda")
        assert (value @ value).sum().item() == 4096
    elif Path("/dev/kfd").exists() or list(Path("/dev/dri").glob("renderD*")):
        raise RuntimeError("GPU devices leaked into CPU role")
    maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    swap = Path("/sys/fs/cgroup/memory.swap.max").read_text().strip()
    expected = (88 if gpu else 1) << 30
    if maximum != str(expected) or swap != "0":
        raise RuntimeError(f"role memory/swap bounds differ: {maximum}/{swap}")
    try:
        Path(inputs, "write-probe").write_text("must fail")
    except OSError:
        pass
    else:
        raise RuntimeError("input mount is writable")
    return {
        "status": "PASS",
        "gpu": gpu,
        "hip": torch.version.hip if gpu else None,
        "memory_max": int(maximum),
        "memory_swap_max": int(swap),
        "workspace_mounted": False,
        "hidden_truth_available": False,
    }


def main():
    import sys

    operation = sys.argv[1]
    handlers = {"train": train, "generate": generate, "score": score, "probe": probe}
    result = handlers[operation](Path("/work/input"), Path("/work/output"))
    write(Path("/work/output/result.json"), result)


if __name__ == "__main__":
    main()
