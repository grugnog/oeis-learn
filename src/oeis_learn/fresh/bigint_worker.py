"""Private bigint roles using qualified Phase 2 algorithms and exact checkpoints."""

from pathlib import Path
import random
import shutil
import time
from .common import read, write, file_hash, verify
from .bigint_state import CheckpointStore
from .learning_policy import Curriculum, Stopping, fixed_panel, score_panel, choose_checkpoint
from .replay import SynthesisReplay
from oeis_learn.synthesis.contracts import canonical_integer, identity, ResourceProfile
from oeis_learn.synthesis.service_policy import ServiceLimits
from oeis_learn.synthesis.state import ExperimentContract, BudgetLedger
from oeis_learn.synthesis.client import SynthesisClient


def models(recipe, device):
    import numpy as np
    import torch

    if device not in {"cpu", "cuda"}:
        raise ValueError("explicit device required")
    if device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("actual HIP required; CPU fallback prohibited")
    random.seed(recipe["model"]["seed"])
    np.random.seed(recipe["model"]["seed"])
    torch.manual_seed(recipe["model"]["seed"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from oeis_learn.experiments.bootstrap_models import construct_bigint_models
    from oeis_learn.encoder.ablation import InputMask

    mask = InputMask(**recipe["inputs"]["mask"])
    encoder, decoder, counts = construct_bigint_models(
        mask=mask, ablation_mode=recipe["inputs"]["ablation_mode"], device=device
    )
    if recipe["inputs"]["profile"] == "tri-stream/v1":
        from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
        from .bootstrap_contract import PRIMES

        if mask.bits() != (True, True, True, False):
            raise ValueError(
                "retained tri-stream control requires all mathematical streams and no byte branch"
            )
        encoder = TriStreamEncoder(
            dropout=0,
            primes=PRIMES,
            base_moduli=PRIMES,
            max_valuation=32,
            input_profile="hybrid-chunks/v1",
        ).to(device)
        counts = dict(
            total_parameters=sum(p.numel() for p in encoder.parameters()),
            active_parameters=sum(p.numel() for p in encoder.parameters()),
            trainable_parameters=sum(p.numel() for p in encoder.parameters()),
        )
    counts = dict(
        counts,
        decoder_parameters=sum(p.numel() for p in decoder.parameters()),
        device=str(next(encoder.parameters()).device),
        input_sha256=identity(recipe["inputs"]),
        decoder_sha256=decoder.codec_sha256,
    )
    return encoder, decoder, counts


def client(settings, output):
    return SynthesisClient(
        settings["contract_sha256"],
        limits=ResourceProfile(**settings["recipe"]["execution"]["resources"]),
        service_limits=ServiceLimits(**settings["recipe"]["fitting"]["service_limits"]),
        cache_root=Path(output) / "native-cache",
    )


def _batch_loss(encoder, decoder, rows, device, *, optimizer=None):
    import torch
    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.rl.sft_trainer import teacher_forced_logits, supervised_step

    sequences = [[canonical_integer(v) for v in row["values"]] for row in rows]
    lists = [[codec.BOS_ID] + row["tokens"] for row in rows]
    tokens = torch.full(
        (len(rows), max(map(len, lists))), codec.PAD_ID, dtype=torch.long, device=device
    )
    for i, ids in enumerate(lists):
        tokens[i, : len(ids)] = torch.tensor(ids, device=device)
    if optimizer is not None:
        return supervised_step(
            encoder, decoder, optimizer, sequences, tokens, device=device, pad_id=codec.PAD_ID
        )
    from oeis_learn.rl.token_loss import token_cross_entropy
    from oeis_learn.rl.policy import trajectory_mean

    with torch.no_grad():
        logits = teacher_forced_logits(
            encoder, decoder, sequences, tokens, device=device, pad_id=codec.PAD_ID
        )
        value = trajectory_mean(
            token_cross_entropy(logits, tokens[:, 1:]), tokens[:, 1:].ne(codec.PAD_ID)
        )
    return float(value)


def validation_loss(encoder, decoder, rows, device, *, deadline):
    encoder.eval()
    decoder.eval()
    total = 0.0
    done = 0
    for i in range(0, len(rows), 8):
        if time.monotonic() >= deadline:
            break
        batch = rows[i : i + 8]
        total += _batch_loss(encoder, decoder, batch, device) * len(batch)
        done += len(batch)
    return dict(
        loss=total / done if done else None,
        complete=done == len(rows),
        attempted=done,
        denominator=len(rows),
    )


def train(inputs, output):
    import numpy as np
    import torch

    inputs, output = Path(inputs), Path(output)
    settings = read(inputs / "settings.json")
    recipe = settings["recipe"]
    examples = read(inputs / "examples.json")
    if identity(examples) != settings["examples_sha256"]:
        raise ValueError("learner view changed")
    if any(
        key not in {"training", "parameter_development", "structural_development"}
        for key in examples
    ):
        raise ValueError("hidden/final data in learner view")
    if any(
        len(r["values"]) != 20 or len(r["indices"]) != 20
        for rows in examples.values()
        for r in rows
    ):
        raise ValueError("visible-only learner data required")
    contract = ExperimentContract(**settings["contract"])
    ledger = BudgetLedger.restore(settings["ledger"])
    encoder, decoder, counts = models(recipe, settings["device"])
    optimizer = torch.optim.AdamW(
        [p for module in (encoder, decoder) for p in module.parameters() if p.requires_grad],
        lr=recipe["learning"]["learning_rate"],
        foreach=False,
    )
    rows = examples["training"]
    index = {r["id"]: r for rows_ in examples.values() for r in rows_}
    order_rng = torch.Generator().manual_seed(recipe["model"]["seed"])
    curriculum = Curriculum(recipe["learning"]["validation_every"])
    stopping = Stopping()
    replay = SynthesisReplay(
        capacity=recipe["learning"]["replay_capacity"],
        track=recipe["track"]["kind"],
        representation_comparison=recipe["learning"]["representation_comparison"],
        training_targets=tuple(r["id"] for r in rows),
    )
    order = dict(cursor=0, permutation=[], epoch=0, completed=0)
    selection = dict(
        evaluations=[],
        selected_update=None,
        best_checkpoint=None,
        panels={},
        stopping=stopping.snapshot(),
    )
    if not settings.get("fixture"):
        selection["panels"] = {split: list(fixed_panel(group)) for split, group in examples.items()}
    else:
        selection["panels"] = {split: [r["id"] for r in group] for split, group in examples.items()}
    if (inputs / "checkpoints/latest.json").exists():
        shutil.copytree(inputs / "checkpoints", output / "checkpoints", dirs_exist_ok=True)
        pointer = read(output / "checkpoints/latest.json")
        restored = CheckpointStore(output / "checkpoints", contract).load(pointer)
        states = restored["states"]
        if any(
            key not in ledger.records or ledger.records[key]["debit"] != row["debit"]
            for key, row in restored["ledger"].records.items()
        ):
            raise ValueError("checkpoint ledger lost prior debit")
        encoder.load_state_dict(states["weights"]["encoder"])
        decoder.load_state_dict(states["weights"]["decoder"])
        optimizer.load_state_dict(states["optimizer"])
        random.setstate(states["rng"]["python"])
        np.random.set_state(states["rng"]["numpy"])
        torch.set_rng_state(states["rng"]["torch"])
        if states["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(states["rng"]["cuda"])
        order = states["data_order"]
        order_rng.set_state(order["rng"])
        selection = states["selection"]
        curriculum = Curriculum.restore(states["curriculum"])
        replay = SynthesisReplay.restore(states["replay"])
        stopping = Stopping.restore(selection["stopping"])
        if (inputs / "selected.json").exists():
            selection["best_checkpoint"] = read(inputs / "selected.json")
    elif settings.get("foundation"):
        parent = settings["foundation"]
        parent_contract = ExperimentContract(**parent["record"]["contract"])
        restored = CheckpointStore(inputs / "foundation/checkpoints", parent_contract).load(
            parent["selected"]
        )
        encoder.load_state_dict(restored["states"]["weights"]["encoder"])
        decoder.load_state_dict(restored["states"]["weights"]["decoder"])
        # Warm initialization only: optimizer, RNG, data order and budget are this branch's.
    deadline = time.monotonic() + settings["seconds"]
    store = CheckpointStore(output / "checkpoints", contract)
    metrics = []
    pointer = None
    early = False

    def save():
        nonlocal pointer
        order["rng"] = order_rng.get_state()
        selection["stopping"] = stopping.snapshot()
        states = dict(
            weights=dict(encoder=encoder.state_dict(), decoder=decoder.state_dict()),
            optimizer=optimizer.state_dict(),
            rng=dict(
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if settings["device"] == "cuda" else None,
            ),
            data_order=order,
            selection=selection,
            curriculum=curriculum.snapshot(),
            replay=replay.snapshot(),
        )
        pointer = store.save(states, ledger, state="active")
        chosen = (
            pointer
            if selection["selected_update"] == order["completed"]
            or selection["best_checkpoint"] is None
            else selection["best_checkpoint"]
        )
        write(output / "selected.json", chosen)
        selection["best_checkpoint"] = chosen
        store.prune(chosen)
        write(
            output / "checkpoint-identity.json",
            dict(
                latest=pointer,
                selected=chosen,
                contract_sha256=contract.sha256,
                update=order["completed"],
                input_sha256=identity(recipe["inputs"]),
                codec_sha256=decoder.codec_sha256,
            ),
        )

    try:
        with client(settings, output) as synthesis:
            target = min(
                recipe["learning"]["max_updates"],
                order["completed"]
                + settings.get("update_limit", recipe["learning"]["validation_every"]),
            )
            while order["completed"] < target and time.monotonic() < deadline:
                from .tuning import permutation

                pool = list(range(len(rows)))
                if recipe["learning"]["curriculum"]:
                    stage = "ABCD"[curriculum.stage]
                    current = [i for i in pool if rows[i]["stage"] == stage]
                    earlier = [i for i in pool if rows[i]["stage"] < stage]
                    if not current or curriculum.stage and not earlier:
                        raise ValueError("curriculum stage coverage missing")
                    count = curriculum.earlier_count(recipe["learning"]["batch_size"])
                    # Reuse the existing counter/RNG permutation for each declared pool.
                    chosen = [
                        earlier[int(torch.randint(len(earlier), (), generator=order_rng))]
                        for _ in range(count)
                    ]
                    chosen += [
                        current[int(torch.randint(len(current), (), generator=order_rng))]
                        for _ in range(recipe["learning"]["batch_size"] - count)
                    ]
                else:
                    if order["cursor"] + recipe["learning"]["batch_size"] > len(
                        order["permutation"]
                    ):
                        order["permutation"] = permutation(
                            rows, order_rng, recipe["learning"]["batch_size"]
                        )
                        order["cursor"] = 0
                        order["epoch"] += 1
                    chosen = order["permutation"][
                        order["cursor"] : order["cursor"] + recipe["learning"]["batch_size"]
                    ]
                    order["cursor"] += len(chosen)
                batch = [rows[i] for i in chosen]
                if recipe["learning"]["replay"]:
                    augmented = replay.sample(len(batch))
                    batch = [
                        dict(index[r["target"]], tokens=r["tokens"]) for r in augmented
                    ] + batch[len(augmented) :]
                encoder.train()
                decoder.train()
                record = _batch_loss(
                    encoder, decoder, batch, settings["device"], optimizer=optimizer
                )
                order["completed"] += 1
                metrics.append(
                    dict(
                        update=order["completed"],
                        loss=record["loss"],
                        phase_timings_ns=record["phase_timings_ns"],
                    )
                )
                write(
                    output / "progress.json",
                    dict(update=order["completed"], last_loss=record["loss"]),
                )
                if (
                    order["completed"] % recipe["learning"]["validation_every"] == 0
                    or order["completed"] == recipe["learning"]["max_updates"]
                ):
                    from .bigint_diagnostics import generate

                    loss = validation_loss(
                        encoder,
                        decoder,
                        examples["parameter_development"] + examples["structural_development"],
                        settings["device"],
                        deadline=deadline,
                    )
                    panel = [index[k] for ids in selection["panels"].values() for k in ids]
                    diagnostic = generate(
                        encoder, decoder, panel, recipe, synthesis, deadline=deadline, periodic=True
                    )
                    visible = score_panel(
                        tuple(r["id"] for r in panel),
                        {
                            r["id"]: dict(
                                complete=bool(
                                    diagnostic["records"][r["id"]]["selections"].get("rescue")
                                    or diagnostic["records"][r["id"]]["selections"].get("raw")
                                )
                                or diagnostic["records"][r["id"]]["complete"],
                                verified=bool(
                                    diagnostic["records"][r["id"]]["selections"].get("rescue")
                                    or diagnostic["records"][r["id"]]["selections"].get("raw")
                                ),
                            )
                            for r in panel
                        },
                    )
                    item = dict(
                        checkpoint=str(order["completed"]),
                        visible=visible,
                        loss=loss["loss"] if loss["loss"] is not None else 1e30,
                        loss_complete=loss["complete"],
                        update=order["completed"],
                    )
                    selection["evaluations"].append(item)
                    selection["selected_update"] = int(
                        choose_checkpoint(
                            selection["evaluations"], panel_sha256=visible["panel_sha256"]
                        )
                    )
                    if recipe["learning"]["curriculum"]:
                        stage = "ABCD"[curriculum.stage]
                        current = [r for r in panel if r.get("stage") == stage]
                        current_ids = tuple(r["id"] for r in current)
                        if current_ids:
                            report = score_panel(
                                current_ids,
                                {
                                    k: dict(
                                        complete=diagnostic["records"][k]["complete"]
                                        or bool(
                                            diagnostic["records"][k]["selections"].get("rescue")
                                            or diagnostic["records"][k]["selections"].get("raw")
                                        ),
                                        verified=bool(
                                            diagnostic["records"][k]["selections"].get("rescue")
                                            or diagnostic["records"][k]["selections"].get("raw")
                                        ),
                                    )
                                    for k in current_ids
                                },
                            )
                            curriculum.observe(report, update=order["completed"])
                    if recipe["learning"]["replay"]:
                        admit_replay(replay, rows, diagnostic, synthesis, deadline)
                    early = stopping.observe(visible["success"])
                    write(
                        output / "validation.json",
                        dict(visible=visible, teacher_forcing=loss, panels=selection["panels"]),
                    )
                    save()
                    if early:
                        break
            if selection["selected_update"] is None:
                selection["selected_update"] = order["completed"]
    finally:
        save()
    write(
        output / "metrics.json",
        dict(
            updates=metrics,
            total_updates=order["completed"],
            parameter_counts=counts,
            fixture=bool(settings.get("fixture")),
            stage=curriculum.stage,
            early_stop=early,
        ),
    )
    return dict(
        update=order["completed"],
        checkpoint=pointer,
        selected=read(output / "selected.json"),
        early_stop=early,
        counts=counts,
        final_access="sealed",
        fixture=bool(settings.get("fixture")),
    )


def _load_models(inputs, settings):
    from .bigint_state import CheckpointStore

    encoder, decoder, counts = models(settings["recipe"], settings["device"])
    contract = ExperimentContract(**settings["contract"])
    store = CheckpointStore(Path(inputs) / "checkpoints", contract)
    loaded = store.load(settings["checkpoint"])
    encoder.load_state_dict(loaded["states"]["weights"]["encoder"])
    decoder.load_state_dict(loaded["states"]["weights"]["decoder"])
    if settings.get("ablation"):
        from oeis_learn.encoder.ablation import InputMask

        if not hasattr(encoder, "group_mask"):
            raise ValueError("four-group attribution requires hybrid-chunks")
        encoder.group_mask = InputMask(**settings["ablation"])
    return encoder, decoder, counts


def generate(inputs, output):
    from .bigint_diagnostics import generate as synthesize

    inputs, output = Path(inputs), Path(output)
    settings = read(inputs / "settings.json")
    rows = read(inputs / "prompts.json")
    if any(
        set(row) - {"id", "indices", "values", "horizon_indices", "family", "strata"}
        for row in rows
    ):
        raise ValueError("only visible prompts and public report strata enter generation")
    encoder, decoder, counts = _load_models(inputs, settings)
    with client(settings, output) as synthesis:
        result = synthesize(
            encoder,
            decoder,
            rows,
            settings["recipe"],
            synthesis,
            deadline=time.monotonic() + settings["seconds"],
            compare_cache=settings.get("compare_cache", False),
        )
    chunks = {}
    details = []
    group = {}

    def flush():
        if not group:
            return
        path = output / f"selections/chunk-{len(chunks):04}.json"
        write(path, group)
        chunks[str(path.relative_to(output))] = dict(
            sha256=file_hash(path), bytes=path.stat().st_size
        )
        group.clear()

    from .common import canonical

    for row in rows:
        key = row["id"]
        record = result["records"][key]
        body = {k: record[k] for k in ("selections", "complete", "family", "strata")}
        if len(canonical({key: body})) > 4 << 20:
            raise ValueError("single sealed selection exceeds transport cap")
        if len(canonical(dict(group, **{key: body}))) > 4 << 20:
            flush()
        group[key] = body
    flush()
    archive_bytes = 0
    archive_omissions = []
    aggregate = dict(
        denominator=len(rows),
        attempts=0,
        terminated=0,
        legal=0,
        ambiguous=0,
        visible_fit20=0,
        elapsed_seconds=0.0,
        cache_comparisons=0,
        cache_equal=0,
        greedy_terminated=0,
    )
    for row in rows:
        record = result["records"][row["id"]]
        body = {mode: value["archive"] for mode, value in record["details"].items()}
        payload = canonical(body)
        name = f"archive/{identity(row['id'])[7:]}.json"
        if archive_bytes + len(payload) <= settings["recipe"]["archive"]["max_bytes"]:
            write(output / name, payload)
            archive_bytes += len(payload)
        else:
            archive_omissions.append(
                dict(
                    path=name, sha256=identity(body), bytes=len(payload), reason="archive byte cap"
                )
            )
        greedy = record["details"]["correct"]["details"]
        aggregate["greedy_terminated"] += bool(greedy and greedy[0]["complete"])
        for mode, value in record["details"].items():
            aggregate["elapsed_seconds"] += value["elapsed_seconds"]
            aggregate["attempts"] += len(value["details"])
            aggregate["terminated"] += sum(attempt["complete"] for attempt in value["details"])
            aggregate["legal"] += sum(
                attempt["complete"] and attempt["outcome"] != "invalid_source"
                for attempt in value["details"]
            )
            aggregate["ambiguous"] += sum(
                attempt.get("native", {}).get("uniqueness") == "multiple"
                for attempt in value["details"]
            )
            if value["cache_parity"] is not None:
                aggregate["cache_comparisons"] += 1
                aggregate["cache_equal"] += value["cache_parity"]
        aggregate["visible_fit20"] += bool(record["selections"].get("equal_budget"))
    write(
        output / "archive-manifest.json",
        dict(
            bytes=archive_bytes,
            max_bytes=settings["recipe"]["archive"]["max_bytes"],
            omitted=archive_omissions,
            scope="finite visible matches; no proof or novelty promotion",
        ),
    )
    for row in rows[: settings["recipe"]["reporting"]["max_examples"]]:
        body = result["records"][row["id"]]
        name = f"details/{identity(row['id'])[7:]}.json"
        write(output / name, body)
        if (output / name).stat().st_size > settings["recipe"]["reporting"]["detail_chunk_bytes"]:
            (output / name).unlink()
            details.append(dict(id=row["id"], reason="detail chunk cap", sha256=identity(body)))
    return dict(
        version="bigint-generation/v1",
        files=chunks,
        controls=result["controls"],
        generation_metrics=aggregate,
        archive=dict(bytes=archive_bytes, omitted=len(archive_omissions)),
        counts=counts,
        denominator=len(rows),
        omitted=details,
        final_access=settings.get("final_access", "sealed"),
    )


def score(inputs, output):
    from .bigint_diagnostics import score as evaluate

    inputs, output = Path(inputs), Path(output)
    settings = read(inputs / "settings.json")
    truth = read(inputs / "truth.json")
    manifest = read(inputs / "selections.json")
    verify(inputs, manifest)
    selections = {}
    for path in manifest:
        selections.update(read(inputs / path))
    with client(settings, output) as synthesis:
        result = evaluate(
            selections, truth, synthesis, deadline=time.monotonic() + settings["seconds"]
        )
    write(output / "outcomes.json", result.pop("outcomes"))
    return result


def probe(inputs, output):
    import torch

    settings = read(Path(inputs) / "settings.json")
    device = settings.get("device", "cuda")
    if device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("actual HIP required")
    return dict(
        device=device,
        torch=torch.__version__,
        hip=torch.version.hip,
        name=torch.cuda.get_device_name() if device == "cuda" else "CPU",
        memory_bytes=torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None,
    )


def admit_replay(replay, rows, diagnostic, synthesis, deadline):
    """Training-only oracle execution; no development/OEIS truth is accepted."""
    from oeis_learn.synthesis.admission import reference_identity, source_identity
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.decoder.bigint_codec import encode_body

    for row in rows:
        selected = diagnostic["records"].get(row["id"], {}).get("selections", {}).get("rescue")
        if selected is None or time.monotonic() >= deadline:
            continue
        if "oracle_source" not in row:
            raise ValueError("training oracle source required for replay")
        indices = row["horizon_indices"]
        source = selected["selected"]["source"]
        expected = evaluate(row["oracle_source"], indices, synthesis.limits, deadline=deadline)
        production = synthesis.execute(source, indices, deadline=deadline)
        independent = evaluate(source, indices, synthesis.limits, deadline=deadline)
        if (
            expected.outcome != independent.outcome
            or expected.outcome != "ok"
            or production.get("outcome") != "ok"
        ):
            continue
        if list(independent.values) != production["values"]:
            raise RuntimeError("replay production/reference disagreement")
        if expected.values != independent.values or len(encode_body(source)) >= len(row["tokens"]):
            continue
        verification = dict(
            target=row["id"],
            source_sha256=source_identity(source),
            runtime_sha256=synthesis.runtime_identity("bigint-body/v1"),
            reference_sha256=reference_identity(),
            indices=indices,
            values=list(independent.values),
            admitted=True,
        )
        replay.admit(
            dict(
                target=row["id"],
                split="training",
                track=replay.track,
                source=source,
                original_source=selected["original_source"],
                assignment=selected["assignment"],
                profile="bigint-body/v1",
                verification=verification,
                independent=True,
                tokens=encode_body(source),
            )
        )
