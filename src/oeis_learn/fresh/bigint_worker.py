"""Private bigint roles using qualified Phase 2 algorithms and exact checkpoints."""

import random
import shutil
import time
from pathlib import Path

from oeis_learn.synthesis.client import SynthesisClient
from oeis_learn.synthesis.contracts import ResourceProfile, canonical_integer, identity
from oeis_learn.synthesis.service_policy import ServiceLimits
from oeis_learn.synthesis.state import BudgetLedger, ExperimentContract

from .bigint_state import CheckpointStore
from .common import file_hash, read, verify, write
from .learning_policy import Curriculum, Stopping, choose_checkpoint, fixed_panel, score_panel
from .replay import SynthesisReplay


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
    from .bootstrap_contract import bigint_profile

    profile = bigint_profile(recipe)
    from oeis_learn.encoder.ablation import InputMask
    from oeis_learn.experiments.bootstrap_models import construct_bigint_models

    mask = InputMask(**recipe["inputs"]["mask"])
    if recipe['version']=='bigint-bootstrap/v2':
        from oeis_learn.experiments.bootstrap_models import construct_sizing_models
        encoder,decoder,actual=construct_sizing_models(recipe['model_profile'],device=device)
        counts=dict(encoder_parameters=actual['encoder'],total_parameters=actual['total'],
            active_parameters=actual['active'],trainable_parameters=actual['active'],actual=actual)
    else:
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
        profile=profile,
    )
    return encoder, decoder, counts


def client(settings, output):
    from dataclasses import replace
    limits = ServiceLimits(**settings['recipe']['fitting']['service_limits'])
    pipeline = settings['recipe'].get('pipeline')
    if pipeline:
        limits = replace(limits, max_inflight_batches=pipeline['workers'])
        if pipeline.get('cache_capacity')=='expanded/v1':
            limits=replace(limits,result_cache_max_entries=32768,result_cache_max_bytes=512<<20,
                disk_cache_max_entries=2048,disk_cache_max_bytes=4<<30)
    return SynthesisClient(
        settings["contract_sha256"],
        limits=ResourceProfile(**settings["recipe"]["execution"]["resources"]),
        service_limits=limits,
        cache_root=Path(output) / "native-cache",
    )


def _batch_loss(encoder, decoder, rows, device, *, optimizer=None, timer=None, observe=True):
    import torch

    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.rl.sft_trainer import supervised_step, teacher_forced_logits

    sequences = [[canonical_integer(v) for v in row["values"]] for row in rows]
    lists = [[codec.BOS_ID] + row["tokens"] for row in rows]
    tokens = torch.full(
        (len(rows), max(map(len, lists))), codec.PAD_ID, dtype=torch.long, device=device
    )
    for i, ids in enumerate(lists):
        tokens[i, : len(ids)] = torch.tensor(ids, device=device)
    if optimizer is not None:
        return supervised_step(
            encoder, decoder, optimizer, sequences, tokens, device=device, pad_id=codec.PAD_ID,
            timer=timer,observe=observe
        )
    from oeis_learn.rl.policy import trajectory_mean
    from oeis_learn.rl.token_loss import token_cross_entropy

    with torch.no_grad():
        logits = teacher_forced_logits(
            encoder, decoder, sequences, tokens, device=device, pad_id=codec.PAD_ID,timer=timer
        )
        value = trajectory_mean(
            token_cross_entropy(logits, tokens[:, 1:]), tokens[:, 1:].ne(codec.PAD_ID)
        )
    return float(value)


def validation_loss(encoder, decoder, rows, device, *, deadline, measure=None,timer=None):
    encoder.eval()
    decoder.eval()
    total = 0.0
    done = 0
    for i in range(0, len(rows), 8):
        if time.monotonic() >= deadline:
            break
        batch = rows[i : i + 8]
        began=time.monotonic()
        total += _batch_loss(encoder, decoder, batch, device,timer=timer) * len(batch)
        if measure is not None:measure(time.monotonic()-began,len(batch))
        done += len(batch)
    return dict(
        loss=total / done if done else None,
        complete=done == len(rows),
        attempted=done,
        denominator=len(rows),
    )


def periodic_panels(examples, scope):
    """The memorization panel is exactly its frozen32, not three32-row panels."""
    if scope == "diverse32":
        ids = [r["id"] for r in examples["training"]]
        if len(ids) != 32 or len(set(ids)) != 32:
            raise ValueError("diverse32 requires the complete32-prompt panel")
        return {"training": ids}
    if scope != "transfer-screen":
        raise ValueError("unknown periodic cost scope")
    return {split: list(fixed_panel(group)) for split, group in examples.items()}


def train(inputs, output):
    """Apply the explicit v2 numeric policy to the complete worker lifetime."""
    settings=read(Path(inputs)/'settings.json')
    from contextlib import nullcontext
    from oeis_learn.pipeline_validation.sizing_execution import execution_context
    with execution_context() if settings['recipe']['version']=='bigint-bootstrap/v2' else nullcontext():
        return _train(inputs,output)


def _train(inputs, output):
    import numpy as np
    import torch

    inputs, output = Path(inputs), Path(output)
    settings = read(inputs / "settings.json")
    recipe = settings["recipe"]
    engineering=settings.get('engineering')
    if engineering:
        from .engineering import parse as parse_engineering
        parse_engineering(engineering,fixture=bool(settings.get('fixture')))
        if recipe['version']!='bigint-bootstrap/v2' and not settings.get('fixture'):
            raise ValueError('explicit sized engineering recipe required')
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
    if contract.sha256!=settings['contract_sha256']:raise ValueError('worker contract identity mismatch')
    encoder, decoder, counts = models(recipe, settings["device"])
    if engineering and not engineering['resume'] and not settings.get('fixture'):
        initial=inputs/'initial.pt'
        if file_hash(initial)!=settings.get('initial_weights_sha256'):raise ValueError('frozen initial weights mismatch')
        weights=torch.load(initial,map_location='cpu',weights_only=True)
        encoder.load_state_dict(weights['encoder']);decoder.load_state_dict(weights['decoder']);del weights
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
        selection["panels"] = periodic_panels(
            examples, settings.get("cost_scope", "transfer-screen")
        )
    else:
        selection["panels"] = {split: [r["id"] for r in group] for split, group in examples.items()}
    restore_root=output if engineering and engineering['resume'] else inputs
    if engineering and engineering['resume'] and not (restore_root/'checkpoints/latest.json').exists():
        raise ValueError('complete engineering restart checkpoint missing')
    if (restore_root / "checkpoints/latest.json").exists():
        if restore_root!=output:
            shutil.copytree(restore_root / "checkpoints", output / "checkpoints", dirs_exist_ok=True)
        pointer = read(output / "checkpoints/latest.json")
        restored = CheckpointStore(output / "checkpoints", contract,policies=recipe.get('pipeline')).load(pointer)
        if engineering and restored['ledger'].snapshot()!=ledger.snapshot():
            raise ValueError('engineering restart must retain exact cumulative ledger')
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
        if (restore_root / "selected.json").exists():
            selection["best_checkpoint"] = read(restore_root / "selected.json")
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
    control=None;timer=None
    if engineering:
        from .engineering import Control
        from oeis_learn.tracking.foundation_metrics import PhaseTimer
        control=Control(settings,selection,output)
        deadline=min(deadline,control.deadline)
        timer=PhaseTimer(settings['device'],synchronized=False)
    batch_size=control.batch_size if control else recipe['learning']['batch_size']
    store = CheckpointStore(output / "checkpoints", contract, policies=recipe.get('pipeline'))
    if recipe.get('pipeline') and 'pipeline_boundary' not in selection:
        from .pipeline_jobs import JobLedger
        selection['pipeline_boundary'] = JobLedger().boundary(recipe['pipeline'])
    metrics = []
    pointer = None
    early = False
    successful=False

    def save():
        nonlocal pointer
        saved_at=time.monotonic()
        if control:control.checkpoint(selection)
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
        if control:control.component('checkpoint',time.monotonic()-saved_at,checkpoints=1)
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
            if control:target=49152
            while order["completed"] < target and time.monotonic() < deadline and (control is None or control.wants_update(order['completed'])):
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
                    if (order['cursor']>=len(order['permutation']) if control else order["cursor"] + batch_size > len(
                        order["permutation"]
                    )):
                        order["permutation"] = permutation(
                            rows, order_rng, batch_size
                        )
                        order["cursor"] = 0
                        order["epoch"] += 1
                    chosen = order["permutation"][
                        order["cursor"] : order["cursor"] + batch_size
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
                update_started=time.monotonic()
                record = _batch_loss(
                    encoder, decoder, batch, settings["device"], optimizer=optimizer,
                    **(dict(timer=timer,observe=False) if control else {})
                )
                order["completed"] += 1
                if control:control.update(batch,time.monotonic()-update_started)
                else:metrics.append(
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
                if (control.cycle_due() if control else (
                    order["completed"] % recipe["learning"]["validation_every"] == 0
                    or order["completed"] == recipe["learning"]["max_updates"]
                )):
                    from .diagnostics import evaluate_bigint as generate
                    validation_started=time.monotonic()
                    loss = validation_loss(
                        encoder,
                        decoder,
                        examples["parameter_development"] + examples["structural_development"],
                        settings["device"],
                        deadline=deadline,
                        **(dict(timer=timer) if control else {})
                    )
                    if control:
                        control.component('validation',time.monotonic()-validation_started,rows=loss['attempted'])
                        if not loss['complete']:raise TimeoutError('complete full-development validation required')
                    panel = [index[k] for ids in selection["panels"].values() for k in ids]
                    if control:
                        write(output/f"cycle-plan-{control.state['cycles']}.json",dict(
                            version='pipeline-soak-cycle/v1',cycle=control.state['cycles'],
                            prompts=[r['id'] for r in panel],mode='correct',attempts_per_prompt=1,
                            maximum_body_tokens=1024,maximum_prompt_seconds=60,snapshot_update=order['completed']))
                    generation_started=time.monotonic()
                    diagnostic = generate(
                        encoder, decoder, panel, recipe, synthesis, deadline=min(deadline,time.monotonic()+900) if control else deadline, periodic=True,
                        **(dict(evidence=control.evidence) if control else {})
                    )
                    if control:
                        total=sum(len(d['details']) for r in diagnostic['records'].values() for d in r['details'].values())
                        control.component('generation_native',time.monotonic()-generation_started,attempts=total,
                            verified=sum(bool(r['selections'].get('rescue')) for r in diagnostic['records'].values()))
                        control.cycled()
                    if recipe.get('pipeline'):
                        from .pipeline_jobs import merge_boundary
                        selection['pipeline_boundary'] = merge_boundary(selection['pipeline_boundary'],diagnostic['pipeline_boundary'])
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
                    if (
                        recipe["learning"]["curriculum"]
                        and order["completed"] % recipe["learning"]["validation_every"] == 0
                    ):
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
                    early = False if control else stopping.observe(visible["success"])
                    write(
                        output / "validation.json",
                        dict(visible=visible, teacher_forcing=loss, panels=selection["panels"]),
                    )
                    save()
                    if early:
                        break
                if control and control.pulse_due():
                    from oeis_learn.pipeline_validation.soak_native import pulse
                    def gpu_probe():
                        import math
                        began=time.monotonic();encoder.eval();decoder.eval()
                        value=_batch_loss(encoder,decoder,rows[:8],settings['device'],timer=timer,observe=False)
                        if not math.isfinite(value):raise FloatingPointError('nonfinite read-only probe')
                        control.component('gpu_readonly_probe',time.monotonic()-began,rows=min(8,len(rows)))
                    pulse(synthesis,control,selection,recipe,deadline,gpu_probe=gpu_probe)
            if selection["selected_update"] is None:
                selection["selected_update"] = order["completed"]
            if control and control.cfg['mode'] in {'batch','observation'}:
                validation_started=time.monotonic()
                loss=validation_loss(encoder,decoder,examples['parameter_development']+examples['structural_development'],settings['device'],deadline=deadline,timer=timer)
                control.component('validation',time.monotonic()-validation_started,rows=loss['attempted'])
                if not loss['complete']:raise TimeoutError('complete batch-arm validation required')
            successful=True
    finally:
        if control:
            engineering_result=control.finish(completed=successful)
        save()
        if control:write(output/'engineering.json',engineering_result)
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
        **(dict(engineering=read(output/'engineering.json')) if control else {}),
    )


def _load_models(inputs, settings):
    from .bigint_state import CheckpointStore

    encoder, decoder, counts = models(settings["recipe"], settings["device"])
    contract = ExperimentContract(**settings["contract"])
    store = CheckpointStore(Path(inputs) / "checkpoints", contract,policies=settings['recipe'].get('pipeline'))
    loaded = store.load(settings["checkpoint"])
    encoder.load_state_dict(loaded["states"]["weights"]["encoder"])
    decoder.load_state_dict(loaded["states"]["weights"]["decoder"])
    if settings.get("ablation"):
        from oeis_learn.encoder.ablation import InputMask

        if not hasattr(encoder, "group_mask"):
            raise ValueError("four-group attribution requires hybrid-chunks")
        encoder.group_mask = InputMask(**settings["ablation"])
        counts = dict(
            counts,
            **encoder.ablation_report(),
            analysis_mask=settings["ablation"],
            post_training=True,
            retraining_claim=False,
        )
    return encoder, decoder, counts


def generate(inputs, output):
    settings=read(Path(inputs)/'settings.json')
    from contextlib import nullcontext
    from oeis_learn.pipeline_validation.sizing_execution import execution_context
    with execution_context() if settings['recipe']['version']=='bigint-bootstrap/v2' else nullcontext():
        return _generate(inputs,output)


def _generate(inputs, output):
    from .diagnostics import evaluate_bigint as synthesize

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
        body = {
            k: record[k] for k in ("selections", "arm_evidence", "complete", "family", "strata")
        }
        if len(canonical({key: body})) > 4 << 20:
            raise ValueError("single sealed selection exceeds transport cap")
        if len(canonical(dict(group, **{key: body}))) > 4 << 20:
            flush()
        group[key] = body
    flush()
    write_diagnostic_evidence(output, rows, result["records"])
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
        from .candidate_archive import retain_alternatives, duplicate_key

        merged = {}
        for value in record["details"].values():
            for alternative in value["archive"]["retained"]:
                if (
                    alternative["id"] not in merged
                    or (duplicate_key(alternative) < duplicate_key(merged[alternative['id']]) if settings['recipe'].get('pipeline')
                        else alternative["cost"] < merged[alternative["id"]]["cost"])
                ):
                    merged[alternative["id"]] = alternative
        body = retain_alternatives(list(merged.values()),policy='structure-pareto/v2' if settings['recipe'].get('pipeline') else 'structure-pareto/v1')
        body["portfolio_omissions"] = {
            mode: value["archive"]["omitted"] for mode, value in record["details"].items()
        }
        payload = canonical(body)
        name = f"archive/{identity(row['id'])[7:]}.json"
        if archive_bytes + len(payload) <= min(
            settings.get("archive_remaining", settings["recipe"]["archive"]["max_bytes"]),
            settings["recipe"]["archive"]["max_bytes"],
        ):
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
                (attempt.get("native") or {}).get("uniqueness") == "multiple"
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
            reference = dict(
                path=name,
                bytes=(output / name).stat().st_size,
                sha256=file_hash(output / name),
                reason="detail chunk cap",
            )
            (output / name).unlink()
            details.append(reference)
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


def write_diagnostic_evidence(output, rows, records):
    """Retain every prompt's compact failure evidence before sampled examples."""
    from .bigint_diagnostics import compact_attempts

    entries = []
    for row in rows:
        record = records[row["id"]]
        correct = record["details"]["correct"]["details"]
        body = dict(
            version="bigint-prompt-evidence/v1",
            id=row["id"],
            prompt=dict(indices=row["indices"], values=row["values"]),
            greedy_source=correct[0]["source"] if correct else None,
            arms=record["arm_evidence"],
            attempts={
                mode: compact_attempts(value["details"])
                for mode, value in record["details"].items()
            },
        )
        entries.append((row["id"], body))
    _evidence_chunks(output, entries)


def _evidence_chunks(output, entries):
    from .common import canonical

    group = {}
    number = 0
    for key, body in entries:
        if len(canonical({key: body})) > 1 << 20:
            raise ValueError("single prompt evidence exceeds1 MiB")
        if len(canonical(dict(group, **{key: body}))) > 1 << 20:
            write(Path(output) / f"evidence/chunk-{number:04}.json", group)
            number += 1
            group = {}
        group[key] = body
    if group:
        write(Path(output) / f"evidence/chunk-{number:04}.json", group)


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
    outcomes = result.pop("outcomes")
    write(output / "outcomes.json", outcomes)
    _evidence_chunks(
        output,
        (
            (key, dict(version="bigint-score-evidence/v1", outcomes=value))
            for key, value in outcomes.items()
        ),
    )
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
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.synthesis.admission import reference_identity, source_identity
    from oeis_learn.synthesis.reference import evaluate

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
