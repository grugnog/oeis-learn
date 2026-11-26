"""Bounded installed measurements on strict training rows for the selected generator."""
from pathlib import Path
import math
import time

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal


def batch(rows, device):
    import torch
    from oeis_learn.decoder.bigint_codec import BOS_ID, PAD_ID
    ids = [[BOS_ID, *row["tokens"]] for row in rows]
    tokens = torch.full((len(rows), max(map(len, ids))), PAD_ID, dtype=torch.long, device=device)
    for i, values in enumerate(ids):
        tokens[i, :len(values)] = torch.tensor(values, device=device)
    return [list(map(int, row["values"][:20])) for row in rows], tokens


def verify_sample(sample):
    from oeis_learn.generalization.progressive.training_rows import first_divergence, validate_row
    if set(sample) != {"positive", "negative", "label"}:
        raise ValueError("closed training pair required")
    row = validate_row(sample["positive"], native_verified=True)
    alt = sample["negative"]
    if alt is None:
        if sample["label"] is not None:
            raise ValueError("preference without verified negative")
    else:
        validate_row(alt, native_verified=True)
        if (row["indices"] != alt["indices"] or row["values"][:20] != alt["values"][:20]
                or row["values"][20:] == alt["values"][20:] or sample["label"] is None
                or first_divergence(row["tokens"], alt["tokens"], [row["tokens"]]) != sample["label"]):
            raise ValueError("divergence does not bind to the positive training target")


def gradient(profile, samples, *, client=None):
    import torch
    from oeis_learn.decoder.bigint_codec import PAD_ID
    from oeis_learn.generalization.progressive.models import construct
    from oeis_learn.rl.sft_trainer import first_divergence_loss, teacher_forced_logits

    sample = next((s for s in samples if s["label"] is not None), None)
    if sample is None:
        raise ValueError("no eligible first-divergence training row")
    verify_sample(sample)
    if client is not None:
        for row in (sample["positive"], sample["negative"]):
            checked = client.execute(row["source"], row["indices"], deadline=time.monotonic() + 4)
            if checked.get("outcome") != "ok" or checked.get("values") != row["values"]:
                raise ValueError("installed native training pair disagrees with original 100 terms")
    encoder, decoder, _ = construct(profile, device="cpu")
    seq, tokens = batch([sample["positive"]], "cpu")
    logits = teacher_forced_logits(encoder, decoder, seq, tokens, device="cpu", pad_id=PAD_ID)
    loss = first_divergence_loss(logits, tokens[:, 1:], [sample["label"]], pad_id=PAD_ID)
    grads = torch.autograd.grad(loss, [*encoder.parameters(), *decoder.parameters()], allow_unused=True)
    norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("no finite nonzero training-only parameter gradient")
    return dict(gradient_l2=norm, gradient_probe_rows=1, gradient_loss=float(loss.detach()),
                parameter_bytes=sum(p.numel() * p.element_size() for p in (*encoder.parameters(), *decoder.parameters())),
                native_verified_terms=200 if client is not None else 0)


def measured_updates(profile, batches, arm, output, *, device, microbatch):
    """Compare the next real AdamW update before and after a cold full-state restore."""
    import random
    import numpy as np
    import torch
    from oeis_learn.decoder.bigint_codec import PAD_ID
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.pipeline_validation.progressive_qualification import _same, require_hip
    from oeis_learn.pipeline_validation.progressive_training import PolicyUpdate, RecoverableUpdate, resolve_update_policy

    if device == "cuda":
        require_hip()
    policy = resolve_update_policy(profile, objective=arm, effective_batch=32, microbatch_size=microbatch,
                                   schedule_horizon_tokens=4_000_000, weight_decay=0)
    runner = PolicyUpdate(profile, policy, device=device)
    initial = model_snapshot(runner.encoder, runner.decoder)
    schedule = []
    for samples in batches:
        if len(samples) != 32:
            raise ValueError("whole trial 05 batch required")
        seq, tokens = batch([s["positive"] for s in samples], device)
        ids = [s["positive"]["draw_identity"] for s in samples]
        schedule.append((seq, tokens, ids, [s["label"] for s in samples], int(tokens[:, 1:].ne(PAD_ID).sum())))
    if len(schedule) != 2:
        raise ValueError("two ordered full updates required for continuation proof")
    charged = [v[-1] for v in schedule]
    cap = sum(charged)
    store = GeneralizationCheckpointStore(output, digest([policy["identity"], [r[2] for r in schedule]]), cap)

    def update(adapter, ledger, i):
        seq, tokens, _, labels, count = schedule[i]
        if device == "cuda":
            torch.cuda.synchronize()
        began = time.monotonic_ns()
        metrics, new = adapter.update(seq, tokens, ledger=ledger.reserve(str(i), count),
                                     preference_labels=labels if arm == "first_divergence" else None)
        if device == "cuda":
            torch.cuda.synchronize()
        return metrics, new, max(1, (time.monotonic_ns() - began) // 1000)

    first, ledger, cold = update(runner, Ledger(cap), 0)
    cut_hash = model_snapshot(runner.encoder, runner.decoder)
    context = dict(weights=dict(encoder=runner.encoder.state_dict(), decoder=runner.decoder.state_dict()),
                   optimizer=runner.state_dict(), scheduler={"completed_updates": 1},
                   data=dict(cursor=32, pending_batch=None), replay={}, history={"charge": charged[0]}, pins={},
                   selection=dict(initial=initial, consumed=digest(schedule[0][2]), objective=arm),
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                            device=torch.cuda.get_rng_state_all() if device == "cuda" else []))
    started = time.monotonic_ns()
    checkpoint = store.save(context, ledger)
    save_us = (time.monotonic_ns() - started) // 1000
    del context
    _, final_ledger, warm = update(runner, ledger, 1)
    expected = model_snapshot(runner.encoder, runner.decoder)
    expected_opt = runner.state_dict()
    del runner
    started = time.monotonic_ns()
    restored = GeneralizationCheckpointStore(output, store.contract, cap).load(checkpoint)
    resumed = PolicyUpdate(profile, policy, device=device)
    state = restored["states"]
    resumed.encoder.load_state_dict(state["weights"]["encoder"])
    resumed.decoder.load_state_dict(state["weights"]["decoder"])
    resumed.load_state_dict(state["optimizer"])
    RecoverableUpdate._rng_restore(state["rng"])
    restore_us = (time.monotonic_ns() - started) // 1000
    if (model_snapshot(resumed.encoder, resumed.decoder) != cut_hash or state["data"]["cursor"] != 32
            or state["selection"] != dict(initial=initial, consumed=digest(schedule[0][2]), objective=arm)):
        raise ValueError("full training-state or row cursor differed after restore")
    _, repeated, replay_us = update(resumed, restored["ledger"], 1)
    if (cut_hash == initial or expected == cut_hash or model_snapshot(resumed.encoder, resumed.decoder) != expected
            or not _same(expected_opt, resumed.state_dict()) or repeated.snapshot() != final_ledger.snapshot()):
        raise ValueError("actual parameter change and exact next update required")
    files = [p for p in Path(output).rglob("*") if p.is_file()]
    return dict(initial=initial, final=expected, row_schedule=digest([v[2] for v in schedule]),
                committed_updates=final_ledger.updates, charged_tokens=charged,
                preference_rows=first.get("preference_rows", 0), full_restore=True, actual_update=True,
                update_us=[cold, warm], save_us=save_us, restore_us=restore_us, replay_us=replay_us,
                checkpoint_bytes=sum(p.stat().st_size for p in files), checkpoint_files=len(files))


def usage_summary(directory, total_us, *, interval=1):
    import json
    # Integrate only adjacent valid one-second samples. Missing counters and long
    # gaps reduce observed coverage and never contribute artificial idle time.
    previous = None
    covered = busy = 0.0
    samples = valid = 0
    path = Path(directory) / "samples.jsonl"
    if not path.is_file():
        return dict(samples=0, readable_samples=0, covered_us=0, busy_us=None, role_wall_us=total_us)
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            samples += 1
            raw = (row.get("gpu") or {}).get("gpu_busy_percent")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = None
            if value is None or not 0 <= value <= 100:
                previous = None
                continue
            valid += 1
            now = row["time"]
            if previous:
                dt = now - previous[0]
                if 0 < dt <= 2 * interval:
                    covered += dt
                    busy += dt * (previous[1] + value) / 200
            previous = now, value
    return dict(samples=samples, readable_samples=valid, covered_us=round(covered * 1e6),
                busy_us=round(busy * 1e6) if covered else None, role_wall_us=total_us)


def run_inside(request, output):
    import os
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.generalization.progressive.models import validate_profile
    from oeis_learn.synthesis.client import SynthesisClient
    from .progressive_qualification import hardware_facts, require_hip

    fields = {"version", "identity", "source", "image", "engine", "stage", "data", "parent", "physical"}
    if (set(request) != fields or seal(request) != request or request["version"] != "round028-probe-request/v1"
            or digest(source_snapshot()) != request["source"] or request["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
            or request["engine"] != os.environ["OEIS_QUALIFICATION_ENGINE"]):
        raise ValueError("candidate source, immutable image and registered request required")
    base = load("/work/inputs/training.json", maximum=128 << 20)
    if (seal(base) != base or base["identity"] != request["data"] or base["version"] != "round028-training-probe/v1"
            or len(base["batches"]) != 2 or any(len(b) != 32 for b in base["batches"])):
        raise ValueError("sealed training-only rehearsal input required")
    profile = validate_profile(base["profile"])
    for samples in base["batches"]:
        for sample in samples:
            verify_sample(sample)
    began = time.monotonic_ns()
    stage = request["stage"]
    if stage == "cpu" and request["parent"] is None:
        with SynthesisClient(digest(["round028-gradient", request["identity"]])) as native:
            details = gradient(profile, base["batches"][0], client=native)
            details["native"] = native.runtime_identity("bigint-body/v1")
    elif stage == "hip" and request["parent"]:
        torch = require_hip()
        torch.cuda.reset_peak_memory_stats()
        from oeis_learn.tracking.pipeline_profile import Recorder
        nodes = sorted(Path("/dev/dri").glob("renderD*"))
        telemetry = Path(output) / "telemetry"
        started = time.monotonic_ns()
        with Recorder(telemetry, instrumented=True, render_node=nodes[0] if len(nodes) == 1 else None,
                      host_counters=False) as recorder:
            arms = {}
            for arm in ("ordinary", "first_divergence"):
                with recorder.phase(arm):
                    arms[arm] = measured_updates(profile, base["batches"], arm, Path(output) / arm,
                                     device="cuda", microbatch=base["microbatch"])
        usage = usage_summary(telemetry, (time.monotonic_ns() - started) // 1000)
        a, b = arms.values()
        if any(a[k] != b[k] for k in ("initial", "row_schedule", "charged_tokens")) or a["final"] == b["final"]:
            raise ValueError("matched initialization, exposure and distinct treatment effect required")
        details = dict(arms=arms, hardware=hardware_facts(), telemetry=usage, peak_device_bytes=torch.cuda.max_memory_allocated())
    else:
        raise ValueError("closed CPU signal followed by matching HIP rehearsal required")
    result = seal(dict(version="round028-probe-producer/v1", request=request["identity"], stage=stage,
                       elapsed_us=(time.monotonic_ns() - began) // 1000, details=details))
    atomic_write(Path(output) / "producer.json", result)
    return result


if __name__ == "__main__":
    run_inside(load("/work/inputs/request.json"), "/work/output")
