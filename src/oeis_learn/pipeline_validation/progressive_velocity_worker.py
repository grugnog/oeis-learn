"""Installed round029 learner and disposable update/restore rehearsal."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from oeis_learn.generalization.budgets import Ledger
from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
from oeis_learn.generalization.progressive.training_rows import matched_batch
from oeis_learn.generalization.progressive.velocity_trials import exposure, update_policy


def _matching_observation(profile, rows, order, donors, *, device, batch_size):
    from math import isfinite
    from statistics import median
    import torch
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.decoder.bigint_codec import PAD_ID
    from oeis_learn.generalization.progressive.models import construct
    from oeis_learn.rl.sft_trainer import (
        calibrate_matching,
        prefix_matching_loss,
        teacher_forced_logits,
    )
    from .progressive_generator_probe import batch

    began = time.monotonic_ns()
    encoder, decoder, _ = construct(profile, device=device)
    initial_weights = model_snapshot(encoder, decoder)
    positive_losses, aux_losses = [], []
    wins = passes = 0
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            keys = order[start : start + batch_size]
            seq, tokens = batch([rows[k] for k in keys], device)
            selected = [i for i, k in enumerate(keys) if k in donors]
            logits = teacher_forced_logits(
                encoder, decoder, seq, tokens, device=device, pad_id=PAD_ID
            )
            labels = tokens[:, 1:]
            nll = -logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            mask = labels.ne(PAD_ID)
            positive_losses.extend(((nll * mask).sum(1) / mask.sum(1)).tolist())
            if not selected:
                continue
            donor_seq = [list(map(int, donors[keys[i]])) for i in selected]
            other = teacher_forced_logits(
                encoder, decoder, donor_seq, tokens[selected], device=device, pad_id=PAD_ID
            )
            passes += 1
            for j, i in enumerate(selected):
                aux, win = prefix_matching_loss(
                    logits[i : i + 1], other[j : j + 1], labels[i : i + 1], [0], pad_id=PAD_ID
                )
                aux_losses.append(float(aux))
                wins += win
    if device == "cuda":
        torch.cuda.synchronize()
    try:
        weight = calibrate_matching(positive_losses, aux_losses)
        reason = None
    except ValueError:
        weight = None
        reason = (
            "no eligible training pairs"
            if not aux_losses
            else "nonfinite or nonpositive training calibration"
        )

    def observed(values):
        value = median(values) if values and all(isfinite(v) and v > 0 for v in values) else None
        return value if value is not None and isfinite(value) else None

    return dict(
        initial_weights=initial_weights,
        positive_rows=len(positive_losses),
        eligible_rows=len(aux_losses),
        wins=wins,
        extra_passes=passes,
        median_positive=observed(positive_losses),
        median_auxiliary=observed(aux_losses),
        weight=weight,
        reason=reason,
        elapsed_us=(time.monotonic_ns() - began) // 1000,
        device=device,
    )


def fit_calibration(profile, rows, order, donors, *, device, batch_size=32):
    """Diagnostic compatibility wrapper for the original learner qualification."""
    result = _matching_observation(
        profile, rows, order, donors, device=device, batch_size=batch_size
    )
    if result["weight"] is None:
        raise ValueError(result["reason"])
    return result["weight"]


def measure_calibration(request, *, device):
    """Installed worker receives only ordinary training prefixes, decoder targets and proven donors."""
    from .progressive_velocity_matching import check_projection

    check_projection(request)
    measured = _matching_observation(
        request["profile"],
        request["rows"],
        request["order"],
        request["donors"],
        device=device,
        batch_size=32,
    )
    if measured["initial_weights"] != request["initial_weights"]:
        raise ValueError("matching calibration did not start from the registered random weights")
    return seal(
        dict(version="round029-matching-observation/v1", request=request["identity"], **measured)
    )


def optimizer_digest(runner):
    """Fingerprint the whole AdamW state without copying a second model onto the device."""
    import hashlib
    import torch

    state = runner.optimizer.state_dict()
    moments = []
    for param, fields in sorted(state["state"].items()):
        for name, tensor in sorted(fields.items()):
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            moments.append(
                [
                    param,
                    name,
                    str(tensor.dtype),
                    list(tensor.shape),
                    hashlib.sha256(raw).hexdigest(),
                ]
            )
    return digest([state["param_groups"], moments])


def train_to(directory, package, *, target, cap, device, interrupt=False):
    """Resume a committed useful-token cut. Each issued update keeps its physical debit."""
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from .progressive_generator_probe import batch
    from .progressive_training import PolicyUpdate, RecoverableUpdate
    from .generalization_persistence import snapshot_due

    if (
        package != seal(package)
        or type(target) is not int
        or not 0 <= target <= cap
        or set(package)
        != {"version", "identity", "profile", "policy", "order", "rows", "donors", "registration"}
        or package["version"] != "round029-learner-input/v1"
    ):
        raise ValueError("sealed registered learner projection and target required")
    rows, order = package["rows"], package["order"]
    if not order or set(order) != set(rows) or not package["donors"].keys() <= rows.keys():
        raise ValueError("ordered training-only support required")
    profile, policy = package["profile"], package["policy"]
    size = policy["effective_batch"]
    contract = digest([package["identity"], cap])
    area = Path(directory)
    store = GeneralizationCheckpointStore(area / "checkpoint", contract, cap)
    pointer_path = store.root / "generalization-latest.json"
    debits = BudgetJournal(area / "accounting", contract, dict(dict.fromkeys(KEYS, 0), tokens=cap))
    runner = PolicyUpdate(profile, policy, device=device)
    useful = updates = cursor = 0
    chain = digest(None)
    metrics = {}
    pointer = None
    if pointer_path.exists():
        pointer = load(pointer_path)
        saved = store.load(pointer)
        state = saved["states"]
        useful, updates = state["history"]["useful_tokens"], saved["ledger"].updates
        prior = exposure(package, rows, updates=updates, batch=size)
        cursor, chain = prior["cursor"], prior["order_digest"]
        if (
            prior["useful_tokens"] != useful
            or state["data"] != dict(cursor=cursor, pending_batch=None)
            or state["selection"] != dict(order_digest=chain, package=package["identity"])
            or saved["ledger"].consumed > debits.snapshot()["consumed"]["tokens"]
        ):
            raise ValueError("checkpoint row cursor, charge or positive-token replay differs")
        runner.encoder.load_state_dict(state["weights"]["encoder"])
        runner.decoder.load_state_dict(state["weights"]["decoder"])
        runner.load_state_dict(state["optimizer"])
        RecoverableUpdate._rng_restore(state["rng"])
        metrics = state["history"]["metrics"]
        if useful > target and target <= useful - prior["last_update_tokens"]:
            raise ValueError("target precedes the last committed update")
        del state, saved
    if interrupt:
        if not pointer:
            raise ValueError("backward fault requires an original completed cut")

        def stop(grad):
            import torch

            peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
            atomic_write(
                area / "backward-interrupted.json",
                seal(dict(package=package["identity"], cursor=cursor, peak_device_bytes=peak)),
            )
            os.kill(os.getpid(), signal.SIGKILL)

        next(p for p in runner.decoder.parameters() if p.requires_grad).register_hook(stop)
    for key in debits.snapshot()["pending"]:
        debits.commit(key)
    accounting = debits.snapshot()
    charged = accounting["consumed"]["tokens"]
    ledger = Ledger(cap, consumed=charged, updates=updates)
    attempt = len(accounting["committed"])
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def save():
        context = dict(
            scheduler={"completed_updates": ledger.updates},
            data=dict(cursor=cursor, pending_batch=None),
            replay={},
            history=dict(useful_tokens=useful, metrics=metrics),
            pins={},
            selection=dict(order_digest=chain, package=package["identity"]),
        )
        return store.save(RecoverableUpdate.capture_runner(runner, context), ledger)

    last_save = time.monotonic()
    if pointer is None:
        pointer = save()
    while useful < target:
        ids = matched_batch(package, cursor, size)
        seq, tokens = batch([rows[k] for k in ids], device)
        count = sum(len(rows[k]["tokens"]) for k in ids)
        if ledger.consumed + count > cap:
            raise ValueError("remaining grant cannot fund the next whole update")
        reserved = debits.reserve(
            "learner",
            "update",
            attempt,
            dict(dict.fromkeys(KEYS, 0), tokens=count),
            boot=boot,
            monotonic_us=time.monotonic_ns() // 1000,
        )
        matching = None
        if policy["matching_weight"] is not None:
            matching = [
                list(map(int, package["donors"][k])) if k in package["donors"] else None
                for k in ids
            ]
        observed, ledger = runner.update(
            seq, tokens, ledger=ledger.reserve(str(attempt), count), matching_donors=matching
        )
        metrics = {
            k: observed.get(k, 0)
            for k in (
                "matching_passes",
                "matching_eligible",
                "matching_wins",
                "matching_loss",
                "matching_forward_us",
                "gradient_norm",
            )
        }
        debits.commit(reserved["key"])
        useful += count
        cursor += size
        attempt += 1
        chain = digest([chain, ids])
        if useful >= target or snapshot_due("hourly/v1", last_save, time.monotonic()):
            pointer = save()
            last_save = time.monotonic()
    replay = exposure(
        package, rows, updates=ledger.updates, batch=size, discarded=ledger.consumed - useful
    )
    if (replay["order_digest"], replay["useful_tokens"]) != (chain, useful):
        raise ValueError("independent positive-stream replay differs")
    return dict(
        checkpoint=pointer,
        weights=model_snapshot(runner.encoder, runner.decoder),
        optimizer=optimizer_digest(runner),
        scheduler=runner.scheduler.state_dict(),
        next_rows=digest(matched_batch(package, cursor, size)),
        metrics=metrics,
        **replay,
    )


def _disposable_rows(native=None):
    """Generated, exposed calibration fixtures. A fixed target demonstrably rejects the donor."""
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.generalization.progressive.training_rows import matching_proof
    from oeis_learn.synthesis.admission import VisiblePrompt
    from oeis_learn.synthesis.reference import evaluate

    source = "(program (return $index))"
    alternate = "(program (return (add $index (int 1))))"
    rows, donors = {}, {}
    # Every slot is in the same positive sign/magnitude band. All examples are disposable.
    for start in (128, 132, 136, 140, 144):
        ns = [str(i) for i in range(start, start + 20)]
        visible = []
        for program in (source, alternate):
            reference = evaluate(program, ns)
            if reference.outcome != "ok":
                raise ValueError("invalid training witness")
            values = list(reference.values)
            if native:
                checked = native.execute(program, ns)
                if checked.get("outcome") != "ok" or checked.get("values") != values:
                    raise ValueError("native/reference mismatch on exposed training input")
            visible.append(values)
            key = digest([program, ns])
            rows[key] = dict(values=values, tokens=encode_body(program))
        proof = matching_proof(encode_body(source), VisiblePrompt(tuple(ns), tuple(visible[1])))
        if proof["status"] != "eligible":
            raise ValueError("disposable matching label is ambiguous")
        donors[digest([source, ns])] = visible[1]
    return rows, donors


def rehearsal_counts():
    rows, _ = _disposable_rows()
    stream = {"order": sorted(rows)}
    result = {}
    for size in (16, 32):
        first = exposure(stream, rows, updates=1, batch=size)["useful_tokens"]
        both = exposure(stream, rows, updates=2, batch=size)["useful_tokens"]
        result[str(size)] = dict(
            useful_tokens=both, discarded_tokens=both - first, charged_tokens=2 * both - first
        )
    return result


def rehearsal_work():
    counts = rehearsal_counts()
    rows, _ = _disposable_rows()
    # Each batch checks every exposed witness natively. Each literal probe can need a fit and replay.
    return dict(
        tokens=sum(row["useful_tokens"] + row["charged_tokens"] for row in counts.values()),
        native_work=len(counts) * len(rows) + 2 * 2,
    )


def literal_probes(native):
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.generalization.progressive.training_rows import matching_proof
    from oeis_learn.synthesis.admission import VisiblePrompt

    indices = tuple(map(str, range(20)))
    tokens = encode_body("(program (return (int 3)))")
    controls = {"constant": tuple("7" for _ in indices), "varying": indices}
    return {
        name: matching_proof(tokens, VisiblePrompt(indices, values), client=native)
        for name, values in controls.items()
    }


def disposable_batch(profile, batch_size, *, device="cpu", native=None):
    if batch_size not in (16, 32):
        raise ValueError("unregistered qualification batch")
    rows, donors = _disposable_rows(native)
    order = sorted(rows)
    weight = (
        fit_calibration(profile, rows, order, donors, device=device) if batch_size == 32 else None
    )
    policy = update_policy(
        profile,
        "02" if batch_size == 32 else "04",
        microbatch=min(8, batch_size),
        matching_weight=weight,
    )
    return seal(
        dict(
            version="round029-learner-input/v1",
            registration=digest("disposable-learner-029"),
            profile=profile,
            policy=policy,
            order=order,
            rows=rows,
            donors=donors if weight else {},
        )
    )


def rehearse_batch(directory, profile, *, batch_size, device, native=None):
    from tempfile import TemporaryDirectory
    from .progressive_pilot_capacity import owned_allocations

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    package = disposable_batch(profile, batch_size, device=device, native=native)
    first = exposure(package, package["rows"], updates=1, batch=batch_size)["useful_tokens"]
    second = exposure(package, package["rows"], updates=2, batch=batch_size)["useful_tokens"]
    cap = second + second - first
    with TemporaryDirectory(dir=directory) as private:
        area = Path(private)
        path = area / "package.json"
        atomic_write(path, package)
        expected = train_to(area / "reference", package, target=second, cap=cap, device=device)
        # The reference trajectory is disposable once its comparison evidence is in memory.
        import shutil

        shutil.rmtree(area / "reference")
        cut = train_to(area / "recover", package, target=first, cap=cap, device=device)
        if device == "cuda":
            import torch
            from .progressive_qualification import _cleanup_memory

            _cleanup_memory(torch)
        began = time.monotonic_ns()
        module = "oeis_learn.pipeline_validation.progressive_velocity_worker"
        process = subprocess.run(
            [
                sys.executable,
                *(["-I"] if device == "cuda" else []),
                "-m",
                module,
                "interrupt",
                str(path),
                str(area / "recover"),
                str(second),
                str(cap),
                device,
            ]
        )
        failed_us = (time.monotonic_ns() - began) // 1000
        if (
            process.returncode != -signal.SIGKILL
            or not (area / "recover/backward-interrupted.json").exists()
        ):
            raise ValueError("disposable backward did not reach the charged interruption")
        fault = load(area / "recover/backward-interrupted.json")
        if fault != seal(fault) or (fault["package"], fault["cursor"]) != (
            package["identity"],
            batch_size,
        ):
            raise ValueError("backward interruption belongs to a different cut")
        actual = train_to(area / "recover", package, target=second, cap=cap, device=device)
        repeated = train_to(area / "recover", package, target=second, cap=cap, device=device)
        if (
            actual != repeated
            or actual["next_rows"] != expected["next_rows"]
            or actual["weights"] != expected["weights"]
            or actual["optimizer"] != expected["optimizer"]
            or actual["scheduler"] != dict(expected["scheduler"], tokens=cap)
            or cut["weights"] == actual["weights"]
            or actual["discarded_tokens"] != second - first
            or actual["updates"] != 2
        ):
            raise ValueError("full-state continuation differs or discarded work was refunded")
        allocated = owned_allocations([area / "recover/checkpoint"])
        return dict(
            batch=batch_size,
            full_restore=True,
            uninterrupted_weights=expected["weights"],
            restored_weights=actual["weights"],
            uninterrupted_next_rows=expected["next_rows"],
            uninterrupted_optimizer=expected["optimizer"],
            restored_optimizer=actual["optimizer"],
            scheduler=actual["scheduler"],
            interrupted_peak_device_bytes=fault["peak_device_bytes"],
            next_rows=actual["next_rows"],
            useful_updates=2,
            discarded_updates=1,
            failed_us=failed_us,
            useful_tokens=second,
            charged_tokens=actual["charged_tokens"],
            discarded_tokens=actual["discarded_tokens"],
            checkpoint_bytes=allocated["bytes"],
            checkpoint_inodes=allocated["inodes"],
            matching_weight=package["policy"]["matching_weight"],
            **actual["metrics"],
        )


def verify_rehearsal(producer, request):
    if (
        producer != seal(producer)
        or producer["version"] != "round029-device-result/v1"
        or producer["request"] != request["identity"]
        or producer["device"] != "hip"
        or set(producer["batches"]) != {"16", "32"}
        or producer["peak_device_bytes"] <= 0
    ):
        raise ValueError("installed HIP rehearsal receipt differs")
    expected = rehearsal_counts()
    for name, row in producer["batches"].items():
        if (
            row["batch"] != int(name)
            or any(row[k] != v for k, v in expected[name].items())
            or row["full_restore"] is not True
            or row["useful_updates"] != 2
            or row["discarded_updates"] != 1
            or row["charged_tokens"] != row["useful_tokens"] + row["discarded_tokens"]
            or row["restored_weights"] != row["uninterrupted_weights"]
            or row["next_rows"] != row["uninterrupted_next_rows"]
            or row["restored_optimizer"] != row["uninterrupted_optimizer"]
            or row["scheduler"]["updates"] != 2
            or row["scheduler"]["tokens"] != row["charged_tokens"]
            or not 0 < row["interrupted_peak_device_bytes"] <= producer["peak_device_bytes"]
            or min(
                row["discarded_tokens"],
                row["checkpoint_bytes"],
                row["checkpoint_inodes"],
                row["failed_us"],
            )
            <= 0
            or (name == "32") != (row["matching_passes"] > 0 and row["matching_eligible"] > 0)
        ):
            raise ValueError("invalid batch, matching or restore evidence")


def main():
    if sys.argv[1:]:
        action, path, area, target, cap, device = sys.argv[1:]
        if action != "interrupt":
            raise ValueError("unknown qualification fault")
        train_to(
            Path(area), load(path), target=int(target), cap=int(cap), device=device, interrupt=True
        )
        raise ValueError("expected a backward interruption")
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.tracking.pipeline_profile import Recorder
    from .progressive_generator_probe import usage_summary
    from .progressive_qualification import require_hip, hardware_facts
    from .progressive_roles import ROUND029_ROLE_POLICY

    req = load("/work/inputs/request.json")
    if (
        req != seal(req)
        or req["version"] != "round029-device-request/v1"
        or req["batch_sizes"] != [16, 32]
        or req["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
        or req["source"] != digest(source_snapshot())
        or req["role_policy"] != ROUND029_ROLE_POLICY
    ):
        raise ValueError("installed worker source, image or projected request differs")
    torch = require_hip()
    torch.cuda.reset_peak_memory_stats()
    out = Path("/work/output")
    began = time.monotonic_ns()
    nodes = sorted(Path("/dev/dri").glob("renderD*"))
    with (
        Recorder(
            out / "telemetry",
            instrumented=True,
            render_node=nodes[0] if len(nodes) == 1 else None,
            host_counters=False,
        ) as recorder,
        SynthesisClient(req["identity"]) as native,
    ):
        batches = {}
        for size in req["batch_sizes"]:
            with recorder.phase("batch-" + str(size)):
                batches[str(size)] = rehearse_batch(
                    out, req["profile"], batch_size=size, device="cuda", native=native
                )
        with recorder.phase("matching-literal-probes"):
            probes = literal_probes(native)
    wall = (time.monotonic_ns() - began) // 1000
    record = seal(
        dict(
            version="round029-device-result/v1",
            request=req["identity"],
            device="hip",
            hardware=hardware_facts(),
            batches=batches,
            literal_probes=probes,
            peak_device_bytes=max(
                torch.cuda.max_memory_allocated(),
                *(b["interrupted_peak_device_bytes"] for b in batches.values()),
            ),
            telemetry=usage_summary(out / "telemetry", wall),
            elapsed_us=wall,
        )
    )
    verify_rehearsal(record, req)
    atomic_write(out / "producer.json", record)


if __name__ == "__main__":
    main()
