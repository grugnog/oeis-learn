"""Frozen analysis workload; no DurableRun, ledger, selection or resume writes."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from functools import wraps
from pathlib import Path
import json
import os
import random
import resource
import signal
import time

from .common import digest, file_hash, read, write
from oeis_learn.tracking.pipeline_profile import Recorder, interval_union


def group_order(group):
    return ("ordinary", "telemetry") if group % 2 == 0 else ("telemetry", "ordinary")


def snapshot(encoder, decoder, optimizer):
    import numpy as np
    import torch

    return deepcopy(
        dict(
            encoder=encoder.state_dict(),
            decoder=decoder.state_dict(),
            optimizer=optimizer.state_dict(),
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )
    )


def restore(state, encoder, decoder, optimizer):
    import numpy as np
    import torch

    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    optimizer.load_state_dict(deepcopy(state["optimizer"]))
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def parameter_digest(encoder, decoder):
    import hashlib
    import torch

    h = hashlib.sha256()
    for name, value in sorted(
        [
            (f"{prefix}.{k}", v)
            for prefix, model in (("encoder", encoder), ("decoder", decoder))
            for k, v in model.state_dict().items()
        ]
    ):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return "sha256:" + h.hexdigest()


def snapshot_bytes(*states):
    import torch

    seen, total = set(), {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total[key[0]] = total.get(key[0], 0) + storage.nbytes()
        elif isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, (tuple, list)):
            for v in value:
                visit(v)

    for state in states:
        visit(state)
    return total


@contextmanager
def instrument(recorder, encoder=None, decoder=None, *, trace=False, profiler_step=None):
    """Scope-local hooks restored even on exceptions; production defaults untouched."""
    from unittest.mock import patch
    from oeis_learn.decoder import bigint_generation, sampler
    from oeis_learn.fresh import bigint_diagnostics
    from oeis_learn.synthesis import client
    from oeis_learn.tracking.foundation_metrics import PhaseTimer

    def wrapper(fn, label):
        @wraps(fn)
        def measured(*a, **kw):
            if trace:
                import torch

                with torch.profiler.record_function(label):
                    return fn(*a, **kw)
            metadata = {}
            if label == "decode":
                requests = a[1]
                metadata = dict(
                    batch=len(requests),
                    seeds=[x.seed for x in requests],
                    temperature=requests[0].temperature,
                )
            elif label == "fit_and_admit":
                metadata = dict(prompt_sha256=a[2].sha256)
            elif label == "native_ipc":
                metadata = dict(operation=a[1])
            with recorder.phase(label, **metadata):
                return fn(*a, **kw)

        return measured

    with ExitStack() as stack:
        if recorder.instrumented or trace:
            if trace and profiler_step is not None:
                original_sample = sampler.WatProgramSampler.sample_trajectories

                def sample_with_steps(*a, **kw):
                    original_hook = kw.get("budget_hook")

                    def hook(active):
                        profiler_step()
                        if original_hook is not None:
                            original_hook(active)

                    return original_sample(*a, **dict(kw, budget_hook=hook))

                stack.enter_context(
                    patch.object(
                        sampler.WatProgramSampler, "sample_trajectories", sample_with_steps
                    )
                )
            hooks = [
                (bigint_generation, "sample_batch", "decode"),
                (client.SynthesisClient, "fit", "fit_and_admit"),
                (client.SynthesisClient, "request", "native_ipc"),
                (client, "admit", "independent_admission"),
                (bigint_diagnostics, "admit", "raw_admission"),
                (bigint_diagnostics, "reference_evaluate", "reference"),
            ]
            if encoder is not None:
                hooks.append((encoder, "forward_from_sequences", "encoder_features_forward"))
            if trace:
                from oeis_learn.decoder.bigint_grammar import EnvironmentTracker
                from oeis_learn.decoder.grammar_masker import GrammarMasker

                hooks.append((sampler, "masked_distribution", "sampling_distribution"))
                hooks.extend(
                    [
                        (EnvironmentTracker, "get_valid_next_tokens", "grammar_support"),
                        (GrammarMasker, "fill_support", "grammar_mask_transfer"),
                    ]
                )
                if decoder is not None:
                    hooks.extend(
                        [
                            (decoder, "prefill", "decoder_prefill"),
                            (decoder, "decode_step", "decoder_cached_step"),
                        ]
                    )
            for obj, name, label in hooks:
                stack.enter_context(patch.object(obj, name, wrapper(getattr(obj, name), label)))
            if not trace:
                original = PhaseTimer.phase

                @contextmanager
                def phase(timer, name):
                    with recorder.phase("synchronized_" + name), original(timer, name):
                        yield

                stack.enter_context(patch.object(PhaseTimer, "phase", phase))
        yield


def compact_group(result):
    selections, attempts = {}, []
    for key, row in result["records"].items():
        selections[key] = {
            k: row[k] for k in ("selections", "arm_evidence", "complete", "family", "strata")
        }
        for mode, portfolio in row["details"].items():
            details = portfolio["details"]
            for item in details:
                attempts.append(
                    dict(
                        prompt=key,
                        mode=mode,
                        position=item["position"],
                        seed=item["seed"],
                        tokens=len(item["tokens"]),
                        tokens_sha256=digest(item["tokens"]),
                        source_sha256=item["source_sha256"],
                        generation_seconds=item["generation_seconds"],
                        native_seconds=item.get("native_seconds"),
                        batch_size=item["generation_batch_size"],
                        batch_id=item.get("generation_batch_id"),native_kind=item.get("native_kind"),
                        generation=item["generation"],
                        fitting=item["fitting"],
                        raw=item["raw"],
                        cache=item.get("fit_cache"),
                    )
                )
    return selections, attempts


def trace_probe(output, label, function, *, wait_steps=0, warmup_steps=0, active_steps=None):
    """Short bounded explanatory trace, never a main throughput measurement."""
    import torch

    folder = output / "traces"
    folder.mkdir(exist_ok=True)
    path = folder / (label + ".json")
    summary = dict(
        label=label,
        status="UNAVAILABLE",
        gpu_kernels=False,
        schedule=dict(wait_steps=wait_steps, warmup_steps=warmup_steps, active_steps=active_steps),
    )
    if active_steps is not None and (
        not 1 <= active_steps <= 8 or not 0 <= wait_steps <= 256 or not 0 <= warmup_steps <= 2
    ):
        raise ValueError("bounded trace schedule required")
    before = resource.getrlimit(resource.RLIMIT_FSIZE)
    before_signal = signal.getsignal(signal.SIGXFSZ)
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
            schedule=(
                torch.profiler.schedule(
                    wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1
                )
                if active_steps is not None
                else None
            ),
        ) as p:
            function(p.step if active_steps is not None else lambda: None)
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (min(40 << 20, before[0]) if before[0] >= 0 else 40 << 20, before[1]),
        )
        p.export_chrome_trace(str(path))
        events = read(path).get("traceEvents", [])
        kernels = [
            (e["ts"], e["ts"] + e["dur"])
            for e in events
            if "kernel" in str(e.get("cat", "")).lower() and "dur" in e and "ts" in e
        ]
        summary.update(
            status="COMPLETE" if kernels else "INCOMPLETE",
            gpu_kernels=bool(kernels),
            gpu_interval_union_seconds=interval_union(kernels) / 1e6,
            bytes=path.stat().st_size,
            sha256=file_hash(path),
            clock="profiler-correlated host/device timestamps",
            reason=None if kernels else "no actual HIP kernel events captured",
        )
    except Exception as exc:
        summary["reason"] = f"{type(exc).__name__}: {exc}"
        # Kineto can swallow EFBIG and leave only its .tmp file. Preserve bounded
        # failure evidence and classify it rather than reporting a missing JSON.
        for partial in (path, Path(str(path) + ".tmp")):
            if partial.is_file():
                size = partial.stat().st_size
                if size >= 40 << 20:
                    summary["reason"] = (
                        "trace export byte cap40MiB reached; reduce active recording window"
                    )
                retained = path.with_suffix(".partial.json")
                partial.replace(retained)
                summary["partial"] = dict(
                    path=str(retained.relative_to(output)), bytes=size, sha256=file_hash(retained)
                )
                break
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, before)
        signal.signal(signal.SIGXFSZ, before_signal)
        write(folder / (label + "-summary.json"), summary)
    return summary


def gpu_work(inputs, output, work, summary):
    import torch
    from oeis_learn.fresh.bigint_worker import models, _batch_loss, validation_loss, client
    from oeis_learn.fresh.bigint_diagnostics import generate
    from oeis_learn.decoder.bigint_generation import GenerationRequest, sample_batch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("actual HIP required; no CPU fallback")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("one actual HIP device required for unambiguous render-node attribution")
    recipe, manifest = work["recipe"], work["manifest"]
    plan = manifest["plan"]
    started = time.monotonic()
    render = manifest["render_node"]
    with Recorder(output / "startup", instrumented=True, render_node=render) as r:
        with r.phase("startup"):
            summary["device"] = dict(
                torch=torch.__version__,
                hip=torch.version.hip,
                name=torch.cuda.get_device_name(),
                properties=str(torch.cuda.get_device_properties(0)),
                sensors_mapped_to=render,
            )
            if file_hash(inputs / "weights.pt") != manifest["weights_sha256"]:
                raise ValueError("weights changed after manifest freeze")
            encoder, decoder, counts = models(recipe, "cuda")
            weights = torch.load(inputs / "weights.pt", map_location="cpu", weights_only=True)
            encoder.load_state_dict(weights["encoder"])
            decoder.load_state_dict(weights["decoder"])
            del weights
            optimizer = torch.optim.AdamW(
                [*encoder.parameters(), *decoder.parameters()],
                lr=recipe["learning"]["learning_rate"],
            )
            frozen = snapshot(encoder, decoder, optimizer)
            summary["parameters_sha256"] = parameter_digest(encoder, decoder)
            summary["counts"] = counts
            summary["optimizer"] = "fresh diagnostic AdamW; no parent optimizer resume"
        with r.phase("idle_background"):
            r.stop.wait(3)
        with r.phase("warmup"):
            until = time.monotonic() + 180
            encoder.train()
            decoder.train()
            for i in range(20):
                if time.monotonic() >= until:
                    raise TimeoutError("warmup update cap")
                _batch_loss(encoder, decoder, work["training"][:8], "cuda", optimizer=optimizer)
            warmed_optimizer = deepcopy(optimizer.state_dict())
            restore(frozen, encoder, decoder, optimizer)
            scratch = deepcopy(frozen)
            scratch["optimizer"] = warmed_optimizer
            summary["analysis_snapshot_storage_bytes"] = snapshot_bytes(frozen, scratch)
            summary["memory_interpretation"] = (
                "raw allocator/cgroup peaks include analysis snapshots and alternate idle cache; not production baseline memory"
            )
            summary["optimizer"] = (
                "diagnostic AdamW warmed20 steps, restored per block; original weights; no parent optimizer resume"
            )
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                memory = encoder.forward_from_sequences(
                    [[int(v) for v in row["values"]] for row in work["prompts"][:8]], device="cuda"
                )
            sample_batch(
                decoder,
                [
                    GenerationRequest(
                        memory[i : i + 1],
                        manifest["seed"] + i,
                        0,
                        min(until, time.monotonic() + 60),
                    )
                    for i in range(8)
                ],
            )
            restore(frozen, encoder, decoder, optimizer)
    summary["updates"] = {arm: [] for arm in plan["arms"]}
    update_deadline = time.monotonic() + 300
    for block in range(20):
        for arm in group_order(block):
            if time.monotonic() >= update_deadline:
                summary["partial_reasons"].append("scratch update/validation stage cap")
                break
            restore(scratch, encoder, decoder, optimizer)
            encoder.train()
            decoder.train()
            torch.cuda.reset_peak_memory_stats()
            with Recorder(
                output / "arms" / arm / f"update-{block:02}",
                instrumented=arm == "telemetry",
                render_node=render,
            ) as r:
                with instrument(r, encoder, decoder), r.phase("updates", block=block):
                    phase_ns, losses, done = {}, [], 0
                    for step in range(25):
                        if time.monotonic() >= update_deadline:
                            break
                        offset = ((block * 25 + step) * 8) % 128
                        record = _batch_loss(
                            encoder,
                            decoder,
                            work["training"][offset : offset + 8],
                            "cuda",
                            optimizer=optimizer,
                        )
                        done += 1
                        losses.append(record["loss"])
                        for k, v in record["phase_timings_ns"].items():
                            phase_ns[k] = phase_ns.get(k, 0) + v
                summary["updates"][arm].append(
                    dict(
                        block=block,
                        completed=done,
                        losses_sha256=digest(losses),
                        parameters_sha256=parameter_digest(encoder, decoder),
                        synchronized_phase_ns=phase_ns,
                        allocated=torch.cuda.memory_allocated(),
                        reserved=torch.cuda.memory_reserved(),
                        peak_allocated=torch.cuda.max_memory_allocated(),
                    )
                )
            write(output / "progress.json", summary)
    summary["validation"] = {}
    for arm in plan["arms"]:
        restore(frozen, encoder, decoder, optimizer)
        with Recorder(
            output / "arms" / arm / "validation",
            instrumented=arm == "telemetry",
            render_node=render,
        ) as r:
            with instrument(r, encoder, decoder), r.phase("validation"):
                summary["validation"][arm] = validation_loss(
                    encoder, decoder, work["validation"], "cuda", deadline=update_deadline
                )
    selections = {arm: {} for arm in plan["arms"]}
    attempts = {arm: [] for arm in plan["arms"]}
    settings = dict(recipe=recipe, contract_sha256=manifest["identity"])
    with ExitStack() as stack:
        clients = {
            arm: stack.enter_context(client(settings, output / "arms" / arm))
            for arm in plan["arms"]
        }
        # Idle alternate worker/cache memory is disclosed, not counted as parallel work.
        summary["cache_policy"] = (
            "independent per-arm clients/directories; both histories retained; one active request"
        )
        for group in range(4):
            for arm in group_order(group):
                restore(frozen, encoder, decoder, optimizer)
                rows = work["prompts"][group * 8 : (group + 1) * 8]
                torch.cuda.reset_peak_memory_stats()
                with Recorder(
                    output / "arms" / arm / f"group-{group}",
                    instrumented=arm == "telemetry",
                    render_node=render,
                ) as r:
                    with instrument(r, encoder, decoder), r.phase("portfolio", group=group):
                        result = generate(
                            encoder,
                            decoder,
                            rows,
                            recipe,
                            clients[arm],
                            deadline=time.monotonic() + plan["generation_group_seconds"],
                            diagnostic_attempts=plan["attempts"],
                        )
                    sealed, items = compact_group(result)
                    for item in items:
                        item["group"] = group
                    selections[arm].update(sealed)
                    attempts[arm].extend(items)
                    write(
                        output / "arms" / arm / f"group-{group}/work.json",
                        dict(
                            attempts=items,
                            controls=result["controls"],
                            worker=dict(clients[arm].worker.stats),
                            cache=clients[arm].caches.metrics(),
                            memory=dict(
                                allocated=torch.cuda.memory_allocated(),
                                reserved=torch.cuda.memory_reserved(),
                                peak_allocated=torch.cuda.max_memory_allocated(),
                            ),
                        ),
                    )
                write(output / "selections.json", selections)
                write(output / "attempts.json", attempts)
                write(output / "progress.json", summary)
    write(output / "attempts.json", attempts)
    expected = 32 * sum(plan["attempts"].values())
    summary["work"] = {
        arm: dict(
            requested=expected,
            started=len(attempts[arm]),
            unstarted=expected - len(attempts[arm]),
            generation_completed=sum(x["generation"]["complete"] for x in attempts[arm]),
            emitted_tokens=sum(x["tokens"] for x in attempts[arm]),
            updates=sum(x["completed"] for x in summary["updates"][arm]),
            validation_rows=summary["validation"][arm]["attempted"],
        )
        for arm in plan["arms"]
    }
    # Explanation-only windows after all headline operations; no main cache changes.
    summary["traces"] = []
    for label in ("updates", "short-decode", "long-decode"):
        restore(scratch if label == "updates" else frozen, encoder, decoder, optimizer)
        trace_deadline = time.monotonic() + 30
        with Recorder(output / "trace-events" / label, instrumented=False, render_node=render) as r:

            def probe(step, label=label, trace_deadline=trace_deadline):
                with instrument(
                    r,
                    encoder,
                    decoder,
                    trace=True,
                    profiler_step=None if label == "updates" else step,
                ):
                    if label == "updates":
                        encoder.train()
                        decoder.train()
                        for _ in range(5):
                            if time.monotonic() >= trace_deadline:
                                break
                            _batch_loss(
                                encoder, decoder, work["training"][:8], "cuda", optimizer=optimizer
                            )
                            step()
                    else:
                        encoder.eval()
                        decoder.eval()
                        complete = [x for x in attempts["ordinary"] if x["mode"] == "correct"]
                        if not complete:
                            raise RuntimeError("no observed decode available for trace")
                        item = sorted(complete, key=lambda x: x["tokens"])[
                            0 if label == "short-decode" else -1
                        ]
                        batch = [
                            x
                            for x in complete
                            if x["group"] == item["group"] and x["position"] == item["position"]
                        ]
                        byid = {x["prompt"]: x for x in batch}
                        batch = [byid[x["id"]] for x in work["prompts"] if x["id"] in byid]
                        rows = [
                            next(x for x in work["prompts"] if x["id"] == b["prompt"])
                            for b in batch
                        ]
                        with torch.no_grad():
                            memory = encoder.forward_from_sequences(
                                [[int(v) for v in row["values"]] for row in rows], device="cuda"
                            )
                        summary.setdefault("trace_inputs", {})[label] = dict(
                            attempts=batch,
                            trace_cap=256,
                            stress=False,
                            long_output_coverage=max(x["tokens"] for x in batch) >= 256,
                            reason="observed shortest/longest batch; no claim of unobserved long-output stress coverage",
                        )
                        sample_batch(
                            decoder,
                            [
                                GenerationRequest(
                                    memory[i : i + 1],
                                    b["seed"],
                                    0 if b["position"] == 0 else 0.8,
                                    trace_deadline,
                                    max_length=256,
                                )
                                for i, b in enumerate(batch)
                            ],
                        )

            # Keep the five-update/256-token replay, but record only one steady
            # update or eight token steps. Operator-dense full windows exceed40MiB.
            summary["traces"].append(
                trace_probe(
                    output,
                    label,
                    probe,
                    wait_steps=1 if label == "updates" else (64 if label == "long-decode" else 0),
                    warmup_steps=1 if label != "short-decode" else 0,
                    active_steps=1 if label == "updates" else 8,
                )
            )
    summary["trace_coverage_complete"] = all(x["status"] == "COMPLETE" for x in summary["traces"])
    summary["seconds"] = time.monotonic() - started
    summary["checkpoint_export"] = dict(
        seconds=None, reason="not measured; projection cannot assume zero"
    )
    summary["status"] = (
        "COMPLETE"
        if all(
            v["updates"] == 500 and v["validation_rows"] == 256 and v["unstarted"] == 0
            for v in summary["work"].values()
        )
        else "PARTIAL"
    )


def scorer_work(inputs, output, work, summary):
    from .bigint_diagnostics import score
    from .bigint_worker import client

    manifest = work["manifest"]
    selections, truth = read(inputs / "selections.json"), read(inputs / "truth.json")
    if (
        digest(truth) != manifest["truth_sha256"]
        or file_hash(inputs / "selections.json") != work["selections_sha256"]
    ):
        raise ValueError("scoring truth/selection integrity mismatch")
    summary["groups"] = {arm: [] for arm in manifest["plan"]["arms"]}
    with ExitStack() as stack:
        clients = {
            arm: stack.enter_context(
                client(
                    dict(recipe=work["recipe"], contract_sha256=manifest["identity"]),
                    output / "arms" / arm,
                )
            )
            for arm in manifest["plan"]["arms"]
        }
        for group in range(4):
            keys = manifest["prompt_ids"][group * 8 : (group + 1) * 8]
            for arm in group_order(group):
                folder = output / "arms" / arm / f"group-{group}"
                with Recorder(folder, instrumented=arm == "telemetry") as r:
                    with instrument(r), r.phase("scoring", group=group):
                        result = score(
                            {k: selections[arm][k] for k in keys},
                            {k: truth[k] for k in keys},
                            clients[arm],
                            deadline=time.monotonic() + manifest["plan"]["scoring_group_seconds"],
                        )
                write(folder / "outcomes.json", result.pop("outcomes"))
                write(
                    folder / "worker.json",
                    dict(worker=clients[arm].worker.stats, cache=clients[arm].caches.metrics()),
                )
                summary["groups"][arm].append(result)
                write(output / "progress.json", summary)
    summary["status"] = "COMPLETE"


def preflight_work(inputs, output, work, summary):
    """Actual HIP plus a tiny bounded trace; no model, fitting or hidden data."""
    import torch

    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight requires one actual HIP device; no CPU fallback")
    started = time.monotonic()

    def operation(step):
        matrix = torch.ones(32, 32, device="cuda")
        result = matrix @ matrix
        torch.cuda.synchronize()
        if not bool(torch.isfinite(result).all()) or float(result[0, 0]) != 32:
            raise RuntimeError("HIP preflight matrix result invalid")

    summary["trace"] = trace_probe(output, "capability", operation)
    if summary["trace"]["status"] != "COMPLETE":
        raise RuntimeError(
            "HIP trace preflight failed; model workload unstarted: "
            + str(summary["trace"].get("reason"))
        )
    from .bigint_worker import models, _batch_loss

    if file_hash(inputs / "weights.pt") != work["manifest"]["weights_sha256"]:
        raise ValueError("preflight weights differ from frozen checkpoint")
    encoder, decoder, _ = models(work["recipe"], "cuda")
    weights = torch.load(inputs / "weights.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(weights["encoder"])
    decoder.load_state_dict(weights["decoder"])
    del weights
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()],
        lr=work["recipe"]["learning"]["learning_rate"],
    )

    def updates(step):
        encoder.train()
        decoder.train()
        for _ in range(5):
            if time.monotonic() - started >= 120:
                raise TimeoutError("representative trace preflight120s cap")
            _batch_loss(encoder, decoder, work["training"][:8], "cuda", optimizer=optimizer)
            step()

    summary["model_trace"] = trace_probe(
        output, "representative-update", updates, wait_steps=1, warmup_steps=1, active_steps=1
    )
    summary["disposable_updates"] = 5
    if summary["model_trace"]["status"] != "COMPLETE":
        raise RuntimeError(
            "representative update trace preflight failed; paired benchmark unstarted: "
            + str(summary["model_trace"].get("reason"))
        )
    summary["device"] = dict(
        name=torch.cuda.get_device_name(), torch=torch.__version__, hip=torch.version.hip
    )
    summary["seconds"] = time.monotonic() - started
    if summary["seconds"] > 120:
        raise TimeoutError("HIP preflight exceeded120s device allowance")
    summary["status"] = "COMPLETE"


def inside(inputs, output, mode):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from .bigint_config import packaged_artifacts

    output.mkdir(parents=True, exist_ok=True)
    summary = dict(status="FAIL", qualified=False, mode=mode, partial_reasons=[])
    try:
        work = read(inputs / "work.json")
        manifest = work["manifest"]
        if manifest["identity"] != digest({k: v for k, v in manifest.items() if k != "identity"}):
            raise ValueError("diagnostic manifest identity mismatch")
        if digest(work["recipe"]) != manifest["recipe_sha256"]:
            raise ValueError("recipe differs from frozen manifest")
        from scripts.profile_bigint_pipeline import workload

        if manifest["plan"] != workload():
            raise ValueError("unknown or modified profiling workload")
        if (
            mode in {"gpu", "preflight"}
            and digest({k: work[k] for k in ("prompts", "training", "validation")})
            != manifest["payload_sha256"]
        ):
            raise ValueError("visible workload differs from frozen manifest")
        source = digest(collect_snapshot(Path(__file__).resolve().parents[3]))
        if source != manifest["source_sha256"]:
            raise ValueError("image/source differs from frozen checkout; rebuild image")
        summary["source_sha256"] = source
        summary["artifacts"] = packaged_artifacts()
        from oeis_learn.tracking.pipeline_profile import device_view, audit_device_view

        view = device_view()
        write(output / "devices.json", view)
        expected = os.environ.get("OEIS_EXPECTED_PIPELINE_DEVICES")
        if expected is None:
            raise ValueError("host-provided expected device identities required")
        audit_device_view([], view, json.loads(expected))
        {"gpu": gpu_work, "scorer": scorer_work, "preflight": preflight_work}[mode](
            inputs, output, work, summary
        )
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        write(output / "summary.json", summary)
    return 0 if summary["status"] in {"COMPLETE", "PARTIAL"} else 1
