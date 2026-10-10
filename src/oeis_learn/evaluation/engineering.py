"""Executable engineering comparisons; correctness precedes practical-effect decisions."""

from copy import deepcopy
from statistics import median
import time

from oeis_learn.tracking.foundation_metrics import PhaseTimer, resource_metrics, sensor_metrics


def equivalent_logits(control, actual, support):
    import torch

    torch.testing.assert_close(actual, control, atol=1e-5, rtol=1e-4)
    if not torch.equal(support[0], support[1]):
        raise ValueError("legal support disagreement")
    # Compare argmax only when the control margin exceeds twice local tolerance.
    top = control.topk(2, dim=-1).values
    tolerance = 1e-5 + 1e-4 * top.abs().amax(-1)
    stable = top[..., 0] - top[..., 1] > 2 * tolerance
    if (stable & (control.argmax(-1) != actual.argmax(-1))).any():
        raise ValueError("stable-margin argmax disagreement")


def inference_stream(encoder, decoder, rows, *, cached, backend, batch=1):
    """Identical frozen autoregressive token streams; no teacher-forced full-pass timing."""
    import torch
    from oeis_learn.decoder.program_codec import BOS_ID, PAD_ID
    from oeis_learn.decoder.environment_tracker import FoundationEnvironmentTracker

    device = next(encoder.parameters()).device
    timer = PhaseTimer(device)
    outputs, supports = [], []
    previous = decoder.attention_backend
    decoder.attention_backend = backend
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch):
                active = rows[start : start + batch]
                streams = [[BOS_ID] + r["tokens"] for r in active]
                trackers = [FoundationEnvironmentTracker() for _ in active]
                with timer.phase("encoder"):
                    memory = encoder.forward_from_sequences(
                        [[int(v) for v in r["entry"]["conditioning_values"]] for r in active],
                        device=device,
                    )
                cache, values, masks = None, [[] for _ in active], [[] for _ in active]
                for position in range(max(map(len, streams)) - 1):
                    with timer.phase("decoder"):
                        if cached:
                            tokens = torch.tensor(
                                [
                                    row[position] if position < len(row) else PAD_ID
                                    for row in streams
                                ],
                                device=device,
                            )
                            if cache is None:
                                logits, cache = decoder.prefill(
                                    tokens[:, None], memory, capacity=1024, backend=backend
                                )
                                logits = logits[:, -1]
                            else:
                                logits = decoder.decode_step(
                                    tokens,
                                    cache,
                                    active=[position < len(row) - 1 for row in streams],
                                )[:, -1]
                        else:
                            tokens = torch.full(
                                (len(active), position + 1), PAD_ID, device=device, dtype=torch.long
                            )
                            for i, row in enumerate(streams):
                                prefix = row[: position + 1]
                                tokens[i, : len(prefix)] = torch.tensor(prefix, device=device)
                            logits = decoder(
                                tokens, memory, tgt_key_padding_mask=tokens.eq(PAD_ID)
                            )[:, -1]
                    for i, row in enumerate(streams):
                        if position >= len(row) - 1:
                            continue
                        values[i].append(logits[i].detach().cpu())
                        masks[i].append(tuple(trackers[i].get_valid_next_tokens()))
                        from oeis_learn.decoder.program_codec import ID_TO_TOKEN

                        trackers[i].update(ID_TO_TOKEN[row[position + 1]])
                outputs.extend(torch.stack(row) for row in values)
                supports.extend(masks)
    finally:
        decoder.attention_backend = previous
    return outputs, supports, timer.values


def paired_benchmark(
    control, challenger, *, warmups=10, repeats=30, end_to_end=None, clock=time.monotonic_ns
):
    """Alternating paired invocations with correctness checks on every invocation."""
    import torch

    times, phases = {"baseline": [], "challenger": []}, []
    for repeat in range(warmups + repeats):
        values = {}
        for name in ("baseline", "challenger") if repeat % 2 == 0 else ("challenger", "baseline"):
            start = clock()
            result = (control if name == "baseline" else challenger)()
            inference_ns = clock() - start
            if end_to_end is not None:
                end_to_end()
            elapsed = clock() - start
            values[name] = result
            if repeat >= warmups:
                times[name].append(dict(inference_ns=inference_ns, end_to_end_ns=elapsed))
        left, right = values["baseline"], values["challenger"]
        if len(left[0]) != len(right[0]):
            raise ValueError("fixed prompt denominator changed")
        for a, b, sa, sb in zip(left[0], right[0], left[1], right[1]):
            if sa != sb:
                raise ValueError("legal support disagreement")
            equivalent_logits(a, b, (torch.tensor([1]), torch.tensor([1])))
        if repeat >= warmups:
            phases.append(dict(baseline=left[2], challenger=right[2]))
    medians = {k: median(x["end_to_end_ns"] for x in v) for k, v in times.items()}
    gain = 1 - medians["challenger"] / medians["baseline"] if medians["baseline"] else None
    return dict(
        warmups=warmups,
        repeats=repeats,
        alternating=True,
        timings=times,
        phases=phases,
        semantic_equivalence=True,
        median_end_to_end_ns=medians,
        gain=gain,
        disposition="adopted" if gain is not None and gain >= 0.05 else "baseline_retained",
    )


def precision_fixture(encoder, decoder, rows, dtype, *, updates=32):
    """Matched initialization/order; actual operation probe and finite scaled AMP updates."""
    import torch
    from oeis_learn.decoder.program_codec import BOS_ID, PAD_ID
    from oeis_learn.rl.sft_trainer import teacher_forced_logits

    device = next(encoder.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("precision acceptance requires actual device")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return dict(status="unavailable", reason="BF16 not advertised")
    probe = torch.ones(8, 8, device=device)
    with torch.autocast("cuda", dtype=dtype):
        actual = probe @ probe
    if not torch.isfinite(actual).all():
        raise FloatingPointError("precision operation probe failed")
    initial = (deepcopy(encoder.state_dict()), deepcopy(decoder.state_dict()))
    losses = {}
    try:
        for label in ("fp32", "amp"):
            encoder.load_state_dict(initial[0])
            decoder.load_state_dict(initial[1])
            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) + list(decoder.parameters()),
                lr=0.0003,
                weight_decay=0.01,
            )
            scaler = torch.amp.GradScaler("cuda", enabled=label == "amp" and dtype == torch.float16)
            losses[label] = []
            for update in range(updates):
                row = rows[update % len(rows)]
                tokens = torch.tensor([[BOS_ID] + row["tokens"]], device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype, enabled=label == "amp"):
                    logits = teacher_forced_logits(
                        encoder,
                        decoder,
                        [[int(v) for v in row["entry"]["conditioning_values"]]],
                        tokens,
                        device=device,
                        pad_id=PAD_ID,
                    )
                    loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tokens[:, 1:])
                if not torch.isfinite(loss):
                    raise FloatingPointError("precision nonfinite loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in list(encoder.parameters()) + list(decoder.parameters())
                ):
                    raise FloatingPointError("precision nonfinite gradients")
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    1,
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                if any(
                    not torch.isfinite(p).all()
                    for p in list(encoder.parameters()) + list(decoder.parameters())
                ):
                    raise FloatingPointError("precision nonfinite parameters")
                torch.cuda.synchronize(device)
                losses[label].append(float(loss.detach()))
        difference = abs(losses["amp"][-1] - losses["fp32"][-1]) / max(
            abs(losses["fp32"][-1]), 1e-12
        )
        return dict(
            status="PASS" if difference <= 0.02 else "FAIL",
            losses=losses,
            relative_final_loss=difference,
            dtype=str(dtype),
            updates=updates,
            runtime_validity="checked independently by frozen source execution",
        )
    finally:
        encoder.load_state_dict(initial[0])
        decoder.load_state_dict(initial[1])


def steady_state(control, challenger, *, seconds=600, device="cuda"):
    """Ten minutes per arm, actual wall/device/resource samples; no extrapolated gain."""
    results = {}
    for name, invoke in (("fp32", control), ("amp", challenger)):
        start, count, samples = time.monotonic(), 0, []
        while time.monotonic() - start < seconds:
            invoke()
            count += 1
            if count == 1 or time.monotonic() - samples[-1]["second"] - start >= 1:
                samples.append(
                    dict(
                        second=time.monotonic() - start,
                        resources=resource_metrics(device),
                        sensors=sensor_metrics(device),
                    )
                )
                check_device_resources(device)
        results[name] = dict(
            completed=count, elapsed_seconds=time.monotonic() - start, samples=samples
        )
    gain = (
        results["amp"]["completed"]
        / results["amp"]["elapsed_seconds"]
        / (results["fp32"]["completed"] / results["fp32"]["elapsed_seconds"])
        - 1
    )
    return dict(
        segments=results, gain=gain, disposition="adopted" if gain >= 0.10 else "baseline_retained"
    )


def check_device_resources(device):
    import torch

    if not str(device).startswith("cuda"):
        return
    capacity = min(64 << 30, int(0.8 * torch.cuda.get_device_properties(device).total_memory))
    if torch.cuda.memory_reserved(device) > capacity:
        raise MemoryError("declared device allocator budget exceeded")


def run_engineering(inputs, output):
    import torch
    from oeis_learn.experiments.artifacts import load_json, compute_canonical_digest as digest
    from oeis_learn.tracking.role_worker import load_models
    from oeis_learn.tracking.role_coordinator import QueueRunner
    from oeis_learn.tracking.foundation_metrics import EventLog, fuel_summary

    settings = load_json((inputs / "settings.json").read_bytes())
    examples = load_json((inputs / "examples.json").read_bytes())
    rows = examples["rows"]
    if len(rows) != 128:
        raise ValueError("engineering requires exactly128 frozen prompts")
    encoder, decoder = load_models(inputs, settings)
    encoder.eval()
    decoder.eval()
    runner, log = QueueRunner(output / "runtime-queue"), EventLog(output / "measurements")
    counter = [0]
    fuel = []

    def execution():
        counter[0] += 1
        for index, row in enumerate(rows):
            value = runner.run_single(
                row["source"], range(20), request_id=f"engineering-{counter[0]}-{index}"
            )
            fuel.extend(value.fuel_terms)
            if (
                value.outcome is not None
                or value.outputs != row["entry"]["outputs"][:20]
                or value.reference_outputs != value.outputs
            ):
                raise ValueError("engineering exact runtime disagreement")

    def invoke(cached, backend, batch):
        return lambda: inference_stream(
            encoder, decoder, rows, cached=cached, backend=backend, batch=batch
        )

    results = {}
    inference_results = {}
    for seed in (7001, 7002, 7003):
        torch.manual_seed(seed)
        from oeis_learn.experiments.pilot_models import construct_pilot_models

        encoder, decoder = construct_pilot_models(settings["profile"], settings["device"])
        encoder.eval()
        decoder.eval()
        inference_results[str(seed)] = {}
        for name, control, challenger in (
            ("cache", invoke(False, "eager", 1), invoke(True, "eager", 1)),
            ("batch", invoke(True, "eager", 1), invoke(True, "eager", 4)),
            ("attention", invoke(True, "eager", 4), invoke(True, "sdpa", 4)),
        ):
            result = paired_benchmark(control, challenger, end_to_end=execution)
            inference_results[str(seed)][name] = result
            log.append(f"{name}-{seed}", result)
    results["inference"] = inference_results
    torch.manual_seed(settings["seed"])
    encoder, decoder = construct_pilot_models(settings["profile"], settings["device"])
    encoder.eval()
    decoder.eval()
    results["grammar"] = grammar_benchmark(rows, settings["device"])
    log.append("grammar", results["grammar"])
    results["workers"] = worker_benchmark(rows, runner)
    log.append("workers", results["workers"])
    results["fuel_streaming"] = fuel_stream_benchmark(rows, runner)
    log.append("fuel_streaming", results["fuel_streaming"])
    results["relations"] = relation_benchmark(rows, encoder, decoder, runner)
    log.append("relations", results["relations"])
    results["precision"] = {}
    from contextlib import nullcontext

    for dtype in (torch.float16, torch.bfloat16):
        label = str(dtype)
        test = precision_fixture(encoder, decoder, rows, dtype)
        if test["status"] == "FAIL":
            raise FloatingPointError("precision final loss disagreement")
        if test["status"] == "PASS":

            def call(amp=False):
                scope = torch.autocast("cuda", dtype=dtype) if amp else nullcontext()
                with scope:
                    inference_stream(encoder, decoder, rows, cached=True, backend="sdpa", batch=4)
                execution()

            test["steady_state"] = steady_state(
                lambda: call(False), lambda: call(True), device=settings["device"]
            )
        results["precision"][label] = test
        log.append("precision-" + label, test)
    return dict(
        version="engineering-session/v1",
        status="complete",
        prompts=128,
        seeds=[7001, 7002, 7003],
        results=results,
        fuel=fuel_summary(fuel, limit=1000000),
        sensors=sensor_metrics(settings["device"]),
        resources=resource_metrics(settings["device"]),
        fixture_sha256=digest(examples),
        final_truth_available=False,
    )


def grammar_benchmark(rows, device):
    import torch
    from oeis_learn.decoder.grammar_masker import GrammarMasker
    from oeis_learn.decoder.environment_tracker import FoundationEnvironmentTracker
    from oeis_learn.decoder.program_codec import FOUNDATION_VOCAB_SIZE, ID_TO_TOKEN

    supports = []
    for row in rows:
        tracker = FoundationEnvironmentTracker()
        for token in row["tokens"]:
            supports.append(tuple(tracker.get_valid_next_tokens()))
            tracker.update(ID_TO_TOKEN[token])
    masker = GrammarMasker(FOUNDATION_VOCAB_SIZE)

    def control():
        result = []
        for legal in supports:
            tensor = torch.zeros((1, FOUNDATION_VOCAB_SIZE), device=device, dtype=torch.bool)
            tensor[0, list(legal)] = True
            result.append(tensor.float())
        return result, supports, {}

    def challenger():
        result = [
            masker.fill_support([list(legal)], device=device).clone().float() for legal in supports
        ]
        return result, supports, {}

    result = paired_benchmark(control, challenger)
    result.update(
        disposition="baseline_retained",
        scope="grammar microbenchmark; end-to-end adoption not established",
    )
    return result


def worker_benchmark(rows, runner):
    from concurrent.futures import ThreadPoolExecutor
    from oeis_learn.tracking.foundation_metrics import fuel_summary

    times, fuel, counter = {4: [], 8: []}, [], 0
    for repeat in range(40):
        for workers in (4, 8) if repeat % 2 == 0 else (8, 4):
            start = time.monotonic_ns()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        runner.run_single,
                        r["source"],
                        range(20),
                        request_id=f"workers-{counter}-{i}",
                    )
                    for i, r in enumerate(rows)
                ]
                for row, future in zip(rows, futures):
                    result = future.result()
                    fuel.extend(result.fuel_terms)
                    if (
                        result.outcome is not None
                        or result.outputs != row["entry"]["outputs"][:20]
                        or result.outputs != result.reference_outputs
                    ):
                        raise ValueError("worker count changes exact outputs")
            counter += 1
            if repeat >= 10:
                times[workers].append(time.monotonic_ns() - start)
    return dict(
        warmups=10,
        repeats=30,
        workers=[4, 8],
        timings_ns=times,
        fuel=fuel_summary(fuel, limit=1000000),
        semantic_equivalence=True,
    )


def fuel_stream_benchmark(rows, runner):
    from oeis_learn.sandbox.streaming import execute_logical, next_block

    results = []
    for row in rows:
        horizons = {}
        for horizon in (20, 50, 100):
            result = runner.run_escalating(
                row["source"],
                range(horizon),
                request_id=f"engineering-horizon-{row['key']}-{horizon}",
            )
            if (
                result.outcome is not None
                or result.outputs != row["entry"]["outputs"][:horizon]
                or result.reference_outputs != result.outputs
            ):
                raise ValueError("fuel horizon exact execution mismatch")
            horizons[str(horizon)] = result.to_dict()
        results.append(
            dict(
                source_sha256=row["entry"]["source_sha256"],
                horizons=horizons,
                unavailable_horizons={"120": "baseline body ABI is0..99"},
            )
        )
    # Real pure/stream implementations with the same generic bounded expression;
    # full120 is supported only in the explicit logical ABI.
    pure = "(pure (return (index)))"
    stream = "(stream (init) (next (return (index))))"
    for horizon in (20, 50, 100, 120):
        a = execute_logical(pure, range(horizon))
        b = next_block(stream, horizon)
        if a["outcome"] is not None or b["outcome"] is not None or a["outputs"] != b["outputs"]:
            raise ValueError("pure/stream parity failed")
        results.append(dict(horizon=horizon, pure=a, stream=b, purpose="ABI parity fixture"))
    return dict(records=results, selected=False)


def relation_benchmark(rows, encoder, decoder, runner):
    results = []
    for size in (128, 256, 512):
        started = time.monotonic()
        value = _relation_corpus(rows, encoder, decoder, runner, size, started + 60)
        value["size"], value["elapsed_seconds"] = size, time.monotonic() - started
        results.append(value)
    return dict(corpora=results, seconds_per_corpus=60, proof_status="not_claimed")


def _relation_corpus(rows, encoder, decoder, runner, size, deadline):
    """Generic source-derived corpora and exact finite verification; no truth feedback."""
    import numpy as np
    import torch
    import sympy as sp
    from oeis_learn.discovery.vector_search import VectorRelationSearcher
    from oeis_learn.discovery.program_features import legacy_features
    from oeis_learn.discovery.numerical_validator import validate_numerical_relation
    from oeis_learn.data.models import SequenceRef
    from oeis_learn.decoder.program_codec import BOS_ID, encode_body
    from oeis_learn.experiments.profiles import I64_LOCAL_NAMES, I32_LOCAL_NAMES
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    device = next(encoder.parameters()).device
    # Seeded offsets are compositional variants of admitted generic sources, never
    # inserted into the training archive. Each variant is independently executed.
    reset = " ".join(
        f"{'i64' if name.startswith('$v') else 'i32'}.const 0 local.set {name}"
        for name in (*I64_LOCAL_NAMES, *I32_LOCAL_NAMES)
        if name != "$n"
    )
    corpus = []
    for number in range(size):
        if time.monotonic() >= deadline:
            return dict(
                status="inconclusive",
                completed_sources=len(corpus),
                shortfall=size - len(corpus),
                reason="60-second corpus preparation cap",
            )
        base = rows[number % len(rows)]
        source = (
            base["source"]
            if number < len(rows)
            else base["source"] + f" i256.const {number} i256.add"
        )
        value = runner.run_single(source, range(100), request_id=f"relations-{number}")
        if value.outcome is not None or value.outputs != value.reference_outputs:
            raise ValueError("generic relation corpus correctness gate")
        corpus.append(
            dict(source=source, values=[int(v) for v in value.outputs], parent=base["key"])
        )
    # Plant a genuinely executed composition, only when its two operands have rank2.
    planted = None
    for b in range(1, min(128, len(corpus))):
        if (
            b == 2
            or sp.Matrix(list(zip(corpus[0]["values"][:10], corpus[b]["values"][:10]))).rank() != 2
        ):
            continue
        source = corpus[0]["source"] + " " + reset + " " + corpus[b]["source"] + " i256.add"
        value = runner.run_single(source, range(100), request_id="relations-planted")
        if value.outcome is None and value.outputs == value.reference_outputs:
            corpus[2] = dict(
                source=source,
                values=[int(v) for v in value.outputs],
                parent=[corpus[0]["parent"], corpus[b]["parent"]],
            )
            planted = {"p0000", f"p{b:04d}", "p0002"}
            break
    features = {"program": {}, "numerical": {}, "synthesis_latent": {}}
    hidden = []
    hook = decoder.lm_head.register_forward_pre_hook(
        lambda module, args: hidden.append(args[0].detach())
    )
    try:
        with torch.no_grad():
            for number, row in enumerate(corpus):
                if time.monotonic() >= deadline:
                    return dict(
                        status="inconclusive",
                        completed_sources=len(corpus),
                        completed_features=number,
                        reason="60-second feature preparation cap",
                    )
                key = f"p{number:04d}"
                info = legacy_features(row["source"])
                bag = np.zeros(128)
                for name, count in info["subtrees"].items():
                    bag[int(digest(name)[7:15], 16) % 128] += count
                features["program"][key] = bag
                values = np.asarray(row["values"][:10], dtype=np.float64)
                features["numerical"][key] = values / max(np.max(np.abs(values)), 1)
                tokens = encode_body(row["source"])
                if len(tokens) + 1 > 1024:
                    # Source feature is valid, but unsupported context is explicit.
                    features["synthesis_latent"][key] = None
                    continue
                memory = encoder.forward_from_sequences([row["values"][:20]], device=device)
                hidden.clear()
                decoder(torch.tensor([[BOS_ID] + tokens], device=device), memory)
                features["synthesis_latent"][key] = hidden[-1].mean(1)[0].cpu().numpy()
    finally:
        hook.remove()
    results = []
    for size in (size,):
        for control in ("planted", "negative"):
            for method, mapping in features.items():
                selected = {k: v for k, v in list(mapping.items())[:size] if v is not None}
                if len(selected) != size:
                    results.append(
                        dict(
                            size=size,
                            control=control,
                            method=method,
                            status="inconclusive",
                            shortfall=size - len(selected),
                        )
                    )
                    continue
                if time.monotonic() >= deadline:
                    results.append(
                        dict(
                            size=size,
                            control=control,
                            method=method,
                            status="inconclusive",
                            reason="60-second corpus cap",
                        )
                    )
                    continue
                actual = {f"p{i:04d}": list(row["values"]) for i, row in enumerate(corpus[:size])}
                if control == "negative":
                    rng = np.random.default_rng(7001 + size)
                    actual = {
                        k: list(np.asarray(v, dtype=object)[rng.permutation(len(v))])
                        for k, v in actual.items()
                    }
                search = VectorRelationSearcher(
                    neighbors=16,
                    max_pairs=4096,
                    seconds=min(10, max(0.001, deadline - time.monotonic())),
                )
                proposals = search.search_additive_triples(selected, max_candidates=16)
                validated, recalled, records = 0, False, []
                for candidate in proposals:
                    if time.monotonic() >= deadline:
                        break
                    ids = candidate.sequences
                    matrix = sp.Matrix([[actual[k][i] for k in ids] for i in range(10)])
                    basis = matrix.nullspace()
                    if not basis:
                        records.append(
                            dict(
                                ids=ids,
                                outcome="no_exact_nullspace",
                                rank=int(matrix.rank()),
                                nullity=0,
                            )
                        )
                        continue
                    vector = basis[0]
                    denominator = sp.ilcm(*[v.q for v in vector])
                    coefficients = [int(v * denominator) for v in vector]
                    proof = validate_numerical_relation(
                        [SequenceRef(k) for k in ids],
                        coefficients,
                        actual,
                        unseen_indices=range(20, 100),
                    ).to_dict()
                    records.append(proof)
                    validated += int(proof["outcome"] == "VERIFIED")
                    recalled |= (
                        control == "planted"
                        and planted is not None
                        and set(ids) == planted
                        and proof["outcome"] == "VERIFIED"
                    )
                results.append(
                    dict(
                        size=size,
                        control=control,
                        method=method,
                        top_k=16,
                        arity_cap=8,
                        attempted=search.last_metrics,
                        exact_validations=records,
                        precision=validated / len(proposals) if proposals else None,
                        planted_recall=int(recalled)
                        if control == "planted" and planted is not None
                        else None,
                        planted_available=planted is not None,
                        resource=resource_metrics(device),
                        status="inconclusive" if time.monotonic() >= deadline else "complete",
                    )
                )
    return dict(
        corpus_sha256=digest(corpus),
        sources="admitted generic compositions; measurement only",
        results=results,
        proof_status="not_claimed",
    )
