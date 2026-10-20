#!/usr/bin/env python3
"""Image-side measurements of qualified APIs, never a production execution profile."""

import argparse
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def summarize(samples):
    if not samples or any(
        type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in samples
    ):
        raise ValueError("nonempty finite measured seconds required")
    ordered = sorted(samples)
    return dict(
        samples=len(samples),
        median_seconds=statistics.median(ordered),
        p95_seconds=ordered[math.ceil(0.95 * len(ordered)) - 1],
    )


def cases():
    from oeis_learn.synthesis.contracts import decimal_integer

    result = []

    def add(name, source, indices, values, *, reason=None, old=None, result_bits=None):
        input_bits = max(abs(n).bit_length() for n in indices)
        literal_bits = max(
            [abs(int(n)).bit_length() for n in re.findall(r"\(int (-?[0-9]+)\)", source)] or [0]
        )
        result.append(
            dict(
                name=name,
                source=source,
                indices=[decimal_integer(n) for n in indices],
                supplied_bits=max(input_bits, literal_bits),
                input_bits=input_bits,
                literal_bits=literal_bits,
                result_bits=result_bits or max(abs(n).bit_length() for n in values),
                expected=[decimal_integer(n) for n in values] if reason is None else [],
                expected_reason=reason,
                old_source=old,
            )
        )

    add(
        "shared-small-add",
        "(program (return (add (int 42) (int 7))))",
        [0],
        [49],
        old="i256.const 42 i256.const 7 i256.add",
    )
    add(
        "shared-small-cancel",
        "(program (return (sub (int -42) (int -42))))",
        [0],
        [0],
        old="i256.const -42 i256.const -42 i256.sub",
    )
    for width in (64, 128, 255):
        n = (1 << (width - 1)) + 31
        add(
            f"shared-add-{width}",
            f"(program (return (add (int {n}) (int 7))))",
            [0],
            [n + 7],
            old=f"i256.const {n} i256.const 7 i256.add",
        )
    for width in (64, 256, 512, 1024, 4096):
        n = (1 << (width - 1)) + 31
        for sign in (1, -1):
            x = sign * n
            add(f"add-{width}-{sign}", "(program (return (add $index (int 1))))", [x], [x + 1])
        add(f"full-mul-{width}", "(program (return (mul $index $index)))", [n], [n * n])
    n = (1 << 4096) - 1001
    square = "(program (local $b0 bigint) (set $b0 (mul $index $index)) "
    add("generated-16384", square + "(return (mul (get $b0) (get $b0))))", [n], [n**4])
    add("generated-cancellation", square + "(return (sub (get $b0) (get $b0))))", [n], [0])
    add(
        "expected-overflow",
        square + "(return (mul (mul (get $b0) (get $b0)) $index)))",
        [n],
        [],
        reason="integer_limit",
        result_bits=20480,
    )
    add(
        "non-scoring-loop",
        "(program (local $b0 bigint) (local $c0 control) "
        "(set $b0 (int 1)) (loop $c0 (to_control $index) "
        "(set $b0 (mul (get $b0) (to_bigint (increment (get $c0)))))) (return (get $b0)))",
        list(range(100)),
        [math.factorial(n) for n in range(100)],
    )
    add(
        "non-scoring-branch",
        "(program (return (select (lt $index (int 50)) "
        "(mul $index $index) (add (mul $index (int 2)) (int 1)))))",
        list(range(100)),
        [n * n if n < 50 else 2 * n + 1 for n in range(100)],
    )
    return result


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def runtime_measurements(case, batch, repetitions):
    """Native compilation and execution are timed inside the production runtime."""
    from oeis_learn.synthesis.reference import evaluate

    # Repeated exact indices measure batch overhead, never pretend to be a dataset.
    indices = (case["indices"] * batch)[:batch]
    started = time.monotonic()
    binary = os.environ.get("OEIS_BENCHMARK_RUNTIME", "/opt/oeis-build-info/benchmark-runtime")
    proc = subprocess.run(
        [binary],
        input=json.dumps(
            dict(source=case["source"], indices=indices, repetitions=repetitions)
        ).encode(),
        capture_output=True,
        timeout=120,
    )
    subprocess_seconds = time.monotonic() - started
    if proc.returncode or len(proc.stdout) > 4 << 20:
        raise RuntimeError(
            "bounded native measurement failed: " + proc.stderr[:1000].decode(errors="replace")
        )
    report = json.loads(proc.stdout)
    if (
        report.get("version") != "bigint-measurements/v1"
        or len(report.get("samples", [])) != repetitions
    ):
        raise ValueError("incomplete native measurement report")
    reference_times = []
    for _ in range(repetitions):
        start = time.monotonic()
        reference = evaluate(case["source"], indices)
        reference_times.append(time.monotonic() - start)
        if case["expected_reason"]:
            if reference.reason != case["expected_reason"]:
                raise RuntimeError("independent failure classification disagrees")
        elif (
            reference.outcome != "ok"
            or list(reference.values) != (case["expected"] * batch)[:batch]
        ):
            raise RuntimeError("independent source interpreter disagrees")
    reference_stats = summarize(reference_times)
    expected = (case["expected"] * batch)[:batch]
    exclusions = []
    for sample in report["samples"]:
        row = sample["result"]
        if case["expected_reason"]:
            if row.get("reason") != case["expected_reason"] or row.get("values"):
                raise RuntimeError("expected overflow was not classified correctly")
        elif row.get("outcome") != "ok":
            if row.get("reason") not in {
                "term_fuel",
                "aggregate_fuel",
                "deadline",
                "arena_limit",
                "memory_limit",
                "output_limit",
            } or row.get("values"):
                raise RuntimeError("unexpected production failure on a legal arithmetic case")
            exclusions.append(dict(reason=row["reason"], coverage=row.get("coverage")))
        elif row.get("values") != expected:
            raise RuntimeError("production/independent expectation disagreement")
    if case["expected_reason"]:
        if reference.reason != case["expected_reason"]:
            raise RuntimeError("independent failure classification disagrees")
    elif reference.outcome != "ok" or list(reference.values) != expected:
        raise RuntimeError("independent source interpreter disagrees")
    report.update(
        case=case["name"],
        batch=batch,
        supplied_bits=case["supplied_bits"],
        input_bits=case.get("input_bits"),
        literal_bits=case.get("literal_bits"),
        result_bits=case["result_bits"],
        expected_reason=case["expected_reason"],
        resource_exclusions=exclusions,
        subprocess_seconds=subprocess_seconds,
        reference_seconds=reference_stats["median_seconds"],
        reference_summary=reference_stats,
        cold=report["samples"][0]["timings"],
        warm={
            key: summarize([s["timings"][key] for s in report["samples"][1:]])
            for key in ("prepare_seconds", "execution_seconds", "total_seconds")
        },
    )
    # Full outputs are independently checked above; preserve hashes, not megabytes.
    from oeis_learn.synthesis.contracts import identity

    for sample in report["samples"]:
        sample["result"]["values_sha256"] = identity(sample["result"].pop("values"))
    if case["old_source"]:
        from oeis_learn.sandbox.pipeline import Runtime, PreparedCandidate

        runtime = Runtime(backend="rust")
        old_prepare, old_execute = [], []
        for _ in range(repetitions):
            start = time.monotonic()
            prepared = runtime.prepare(case["old_source"])
            old_prepare.append(time.monotonic() - start)
            if not isinstance(prepared, PreparedCandidate):
                raise RuntimeError("retained native compilation failed")
            start = time.monotonic()
            values = [runtime.production_term(prepared, 0, 1_000_000) for _ in range(batch)]
            old_execute.append(time.monotonic() - start)
            if any(
                failure is not None or str(v) != case["expected"][0] for v, _, failure in values
            ):
                raise RuntimeError("retained small-value semantics disagreed")
        old = summarize(old_execute[1:])
        ratio = report["warm"]["execution_seconds"]["median_seconds"] / max(
            old["median_seconds"], 1e-12
        )
        report["retained"] = dict(
            prepare=summarize(old_prepare[1:]),
            execution=old,
            small_value_slowdown=ratio,
            investigation_required=ratio > 5,
        )
    else:
        report["retained"] = dict(
            excluded=True, reason="outside declared shared add/sub constant subset"
        )
    return report


def fitting_measurements(repetitions):
    """Time provisional native fit and independent admission separately, no trial callbacks."""
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.contracts import identity, decimal_integer
    from oeis_learn.synthesis.admission import VisiblePrompt, admit

    fixtures = [
        (
            "affine-large",
            "(program (return (add (mul (param.get p0) $index) (param.get p1))))",
            [str(7 * n + 100000000000000000001) for n in range(20)],
            "assignment_found",
        ),
        (
            "coefficient4096",
            "(program (return (param.get p0)))",
            [decimal_integer((1 << 4096) - 1)] * 20,
            "assignment_found",
        ),
        (
            "rank-deficient",
            "(program (return (add (param.get p0) (param.get p1))))",
            ["3"] * 20,
            "assignment_found",
        ),
        (
            "nonlinear-shared",
            "(program (return (mul (param.get p0) (param.get p0))))",
            ["4"] * 20,
            "assignment_found",
        ),
        (
            "inconsistent",
            "(program (return (param.get p0)))",
            list(map(str, range(20))),
            "unsat_in_scope",
        ),
        (
            "tainted-control",
            "(program (return (select (lt (param.get p0) $index) (int 1) (int 2))))",
            ["1"] * 20,
            "unsupported",
        ),
    ]
    samples = []
    complete = True
    with SynthesisClient(identity("phase5-measurement-fit/v1")) as client:
        runtime = client.runtime_identity("bigint-body/v1")
        for name, source, values, expected in fixtures:
            prompt = VisiblePrompt(tuple(map(str, range(20))), tuple(values))
            for i in range(repetitions):
                start = time.monotonic()
                response = client.request(
                    "fit",
                    [dict(candidate_id=str(i), source=source)],
                    prompt.indices,
                    prompt.values,
                    deadline=start + 2,
                )
                native_seconds = time.monotonic() - start
                rows = response["results"]
                native = (
                    rows[0]
                    if len(rows) == 1 and response["complete"]
                    else dict(outcome="resource_exhausted")
                )
                outcome = native["outcome"]
                if outcome not in {expected, "unknown", "resource_exhausted"}:
                    raise RuntimeError(
                        "native fitting fixture unexpected outcome: " + name + ":" + outcome
                    )
                complete &= outcome == expected
                reference_seconds = None
                if outcome == "assignment_found":
                    start = time.monotonic()
                    admitted = [
                        admit(
                            row,
                            prompt,
                            runtime_sha256=runtime,
                            limits=client.limits,
                            deadline=start + 2,
                        )
                        for row in native["assignments"]
                    ]
                    reference_seconds = time.monotonic() - start
                    if not admitted or not all(r.outcome == "verified_solution" for r in admitted):
                        raise RuntimeError("non-scoring fit independent admission failed: " + name)
                samples.append(
                    dict(
                        fixture=name,
                        outcome=outcome,
                        native_seconds=native_seconds,
                        reference_seconds=reference_seconds,
                    )
                )
    return dict(
        samples=samples,
        native=summarize([r["native_seconds"] for r in samples]),
        reference=summarize(
            [r["reference_seconds"] for r in samples if r["reference_seconds"] is not None]
        ),
        complete=complete,
    )


def neural_fixtures(shapes=None):
    """Capacity matrix plus a separate shape-matched, non-scoring cost fixture."""
    from oeis_learn.decoder.bigint_codec import encode_body

    matrix = [
        ("capacity", 64, 33),
        ("capacity", 256, 152),
        ("capacity", 1024, 560),
        ("capacity", 4096, 1024),
    ]
    if shapes is not None:
        width, tokens = shapes["max_input_bits"], shapes["max_tokens"]
        if (
            type(width) is not int
            or not 1 <= width <= 4096
            or type(tokens) is not int
            or not 1 <= tokens <= 1024
        ):
            raise ValueError("bounded prepared input/token shapes required")
        matrix.append(("preflight", width, tokens))
    fixtures = []
    for role, width, goal in matrix:
        count = min(59, max(1, math.ceil((goal - 16) / 17)))
        source = (
            "(program (local $b0 bigint) "
            + "(set $b0 (add (get $b0) (int 1))) " * count
            + "(return (get $b0)))"
        )
        tokens = encode_body(source)
        if len(tokens) < goal:
            source = source.replace("(int 1)", "(int " + "1" * (1 + goal - len(tokens)) + ")", 1)
            tokens = encode_body(source)
        base = 1 << (width - 1)
        fixtures.append(
            dict(
                fixture_role=role,
                values=[str((-1 if n % 2 else 1) * (base + n % min(base, 20))) for n in range(20)],
                tokens=tokens,
            )
        )
    return fixtures


def neural_measurements(recipe, output, device, repetitions, shapes=None):
    """Non-scoring tensor fixtures at production widths; no evaluation data mounted."""
    import torch
    from oeis_learn.fresh.bigint_worker import models, _batch_loss
    from oeis_learn.decoder.sampler import WatProgramSampler
    from oeis_learn.rl.policy import scoring_mode
    from oeis_learn.synthesis.contracts import identity
    from oeis_learn.synthesis.state import ExperimentContract, BudgetLedger
    from oeis_learn.fresh.bigint_state import CheckpointStore

    encoder, decoder, counts = models(recipe, device)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=1e-4)
    fixtures = neural_fixtures(shapes)
    records = []
    contract = ExperimentContract(
        **{
            k: identity(["non-scoring-phase5/v1", k, recipe["identity"]])
            for k in (
                "data",
                "split",
                "model",
                "inputs",
                "decoder",
                "solver",
                "runtime",
                "resources",
                "policy",
            )
        },
        track="generic_only",
        foundation=None,
        seed=recipe["model"]["seed"],
        budget=7200,
    )
    store = CheckpointStore(Path(output) / "measurement-checkpoints", contract)
    ledger = BudgetLedger(7200)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def elapsed(function):
        sync()
        start = time.monotonic()
        value = function()
        sync()
        return time.monotonic() - start, value

    for row in fixtures:
        for _ in range(repetitions):
            encoder.train()
            decoder.train()
            training, loss = elapsed(
                lambda: _batch_loss(encoder, decoder, [row] * 8, device, optimizer=optimizer)
            )
            encoder.eval()
            decoder.eval()
            teacher, _ = elapsed(lambda: _batch_loss(encoder, decoder, [row] * 8, device))
            with scoring_mode(encoder, decoder), torch.no_grad():
                # Production cached sampler, including the encoder and all masks.
                def generate():
                    memory = encoder.forward_from_sequences(
                        [[int(v) for v in row["values"]]], device=device
                    )
                    sampler = WatProgramSampler(
                        decoder,
                        max_length=1024,
                        temperature=0,
                        codec_profile="bigint-codec/v1",
                        use_cache=True,
                    )

                    def hook(_):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("non-scoring generation exceeded60s")

                    return sampler.sample_trajectories(memory, [20261016], budget_hook=hook)[0]

                deadline = time.monotonic() + 60
                generation, trajectory = elapsed(generate)
            import random
            import numpy as np

            states = dict(
                weights=dict(encoder=encoder.state_dict(), decoder=decoder.state_dict()),
                optimizer=optimizer.state_dict(),
                rng=dict(
                    python=random.getstate(),
                    numpy=np.random.get_state(),
                    torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if device == "cuda" else None,
                ),
                data_order=dict(cursor=0, permutation=list(range(8))),
                selection={},
                curriculum={},
                replay={},
            )

            def save_checkpoint():
                value = store.save(states, ledger)
                store.prune()
                return value

            checkpoint, _ = elapsed(save_checkpoint)
            records.append(
                dict(
                    fixture_role=row["fixture_role"],
                    input_bits=max(abs(int(v)).bit_length() for v in row["values"]),
                    difference_bits=max(
                        abs(int(b) - int(a)).bit_length()
                        for a, b in zip(row["values"], row["values"][1:])
                    ),
                    target_tokens=len(row["tokens"]),
                    generated_tokens=max(0, len(trajectory.tokens) - 1),
                    training=training,
                    teacher_forcing=teacher / 8,
                    generation=generation,
                    checkpoint=checkpoint,
                )
            )
    import shutil

    shutil.rmtree(store.root)
    measured_costs = [r for r in records if r["fixture_role"] == "preflight"] or records
    return dict(
        shape_sha256=identity(shapes) if shapes is not None else None,
        device=str(next(encoder.parameters()).device),
        hip=torch.version.hip,
        parameters=counts,
        fixture_sha256=identity(fixtures),
        scoring_data=False,
        samples=records,
        summaries={
            k: summarize([r[k] for r in measured_costs])
            for k in ("training", "teacher_forcing", "generation", "checkpoint")
        },
    )


def admit_cost_recipe(report, recipe, shapes, output):
    """Bind observed costs to the actual image/profile and complete prepared shape audit."""
    from copy import deepcopy
    from oeis_learn.synthesis.contracts import identity
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.cost_policy import screen_cost
    import yaml

    counts = {"training": 512, "parameter_development": 128, "structural_development": 128}
    forecast = screen_cost(
        **report["costs"],
        updates=recipe["learning"]["max_updates"],
        cadence=recipe["learning"]["validation_every"],
        budget=recipe["budgets"]["training_seconds"],
    )
    if (
        report.get("status") != "PASS"
        or not report.get("neural", {}).get("hip")
        or not report["neural"]["device"].startswith("cuda")
        or not forecast["within_budget"]
        or report.get("fitting", {}).get("complete") is not True
        or report["neural"].get("shape_sha256") != identity(shapes)
        or shapes.get("counts") != counts
        or shapes.get("tiny_count") != 32
        or shapes.get("native_admitted") is not True
        or any(not shapes.get("stages", {}).get(s) for s in ("A", "B", "C", "D"))
        or shapes.get("max_input_bits", 4097) > 4096
        or shapes.get("max_tokens", 1025) > 1024
        or not shapes.get("prepared_sha256")
        or not shapes.get("split_sha256")
    ):
        raise ValueError(
            "HIP, complete shape-audited512/128/128+diverse32 and passing80% cost screen required"
        )
    value = deepcopy(recipe)
    value.pop("identity", None)
    value["learning"]["cost_preflight"] = dict(
        image=report["image"],
        profile_sha256=identity(
            {k: recipe[k] for k in ("execution", "inputs", "model", "fitting", "search")}
        ),
        scoring_data=False,
        costs=report["costs"],
        fixture_sha256=identity(
            dict(
                fixtures=report["neural"]["fixture_sha256"],
                shapes=shapes,
                runtime_cases=[r["case"] for r in report["cases"]],
            )
        ),
    )
    resolve(value)
    path = Path(output) / "measured-recipe.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return dict(
        path=path.name,
        shape_sha256=identity(shapes),
        qualification="bounded cost estimate, not transfer evidence",
    )


def run(
    output,
    *,
    repetitions=3,
    batches=(1, 20, 100),
    neural=False,
    device="cpu",
    config=None,
    shapes=None,
):
    from oeis_learn.synthesis.contracts import identity
    from oeis_learn.experiments.source_snapshot import collect_snapshot

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "benchmark.json").exists():
        raise ValueError("new output directory required; preserve previous measurements")
    report = dict(
        version="bigint-performance/v1",
        status="FAIL",
        qualified=False,
        python=sys.version,
        source_sha256=identity(collect_snapshot(ROOT)),
        image=os.environ.get("OEIS_QUALIFICATION_IMAGE"),
        cases=[],
        semantics="production bigint-resources/v1; no wider-input profile",
        accounting="prepare includes parse/lower/candidate compilation; execute includes fresh stores, loading and serialization; runtime compilation separate; warm reuses runtime module only",
    )
    try:
        from oeis_learn.fresh.bigint_config import packaged_artifacts
        from oeis_learn.fresh.common import file_hash

        if neural and device == "cuda":
            import torch

            if not torch.version.hip or not torch.cuda.is_available():
                raise RuntimeError("actual HIP required; CPU fallback prohibited")

        report["artifacts"] = packaged_artifacts()
        report["measurement_binary_sha256"] = file_hash(
            os.environ.get("OEIS_BENCHMARK_RUNTIME", "/opt/oeis-build-info/benchmark-runtime")
        )
        if not 2 <= repetitions <= 5 or not batches or any(b not in (1, 20, 100) for b in batches):
            raise ValueError("2..5 repetitions and batches1/20/100 required")
        for case in cases():
            for batch in batches:
                report["active_measurement"] = dict(case=case["name"], batch=batch)
                report["cases"].append(runtime_measurements(case, batch, repetitions))
                write(output / "partial.json", report)
                print("MEASURED", case["name"], "batch", batch, flush=True)
        report["active_measurement"] = dict(phase="fitting")
        report["fitting"] = fitting_measurements(repetitions)
        import resource

        report["resource_metrics"] = dict(
            peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
            accounting="Linux RUSAGE_CHILDREN maximum among reaped children, not aggregate resident bytes; per-evaluation fuel is retained",
        )
        if neural:
            report["active_measurement"] = dict(phase="neural")
            from oeis_learn.fresh.bigint_config import load, packaged_artifacts

            recipe = load(
                config or ROOT / "configs/foundation/bigint_smoke.yaml",
                artifacts=packaged_artifacts(),
            )
            report["neural"] = neural_measurements(
                recipe, output, device, repetitions, shapes=shapes
            )
            from oeis_learn.fresh.cost_policy import screen_cost

            summaries = report["neural"]["summaries"]
            costs = {k + "_p95": summaries[k]["p95_seconds"] for k in summaries}
            costs.update(
                fit_p95=report["fitting"]["native"]["p95_seconds"],
                reference_p95=report["fitting"]["reference"]["p95_seconds"],
                hidden_scoring_p95=max(
                    c["runtime_compile_seconds"]
                    + c["warm"]["total_seconds"]["p95_seconds"]
                    + c["reference_summary"]["p95_seconds"]
                    for c in report["cases"]
                    if c["batch"] == 100
                ),
            )
            forecast = screen_cost(
                **costs,
                updates=recipe["learning"]["max_updates"],
                cadence=recipe["learning"]["validation_every"],
                budget=recipe["budgets"]["training_seconds"],
            )
            report["cost_projection"] = forecast
            report["costs"] = costs
            report["preflight_eligible"] = False
            report["preflight_reason"] = (
                "tensor cost screen only; matched complete dataset size/complexity and hardware binding must be reviewed before recipe admission"
            )
        report["slowdown_investigations"] = [
            dict(case=r["case"], batch=r["batch"], ratio=r["retained"]["small_value_slowdown"])
            for r in report["cases"]
            if r["retained"].get("investigation_required")
        ]
        report["status"] = "PASS"
        report.pop("active_measurement", None)
        if neural and shapes is not None:
            try:
                report["measured_recipe"] = admit_cost_recipe(report, recipe, shapes, output)
                report["preflight_eligible"] = True
                report["preflight_reason"] = (
                    "HIP cost estimate and complete prepared shape audit bound to current image/profile"
                )
            except ValueError as exc:
                report["preflight_reason"] = str(exc)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["report_sha256"] = identity(report)
        write(output / "benchmark.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batches", default="1,20,100")
    parser.add_argument("--neural", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--config")
    parser.add_argument("--shape-manifest")
    args = parser.parse_args()
    try:
        run(
            args.output,
            repetitions=args.repetitions,
            batches=tuple(map(int, args.batches.split(","))),
            neural=args.neural,
            device=args.device,
            config=args.config,
            shapes=json.loads(Path(args.shape_manifest).read_text())
            if args.shape_manifest
            else None,
        )
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
