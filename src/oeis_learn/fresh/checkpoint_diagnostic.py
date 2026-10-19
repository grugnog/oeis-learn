"""Read-only diagnosis of an existing checkpoint under explicitly recorded code."""

from pathlib import Path
from .common import digest, read, write


def diagnose(inputs, output):
    from .worker import models, runtime
    from .diagnostics import evaluate
    from .bootstrap_config import validate_learning
    from oeis_learn.tracking.role_coordinator import QueueRunner
    from oeis_learn.tracking.role_worker import validation_loss
    from oeis_learn.tracking.training_checkpoint import QualifiedCheckpointStore

    inputs, output = Path(inputs), Path(output)
    settings, examples = read(inputs / "settings.json"), read(inputs / "examples.json")
    if digest(examples) != settings["examples_sha256"]:
        raise ValueError("diagnostic inputs differ from source run")
    learning = validate_learning(settings["learning"])
    encoder, decoder, counts = models(settings)
    store = QualifiedCheckpointStore(inputs / "checkpoints", free_floor_bytes=0)
    # Verify the original checkpoint's contract/profile/runtime. This is an
    # explicit weights-only diagnostic, never a resume under changed code.
    _, base, state = store.load_learning(
        inputs / "checkpoints" / settings["checkpoint"],
        expected_contract=settings["source_contract"],
        expected_profile=settings["model"]["identity_sha256"],
        expected_runtime=settings["source_runtime"],
    )
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    import time

    deadline = time.monotonic() + learning["diagnostic_seconds"]
    if settings.get("native_fitting"):
        learning = dict(learning, diagnostic_seconds=learning["diagnostic_seconds"] // 2)
    result = evaluate(
        encoder,
        decoder,
        examples,
        learning,
        QueueRunner(output / "runtime-queue"),
        full_development=True,
        compare_generation=True,
        detail=True,
    )
    if settings.get("native_fitting"):
        result["native_fitting"] = retained_fits(result, examples, settings, output, deadline)
        result["analysis_sha256"] = settings["analysis_sha256"]
    result["checkpoint_sha256"] = base["blob_sha256"]
    result["source_runtime_sha256"] = settings["source_runtime"]
    result["diagnostic_runtime"] = runtime(settings["device"])
    result["parameter_count"] = counts
    result["validation_loss"] = validation_loss(
        encoder,
        decoder,
        [
            {"tokens": r["tokens"], "entry": {"conditioning_values": r["values"]}}
            for r in examples["development"]
        ],
        settings["device"],
        limit=None,
        batch_size=4,
    )
    result["validation_denominator"] = len(examples["development"])
    result["mode"] = "weights-only diagnostic; optimizer, source run and budgets unchanged"
    result["qualified"] = False
    result["identity"] = digest({k: v for k, v in result.items() if k != "identity"})
    write(output / "diagnostic.json", result)
    return result


def retained_fits(report, examples, settings, output, deadline):
    """Fit each original greedy candidate without changing its numerical profile."""
    import time
    from .search import native_variants
    from .diagnostics import execute_values
    from oeis_learn.decoder.program_codec import decode_body
    from oeis_learn.tracking.role_coordinator import QueueRunner
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.reference import evaluate_retained
    from oeis_learn.synthesis.admission import source_identity

    mode = settings["native_fitting"]
    records = {}
    unsupported = 0
    denominator = 0
    verified = 0
    rescued = 0
    seconds = 0.0
    runner = QueueRunner(Path(output) / "runtime-queue")
    with SynthesisClient(
        settings["source_contract"], cache_root=Path(output) / "native-cache"
    ) as client:
        for split, summary in report["splits"].items():
            byid = {r["program_id"]: r for r in examples[split]}
            for candidate in summary["attempts"]:
                denominator += 1
                key = split + ":" + candidate["program_id"]
                row = byid[candidate["program_id"]]
                source = candidate.get("source")
                start = time.monotonic()
                record = dict(
                    raw=bool(candidate["matched"]),
                    same_candidate_source=source,
                    exact=False,
                    full_continuation100=False,
                    outcome="incomplete",
                )
                records[key] = record
                if mode == "raw" or not source or not candidate["complete"] or start >= deadline:
                    continue
                result = native_variants(
                    source, row["values"], client, deadline=min(deadline, start + 4)
                )
                seconds += time.monotonic() - start
                record["outcome"] = result["outcome"]
                record["fit"] = result
                unsupported += result["outcome"] == "unsupported"
                if not result["admitted"]:
                    continue
                from .synthesis_policy import select_visible

                chosen = select_visible(
                    [
                        dict(
                            source=r["source"],
                            source_sha256=r["source_sha256"],
                            verified=True,
                            visible_count=20,
                            cost=seconds,
                        )
                        for r in result["admitted"]
                    ]
                )
                record["selected"] = chosen
                record["exact"] = True
                verified += 1
                oracle, failure = execute_values(
                    runner,
                    decode_body(row["tokens"]),
                    range(100),
                    digest([key, "retained-oracle"]),
                    int(deadline * 1e9),
                )
                if failure or time.monotonic() >= deadline:
                    continue
                indices = list(map(str, range(100)))
                production = client.execute(
                    chosen["source"], indices, profile="wat_i256_checked_v1", deadline=deadline
                )
                independent = evaluate_retained(
                    chosen["source"], indices, client.limits, deadline=deadline
                )
                if production.get("outcome") == "ok" and independent.outcome == "ok":
                    if production["values"] != list(independent.values):
                        raise RuntimeError("retained production/reference disagreement")
                    record["full_continuation100"] = production["values"] == oracle
                    record["selected_source_sha256"] = source_identity(chosen["source"])
                    rescued += record["full_continuation100"] and not record["raw"]
    return dict(
        mode=mode,
        profile="wat_i256_checked_v1",
        denominator=denominator,
        unsupported=unsupported,
        unsupported_fraction=unsupported / denominator if denominator else None,
        visible_fit20=verified,
        same_candidate_rescue100=rescued,
        fit_and_reference_seconds=seconds,
        records=records,
        scope="finite100-index match; no proof/novelty claim",
        parent_mutated=False,
    )
