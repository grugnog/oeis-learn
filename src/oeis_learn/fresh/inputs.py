"""CPU preparation using actual indexed source and strict program admission."""

import time
import uuid
from collections import Counter
from pathlib import Path

from .admission_sampling import SamplerShards
from .bulk import import_snapshot
from .common import digest, file_hash, lock, read, verify, write
from .input_export import collect, export


def check_inputs(root):
    root = Path(root)
    manifest = read(root / "prepared.json")
    if manifest["version"] != "fresh-inputs/v1" or manifest["identity"] != digest(
        {k: v for k, v in manifest.items() if k != "identity"}
    ):
        raise ValueError("prepared input identity mismatch")
    verify(root, manifest["files"])
    examples = read(root / "examples.json")
    completion = manifest.get("admission_completion")
    if completion is not None:
        from .input_sealing import completion_record

        report = read(root / "archive-preparation-report.json")
        expected = completion_record(
            manifest["config"],
            len({digest(r["values"]) for r in examples["training"]}),
            report["development_structures"],
            file_hash(root / "archive-plan.json"),
            completion["budget"],
        )
        if (
            completion != expected
            or read(root / "admission-completion.json") != completion
            or len(examples["development"]) != completion["exported_development_structures"]
        ):
            raise ValueError("partial admission completion differs from sealed inputs")
    if (
        len({digest(r["values"]) for r in examples["training"]}) < 128
        or not examples["development"]
    ):
        raise ValueError("128 distinct admitted prompts and structural development data required")
    from oeis_learn.data.structural_splits import split_for_structure
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.experiments.learning_profiles import load_holdouts

    archive = ProgramArchive(root / "archive", holdouts=load_holdouts(read(root / "holdouts.json")))
    if archive.freeze() != manifest["archive_snapshot"]:
        raise ValueError("prepared archive snapshot changed")
    for split, rows in examples.items():
        if split not in {"training", "development"}:
            raise ValueError("unknown learner view partition")
        for row in rows:
            packet = archive.get(row["program_id"])
            entry = packet["entry"]
            if (
                row["values"] != entry["conditioning_values"]
                or row["tokens"] != packet["tokens"]
                or row["structure"] != entry["ir_sha256"]
            ):
                raise ValueError("learner row differs from admitted archive")
            expected_stage = (
                packet.get("provenance", {}).get("sample", {}).get("statistics", {}).get("stage")
            )
            if row.get("stage") != expected_stage:
                raise ValueError("learner curriculum stage differs from admitted provenance")
            if split == "training" and not entry["training_eligible"]:
                raise ValueError("excluded program in training view")
            if (
                split == "development"
                and split_for_structure(entry["ir_sha256"], archive.holdouts.seed) != "development"
            ):
                raise ValueError("development row is outside its structural complement")
    if manifest.get("config", {}).get("run", {}).get("learning", {}).get("curriculum"):
        if any({r.get("stage") for r in rows} != {0, 1, 2, 3} for rows in examples.values()):
            raise ValueError(
                "curriculum requires admitted training/development coverage of all four stages"
            )
    return manifest


def prepare(config, cache, output, *, backend="rust"):
    import yaml

    from oeis_learn.data.sampler_factory import load_sampler_config
    from oeis_learn.data.program_admission import validate_sample
    from oeis_learn.data.structural_splits import StructuralHoldouts
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.evaluation.foundation_cohort import build_cohort
    from oeis_learn.evaluation.release_source import (
        read_indexed_source,
    )
    from oeis_learn.experiments.artifacts import compute_canonical_digest as contract_digest
    from oeis_learn.experiments.learning_profiles import holdout_profile
    from oeis_learn.sandbox.pipeline import runtime_identity
    from oeis_learn.sandbox.runner import FoundationRunner
    from oeis_learn.tracking.budget_ledger import BudgetLedger

    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / "prepare.lock"):
        if (root / "prepared.json").exists():
            saved = check_inputs(root)
            if saved["config"] != config:
                raise ValueError("prepared config changed; create a new input set")
            return saved
        imported = import_snapshot(cache, root / "indexed-source")
        rows, _ = read_indexed_source(root / "indexed-source/manifest.json")
        cohort = dict(
            schema_version="foundation/v1",
            profile="prefix20_total100_affine_v2",
            **config["cohort"],
        )
        write(root / "cohort.yaml", yaml.safe_dump(cohort, sort_keys=True).encode(), immutable=True)
        built = build_cohort(rows, affine=True, **config["cohort"])
        prefixes = frozenset(
            contract_digest(built["records"][m]["values"][:20])
            for g in built["groups"]
            if g["partition"] != "reserved"
            for m in g["members"]
        )
        holdouts = StructuralHoldouts(seed=config["cohort"]["seed"], reserved_prefixes=prefixes)
        write(root / "holdouts.json", holdout_profile(holdouts), immutable=True)
        sampler_config = load_sampler_config(Path(config["sampler_config"]))
        if (
            config.get("run", {}).get("learning", {}).get("curriculum")
            and sampler_config["profile"] != "connected_sampler_v1"
        ):
            raise ValueError("curriculum requires the versioned connected sampler")
        spec = config["archive"]
        series = SamplerShards(sampler_config, spec.get("sampling_shards", 1))
        plan = {
            "version": "fresh-admission-plan/v1",
            "config": config,
            "source_manifest": file_hash(root / "indexed-source/manifest.json"),
            "holdouts": holdouts.identity(),
            "sampler": sampler_config,
            "runtime": runtime_identity(backend),
            "execution": {
                "workers": sampler_config["workers"],
                "chunk_size": sampler_config["chunk_size"],
                "cache_per_worker_bytes": 128 << 20,
            },
        }
        write(root / "archive-plan.json", plan, immutable=True)
        archive = ProgramArchive(root / "archive", holdouts=holdouts)
        ledger = BudgetLedger(
            root / "archive-budget", str(uuid.UUID(digest(plan)[7:39])), spec["seconds"] * 10**9
        )

        training, development = {}, {}

        def observe(keys):
            collect(archive, keys, training, development)

        def target_reached():
            complete = (
                len(training) >= spec["training_prompts"]
                and len(development) >= spec["development_structures"]
            )
            if config.get("run", {}).get("learning", {}).get("curriculum"):
                complete = complete and all(
                    {row.get("stage") for row in view.values()} >= {0, 1, 2, 3}
                    for view in (training, development)
                )
            return complete

        observe(archive.freeze()["entries"])
        ledger.transition("bootstrap")
        reasons = Counter()
        stop_reason = "interrupted"
        try:
            with FoundationRunner(
                root / "execution",
                backend=backend,
                workers=sampler_config["workers"],
                cache_bytes=128 << 20,
            ) as runner:
                pending = {}
                for counter in range(series.limit):
                    if target_reached():
                        break
                    if ledger.remaining_ns() <= 0:
                        break
                    # Execute bounded chunks concurrently; admit strictly in
                    # sample-counter order, independent of completion speed.
                    if counter % sampler_config["chunk_size"] == 0:
                        for candidate in range(
                            counter,
                            min(
                                counter + sampler_config["chunk_size"],
                                series.limit,
                            ),
                        ):
                            if (root / "decisions" / f"{candidate:08d}.json").exists():
                                continue
                            draw_sampler, local = series.select(candidate)
                            draw = draw_sampler.sample(local)
                            if validate_sample(draw, draw_sampler) is None:
                                pending[candidate] = runner.pool.submit(
                                    draw["canonical_source"],
                                    list(range(100)),
                                    request_id=contract_digest([digest(plan), candidate]),
                                    deadline_ns=time.monotonic_ns() + ledger.remaining_ns(),
                                )
                    sampler, local = series.select(counter)
                    sample = sampler.sample(local)
                    path = root / "decisions" / f"{counter:08d}.json"
                    if path.exists():
                        old = read(path)
                        if old["sample_sha256"] != contract_digest(sample):
                            raise ValueError("sample changed on preparation resume")
                        if (
                            old["key"] is not None
                            and contract_digest(archive.get(old["key"])["provenance"]["sample"])
                            != old["sample_sha256"]
                        ):
                            raise ValueError("admission decision binding changed")
                        reasons[old["reason"] or "admitted"] += 1
                        continue
                    reason, key = validate_sample(sample, sampler), None
                    if reason is None:
                        request_id = contract_digest([digest(plan), counter])
                        evidence = (
                            pending.pop(counter).result()
                            if counter in pending
                            else runner.run_single(
                                sample["canonical_source"],
                                range(100),
                                request_id=request_id,
                                deadline_ns=time.monotonic_ns() + ledger.remaining_ns(),
                            )
                        )
                        if evidence.reason and "disagreement" in evidence.reason:
                            raise RuntimeError("native/reference disagreement during admission")
                        reason = evidence.outcome
                        if reason is None:
                            if (
                                len(evidence.outputs) != 100
                                or evidence.outputs != evidence.reference_outputs
                            ):
                                raise RuntimeError("incomplete independent execution")
                            outputs = list(map(int, evidence.outputs))
                            key = archive.admit(
                                sample["canonical_source"],
                                outputs[:20],
                                outputs,
                                origin="generic_sample",
                                provenance={"config": sampler.config, "sample": sample},
                                runner=runner,
                                request_id=request_id,
                            )
                            observe([key])
                    write(
                        path,
                        {"sample_sha256": contract_digest(sample), "key": key, "reason": reason},
                        immutable=True,
                    )
                    reasons[reason or "admitted"] += 1
                    if counter % 25 == 0:
                        ledger.heartbeat()
                        print(
                            f"ADMIT counter={counter}/{series.limit} shard={counter // sampler_config['max_attempts']} train={len(training)}/{spec['training_prompts']} structural_dev={len(development)}/{spec['development_structures']} remaining={ledger.remaining_ns() / 1e9:.0f}s",
                            flush=True,
                        )
            stop_reason = (
                "target_reached"
                if target_reached()
                else "lifetime_budget"
                if ledger.remaining_ns() <= 0
                else "candidate_limit"
            )
        finally:
            ledger.transition("paused")
            write(
                root / "archive-preparation-report.json",
                {
                    "training_prompts": len(training),
                    "development_structures": len(development),
                    "training_structures": len({r["structure"] for r in training.values()}),
                    "curriculum_counts": {
                        split: dict(
                            Counter(str(r.get("stage", "unlabelled")) for r in view.values())
                        )
                        for split, view in (("training", training), ("development", development))
                    },
                    "training_body_tokens": sorted(len(r["tokens"]) for r in training.values()),
                    "source_census": built["census"],
                    "reasons": dict(reasons),
                    "sampling_shards": series.shards,
                    "candidate_limit": series.limit,
                    "stop_reason": stop_reason,
                    "budget": ledger.snapshot(),
                },
            )
        if not target_reached():
            raise ValueError(
                f"archive shortfall ({stop_reason}): train={len(training)}/{spec['training_prompts']}, "
                f"structural_dev={len(development)}/{spec['development_structures']}, "
                f"candidate_limit={series.limit}; journal and remaining lifetime budget retained"
            )
        return export(root, config, imported, built, archive, training, development)
