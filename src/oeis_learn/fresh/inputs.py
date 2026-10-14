"""CPU preparation using actual indexed source and strict program admission."""

import time
import uuid
from collections import Counter
from pathlib import Path

from .admission_sampling import SamplerShards
from .bulk import import_snapshot
from .common import digest, file_hash, inventory, lock, read, verify, write


def check_inputs(root):
    root = Path(root)
    manifest = read(root / "prepared.json")
    if manifest["version"] != "fresh-inputs/v1" or manifest["identity"] != digest(
        {k: v for k, v in manifest.items() if k != "identity"}
    ):
        raise ValueError("prepared input identity mismatch")
    verify(root, manifest["files"])
    examples = read(root / "examples.json")
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
            if split == "training" and not entry["training_eligible"]:
                raise ValueError("excluded program in training view")
            if (
                split == "development"
                and split_for_structure(entry["ir_sha256"], archive.holdouts.seed) != "development"
            ):
                raise ValueError("development row is outside its structural complement")
    return manifest


def prepare(config, cache, output, *, backend="rust"):
    import yaml

    from oeis_learn.data.generic_programs import load_sampler_config
    from oeis_learn.data.program_admission import validate_sample
    from oeis_learn.data.structural_splits import StructuralHoldouts, split_for_structure
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.evaluation.foundation_cohort import build_cohort
    from oeis_learn.evaluation.release_source import (
        audit_release,
        prepare_release,
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
            for key in keys:
                packet = archive.get(key)
                entry = packet["entry"]
                item = {
                    "program_id": key,
                    "values": entry["conditioning_values"],
                    "tokens": packet["tokens"],
                    "structure": entry["ir_sha256"],
                }
                if entry["training_eligible"]:
                    identity = digest(item["values"])
                    old = training.get(identity)
                    if old is None or (len(item["tokens"]), key) < (
                        len(old["tokens"]),
                        old["program_id"],
                    ):
                        training[identity] = item
                elif (
                    split_for_structure(entry["ir_sha256"], holdouts.seed) == "development"
                    and entry["context_tokens"] <= 1024
                    and contract_digest(item["values"]) not in holdouts.reserved_prefixes
                ):
                    development[entry["ir_sha256"]] = item

        observe(archive.freeze()["entries"])
        ledger.transition("bootstrap")
        reasons = Counter()
        try:
            with FoundationRunner(
                root / "execution",
                backend=backend,
                workers=sampler_config["workers"],
                cache_bytes=128 << 20,
            ) as runner:
                pending = {}
                for counter in range(series.limit):
                    if (
                        len(training) >= spec["training_prompts"]
                        and len(development) >= spec["development_structures"]
                    ):
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
                            f"ADMIT counter={counter}/{series.limit} shard={counter // sampler_config['max_attempts']} train={len(training)}/{spec['training_prompts']} structural_dev={len(development)}/{spec['development_structures']} remaining={ledger.remaining_ns()/1e9:.0f}s",
                            flush=True,
                        )
        finally:
            ledger.transition("paused")
            write(
                root / "archive-preparation-report.json",
                {
                    "training_prompts": len(training),
                    "development_structures": len(development),
                    "training_structures": len({r["structure"] for r in training.values()}),
                    "training_body_tokens": sorted(len(r["tokens"]) for r in training.values()),
                    "source_census": built["census"],
                    "reasons": dict(reasons),
                    "sampling_shards": series.shards,
                    "candidate_limit": series.limit,
                    "budget": ledger.snapshot(),
                },
            )
        if (
            len(training) < spec["training_prompts"]
            or len(development) < spec["development_structures"]
        ):
            raise ValueError("archive shortfall; journal and remaining lifetime budget retained")
        release = root / "release"
        if not release.exists():
            prepare_release(root / "indexed-source/manifest.json", root / "cohort.yaml", release)
        audit = audit_release(release / "cohort", archive=archive, snapshot=archive.freeze())
        write(root / "release-source-audit.json", audit, immutable=True)
        # The learner receives only twenty visible values and full body labels.
        write(
            root / "examples.json",
            {
                "training": sorted(training.values(), key=lambda x: x["program_id"]),
                "development": sorted(development.values(), key=lambda x: x["program_id"]),
            },
            immutable=True,
        )
        truths = {split: [] for split in ("development", "final")}
        for group in built["groups"]:
            if group["partition"] in truths:
                record = built["records"][group["representative"]]
                truths[group["partition"]].append(
                    {
                        "group_id": group["group_id"],
                        "values": record["values"],
                        "members": group["members"],
                    }
                )
        write(root / "truth.json", truths, immutable=True)
        files = inventory(root, exclude=("execution", "decisions", "archive-budget", "release"))
        manifest = {
            "version": "fresh-inputs/v1",
            "config": config,
            "files": files,
            "import_report": imported,
            "archive_snapshot": archive.freeze(),
            "qualified": False,
            "purpose": "fresh training diagnostic; historical and release gates remain independent",
        }
        manifest["identity"] = digest(manifest)
        write(root / "prepared.json", manifest, immutable=True)
        return manifest
