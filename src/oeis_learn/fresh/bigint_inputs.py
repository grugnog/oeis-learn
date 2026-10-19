"""Exact matched archives; scoring truth is published only in private views."""

from collections import Counter
from pathlib import Path
import time
from .common import read, write, file_hash, inventory, verify, lock
from .bigint_splits import choose_splits, diverse_tiny, overlap_groups, paired_tokens
from oeis_learn.data.bigint_programs import BigintSampler, materialize
from oeis_learn.synthesis.contracts import identity, canonical_integer, ResourceProfile
from oeis_learn.synthesis.client import SynthesisClient


def visible_record(row, view):
    if view not in {"concrete", "parameterized"}:
        raise ValueError("unknown matched view")
    return dict(
        id=row["id"],
        family=row["family"],
        stage=row.get("stage", "OEIS"),
        indices=row["indices"][:20],
        values=row["values"][:20],
        horizon_indices=row["indices"][:100],
        source=row["source"] if view == "concrete" else row["parameter_source"],
        tokens=paired_tokens(row)[view],
        origin=row.get("origin", "generic_random"),
    )


def _cohorts(indexed, generic, config):
    rows = []
    excluded = Counter()
    for item in indexed:
        try:
            if item["metadata"]["finite"] or item["metadata"]["table"]:
                raise ValueError("finite/table source")
            indices = [str(n) for n in item["indices"][:100]]
            values = item["values"][:100]
            if len(indices) != 100 or len(values) != 100:
                raise ValueError("100-index horizon missing")
            ns = [canonical_integer(n) for n in indices]
            for value in values[:20]:
                canonical_integer(value)
            for value in values[20:]:
                canonical_integer(value, bits=16384)
            if any(a >= b for a, b in zip(ns, ns[1:])):
                raise ValueError("source indices not strictly increasing")
            rows.append(
                dict(
                    id="oeis:" + item["record_id"],
                    family="oeis:" + item["record_id"],
                    indices=indices,
                    values=values,
                )
            )
        except ValueError as exc:
            excluded[str(exc)] += 1
    groups = overlap_groups([*generic, *rows])
    used = {groups[row["id"]] for row in generic}
    selected = []
    for row in sorted(rows, key=lambda r: identity([config["data"]["seed"], r["id"]])):
        if groups[row["id"]] in used:
            excluded["observed prefix/shift overlap"] += 1
            continue
        used.add(groups[row["id"]])
        selected.append(row)
    dev = config["data"]["oeis_development"]
    final = config["data"]["oeis_final"]
    if len(selected) < dev + final:
        raise ValueError(
            f"OEIS cohort shortfall: {len(selected)}/{dev + final}; exclusions={dict(excluded)}"
        )
    return dict(development=selected[:dev], final=selected[dev : dev + final]), dict(excluded)


def publish(
    root, config, selected, cohorts, *, source, exclusions, admissions=None, assistance=None
):
    root = Path(root)
    splits = selected["splits"]
    tiny = diverse_tiny(splits["training"])
    views = {
        view: {split: [visible_record(row, view) for row in rows] for split, rows in splits.items()}
        for view in ("concrete", "parameterized")
    }
    for view in views:
        for visible, row in zip(views[view]["training"], splits["training"]):
            visible["oracle_source"] = row["source"]
        for row in assistance or []:
            visible = visible_record(row, view)
            visible["oracle_source"] = row["source"]
            views[view]["training"].append(visible)
    tiny_ids = [r["id"] for r in tiny]
    write(root / "admissions.json", admissions or {}, immutable=True)
    if assistance:
        write(
            root / "assistance.json",
            dict(
                manifest_sha256=identity(read_assistance(config)["manifest"]),
                demonstrations=[r["id"] for r in assistance],
            ),
            immutable=True,
        )
    write(root / "examples.json", dict(views=views, tiny_ids=tiny_ids), immutable=True)
    truths = {
        r["id"]: dict(indices=r["indices"], values=r["values"])
        for rows in [*splits.values(), assistance or []]
        for r in rows
    }
    write(root / "private/synthetic-truth.json", truths, immutable=True)
    cohort_refs = {}
    for split, rows in cohorts.items():
        truth = {r["id"]: dict(indices=r["indices"], values=r["values"]) for r in rows}
        write(root / f"private/oeis-{split}.json", truth, immutable=True)
        write(
            root / f"private/oeis-{split}-prompts.json",
            [
                dict(
                    id=r["id"],
                    indices=r["indices"][:20],
                    values=r["values"][:20],
                    horizon_indices=r["indices"],
                )
                for r in rows
            ],
            immutable=True,
        )
        cohort_refs[split] = dict(identity=identity(truth), count=len(rows))
    write(
        root / "split-manifest.json",
        {k: v for k, v in selected.items() if k != "splits"},
        immutable=True,
    )
    write(root / "cohorts.json", cohort_refs, immutable=True)
    manifest = dict(
        version="bigint-inputs/v1",
        config=config,
        source=source,
        split_sha256=file_hash(root / "split-manifest.json"),
        cohorts=cohort_refs,
        counts={k: len(v) for k, v in splits.items()},
        exclusions=exclusions,
        files=inventory(root, exclude=("sampling", "native-cache")),
        native_admitted=True,
        assisted_count=len(assistance or []),
        limitations="observed100-index and20-window grouping; not universal equivalence",
    )
    manifest["identity"] = identity(manifest)
    write(root / "prepared.json", manifest, immutable=True)
    return manifest


def check(root):
    root = Path(root)
    manifest = read(root / "prepared.json")
    if manifest["version"] != "bigint-inputs/v1" or manifest["identity"] != identity(
        {k: v for k, v in manifest.items() if k != "identity"}
    ):
        raise ValueError("prepared bigint identity mismatch")
    verify(root, manifest["files"])
    if (
        manifest["counts"]
        != {"training": 512, "parameter_development": 128, "structural_development": 128}
        or manifest["native_admitted"] is not True
    ):
        raise ValueError("complete native-admitted split counts required")
    examples = read(root / "examples.json")
    for view in ("concrete", "parameterized"):
        for split, rows in examples["views"][view].items():
            if len(rows) != manifest["counts"][split] + (
                manifest.get("assisted_count", 0) if split == "training" else 0
            ) or any(
                len(row["values"]) != 20 or len(row["horizon_indices"]) != 100 for row in rows
            ):
                raise ValueError("changed matched learner view")
    if len(examples["tiny_ids"]) != 32:
        raise ValueError("diverse32 missing")
    return manifest


def prepare(config, cache, output, *, client_factory=SynthesisClient):
    from .bulk import import_snapshot
    from oeis_learn.evaluation.release_source import read_indexed_source

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / "prepare.lock"):
        if (root / "prepared.json").exists():
            saved = check(root)
            if saved["config"] != config:
                raise ValueError("prepared recipe changed; use a new output")
            return saved
        imported = import_snapshot(cache, root / "sampling/indexed-source")
        indexed, _ = read_indexed_source(root / "sampling/indexed-source/manifest.json")
        start = (
            read(root / "preparation-budget.json")
            if (root / "preparation-budget.json").exists()
            else dict(allowance=config["budgets"]["preparation_seconds"], debited=True)
        )
        if (root / "preparation-budget.json").exists():
            raise ValueError("interrupted preparation allowance cannot restart; create new inputs")
        write(root / "preparation-budget.json", start, immutable=True)
        deadline = time.monotonic() + start["allowance"]
        sampler = BigintSampler(config["data"]["seed"])
        rows = []
        reasons = Counter()
        accepted = {}
        limits = ResourceProfile(**config["execution"]["resources"])
        try:
            with client_factory(
                config["identity"], limits=limits, cache_root=root / "native-cache"
            ) as client:
                for counter in range(config["data"]["max_families"] * 16):
                    if time.monotonic() >= deadline:
                        break
                    try:
                        rows.append(materialize(sampler.sample(counter), limits))
                    except ValueError as exc:
                        reasons[str(exc)] += 1
                        continue
                    if counter < 2815 or (counter + 1) % 4096:
                        continue
                    while True:
                        try:
                            selected = choose_splits(rows)
                        except ValueError:
                            reasons["split shortfall"] += 1
                            break
                        rejected = set()
                        for group in selected["splits"].values():
                            for row in group:
                                if row["id"] in accepted:
                                    continue
                                result = client.execute(
                                    row["source"],
                                    row["indices"],
                                    deadline=min(deadline, time.monotonic() + 30),
                                )
                                if (
                                    result.get("outcome") == "ok"
                                    and result.get("values") == row["values"]
                                ):
                                    from oeis_learn.synthesis.admission import (
                                        source_identity,
                                        reference_identity,
                                    )

                                    accepted[row["id"]] = dict(
                                        source_sha256=source_identity(row["source"]),
                                        runtime_sha256=client.runtime_identity("bigint-body/v1"),
                                        reference_sha256=reference_identity(),
                                        resource_sha256=limits.identity,
                                        horizon_sha256=identity(
                                            dict(indices=row["indices"], values=row["values"])
                                        ),
                                        admitted=True,
                                    )
                                elif result.get("outcome") == "ok":
                                    raise RuntimeError(
                                        "production/reference disagreement in generic admission"
                                    )
                                else:
                                    rejected.add(row["id"])
                                    reasons[
                                        "native:" + str(result.get("reason", result.get("outcome")))
                                    ] += 1
                        if rejected:
                            rows = [r for r in rows if r["id"] not in rejected]
                            if time.monotonic() >= deadline:
                                break
                            continue
                        assistance = (
                            admit_assistance(config, selected, client, deadline)
                            if config["track"]["kind"] == "assisted"
                            else []
                        )
                        cohorts, source_exclusions = _cohorts(
                            indexed,
                            [r for rs in selected["splits"].values() for r in rs] + assistance,
                            config,
                        )
                        return publish(
                            root,
                            config,
                            selected,
                            cohorts,
                            source=imported,
                            exclusions=dict(generic=dict(reasons), oeis=source_exclusions),
                            admissions=accepted,
                            assistance=assistance,
                        )
            raise ValueError(
                f"complete512/128/128 preparation shortfall; exclusions={dict(reasons)}"
            )
        finally:
            write(
                root / "preparation-report.json",
                dict(
                    elapsed_seconds=time.monotonic() - (deadline - start["allowance"]),
                    exclusions=dict(reasons),
                    sampled=len(rows),
                    native_admitted=len(accepted),
                ),
            )


def read_assistance(config):
    import os
    from .common import confined
    from .provenance import validate_assistance

    root = Path(os.environ.get("OEIS_WORKSPACE", Path.cwd()))
    packet = read(confined(root, config["track"]["assisted_manifest"]))
    if set(packet) != {"manifest", "demonstrations"} or len(packet["demonstrations"]) > 4096:
        raise ValueError("bounded reviewed assistance packet required")
    validate_assistance(packet["manifest"])
    return packet


def admit_assistance(config, selected, client, deadline):
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.admission import source_identity, reference_identity
    from oeis_learn.decoder.bigint_codec import encode_body

    packet = read_assistance(config)
    manifest = packet["manifest"]
    demos = packet["demonstrations"]
    registered = dict(manifest["demonstrations"])
    result = []
    for demo in demos:
        if (
            set(demo) != {"id", "source", "indices", "values", "verification"}
            or identity({k: demo[k] for k in ("source", "indices", "values")}) != demo["id"]
            or registered.get(demo["id"]) != identity(demo["verification"])
        ):
            raise ValueError("reviewed demonstration identity/verification mismatch")
        expected = dict(
            source_sha256=source_identity(demo["source"]),
            runtime_sha256=client.runtime_identity("bigint-body/v1"),
            reference_sha256=reference_identity(),
            indices=demo["indices"],
            values=demo["values"],
            admitted=True,
        )
        if (
            demo["verification"] != expected
            or len(demo["indices"]) != 100
            or len(demo["values"]) != 100
        ):
            raise ValueError("current full100 verification required for demonstration")
        production = client.execute(demo["source"], demo["indices"], deadline=deadline)
        reference = evaluate(demo["source"], demo["indices"], client.limits, deadline=deadline)
        if (
            production.get("outcome") != "ok"
            or reference.outcome != "ok"
            or production["values"] != demo["values"]
            or list(reference.values) != demo["values"]
        ):
            raise ValueError("reviewed demonstration failed native/independent admission")
        encode_body(demo["source"])
        result.append(
            dict(
                id=demo["id"],
                source=demo["source"],
                parameter_source=demo["source"],
                indices=demo["indices"],
                values=demo["values"],
                family=identity(["named-demonstration", demo["id"]]),
                stage="D",
                origin="named_demonstration",
            )
        )
    if set(registered) != {r["id"] for r in result}:
        raise ValueError("complete reviewed demonstration roster required")
    generic = [r for rows in selected["splits"].values() for r in rows]
    groups = overlap_groups(generic + result)
    held = {
        groups[r["id"]]
        for split, rows in selected["splits"].items()
        if split != "training"
        for r in rows
    }
    demo_groups = {groups[r["id"]] for r in result}
    if demo_groups & held:
        raise ValueError("assistance overlaps synthetic development")
    if not demo_groups <= set(manifest["excluded_groups"]) or not demo_groups <= set(
        manifest["exposed_groups"]
    ):
        raise ValueError("review must freeze all observed demonstration exclusion/exposure groups")
    return result
