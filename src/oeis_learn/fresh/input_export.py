"""Export bounded learner views from independently verified admission packets."""

from .common import digest, inventory, write


def collect(archive, keys, training, development):
    from oeis_learn.data.structural_splits import split_for_structure
    from oeis_learn.experiments.artifacts import compute_canonical_digest

    for key in keys:
        packet = archive.get(key)
        entry = packet["entry"]
        item = {
            "program_id": key,
            "values": entry["conditioning_values"],
            "tokens": packet["tokens"],
            "structure": entry["ir_sha256"],
        }
        if packet.get("provenance", {}).get("config", {}).get("profile") == "connected_sampler_v1":
            item["stage"] = packet["provenance"]["sample"]["statistics"]["stage"]
        if entry["training_eligible"]:
            identity = digest(item["values"])
            old = training.get(identity)
            if old is None or (len(item["tokens"]), key) < (len(old["tokens"]), old["program_id"]):
                training[identity] = item
        elif (
            split_for_structure(entry["ir_sha256"], archive.holdouts.seed) == "development"
            and entry["context_tokens"] <= 1024
            and compute_canonical_digest(item["values"]) not in archive.holdouts.reserved_prefixes
        ):
            development[entry["ir_sha256"]] = item


def validation_view(development, count):
    rows = sorted(development.values(), key=lambda r: r["program_id"])
    if not any("stage" in r for r in rows):
        return rows[:count]
    # Round-robin stages so a small early stage cannot disappear in truncation.
    groups = [[r for r in rows if r["stage"] == stage] for stage in range(4)]
    return [
        row
        for index in range(max(map(len, groups), default=0))
        for group in groups
        for row in group[index : index + 1]
    ][:count]


def export(root, config, imported, built, archive, training, development, *, completion=None):
    from oeis_learn.evaluation.release_source import audit_release, prepare_release

    release = root / "release"
    if not release.exists():
        prepare_release(root / "indexed-source/manifest.json", root / "cohort.yaml", release)
    audit = audit_release(release / "cohort", archive=archive, snapshot=archive.freeze())
    write(root / "release-source-audit.json", audit, immutable=True)
    # Preserve all admitted evidence, but bound the actual validation workload.
    write(
        root / "examples.json",
        {
            "training": sorted(training.values(), key=lambda x: x["program_id"]),
            "development": validation_view(
                development, config["archive"]["development_structures"]
            ),
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
    if completion is not None:
        write(root / "admission-completion.json", completion, immutable=True)
    manifest = {
        "version": "fresh-inputs/v1",
        "config": config,
        "files": inventory(root, exclude=("execution", "decisions", "archive-budget", "release")),
        "import_report": imported,
        "archive_snapshot": archive.freeze(),
        "qualified": False,
        "purpose": "fresh training diagnostic; historical and release gates remain independent",
    }
    if completion is not None:
        manifest["admission_completion"] = completion
    manifest["identity"] = digest(manifest)
    write(root / "prepared.json", manifest, immutable=True)
    return manifest
