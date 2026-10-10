"""Explicit indexed source import and release split audit; no inferred OEIS metadata."""

from pathlib import Path
from oeis_learn.experiments.artifacts import (
    ArtifactPath,
    load_json,
    compute_file_hash,
    compute_canonical_digest as digest,
)
from oeis_learn.evaluation.foundation_cohort import (
    write_json,
    freeze_cohort,
    load_cohort,
    build_cohort,
)
from oeis_learn.experiments.models import INTEGER_TEXT_RE


def read_indexed_source(manifest):
    """Manifest lists hashed b-files and hashed authoritative metadata per source record.

    Caller supplies local release inputs. Stripped/names fixtures have no reliable
    offsets, indices or keywords and cannot be promoted by inventing them here.
    """
    path, root = Path(manifest), Path(manifest).parent
    inputs = load_json(path.read_bytes())
    if set(inputs) != {"version", "release", "records"} or inputs["version"] != "indexed-source/v1":
        raise ValueError("explicit indexed release manifest required")
    if not isinstance(inputs["release"], str) or not inputs["release"]:
        raise ValueError("source release locator required")
    rows = []
    for record in inputs["records"]:
        if set(record) != {"record_id", "terms", "metadata"}:
            raise ValueError("source record must bind terms and metadata")
        blobs = {}
        for name in ("terms", "metadata"):
            ref = record[name]
            if set(ref) != {"path", "sha256"}:
                raise ValueError("hashed source reference required")
            file = ArtifactPath(root, ref["path"]).as_path()
            if compute_file_hash(file) != ref["sha256"]:
                raise ValueError("source file digest mismatch")
            blobs[name] = file.read_text()
        meta = load_json(blobs["metadata"])
        if set(meta) != {"offset", "keywords", "finite", "table", "locator"}:
            raise ValueError("authoritative offset/keywords/finite/table metadata required")
        if (
            type(meta["offset"]) is not int
            or type(meta["finite"]) is not bool
            or type(meta["table"]) is not bool
            or not isinstance(meta["keywords"], list)
            or any(not isinstance(k, str) for k in meta["keywords"])
            or not isinstance(meta["locator"], str)
            or not meta["locator"]
        ):
            raise ValueError("invalid authoritative metadata")
        indices, values = [], []
        for line in blobs["terms"].splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if (
                len(parts) != 2
                or not INTEGER_TEXT_RE.fullmatch(parts[0])
                or not INTEGER_TEXT_RE.fullmatch(parts[1])
                or len(parts[1]) > 4096
            ):
                raise ValueError("malformed indexed term; no index compression")
            indices.append(int(parts[0]))
            values.append(parts[1])
        rows.append(
            dict(
                record_id=record["record_id"],
                first_index=meta["offset"],
                indices=indices,
                values=values,
                metadata=dict(
                    meta,
                    source_keywords=meta["keywords"],
                    source_manifest_sha256=compute_file_hash(path),
                    source_files=record,
                ),
            )
        )
    if not rows:
        raise ValueError("empty indexed release source")
    report = dict(
        version="indexed-source-import/v1",
        release=inputs["release"],
        manifest_sha256=compute_file_hash(path),
        records_sha256=digest(rows),
        count=len(rows),
        authoritative_metadata=True,
        inferred_metadata=False,
    )
    return rows, report


def prepare_indexed_source(manifest, output):
    rows, report = read_indexed_source(manifest)
    write_json(Path(output), "records.json", rows)
    write_json(Path(output), "source-evidence.json", report)
    return report


def audit_release(cohort_root, *, archive=None, snapshot=None):
    """Replay affine grouping and all-member reserved prefixes before release use."""
    from oeis_learn.data.structural_splits import structural_report
    from oeis_learn.evaluation.foundation_cohort import read_ref

    cohort = load_cohort(cohort_root)
    if cohort["profile"] != "prefix20_total100_affine_v2":
        raise ValueError("release requires exact affine/shift grouping")
    frozen = read_ref(cohort_root, cohort["source_records"])
    rows = list(frozen["records"].values())
    # Reopen the exact hashed authoritative files, then replay the import and
    # exclusions. Metadata assertions alone are never source evidence.
    source_manifest = Path(cohort_root) / "source-inputs/manifest.json"
    imported, source_report = read_indexed_source(source_manifest)
    imported_built = build_cohort(
        imported,
        affine=True,
        **{k: cohort["config"][k] for k in ("seed", "dev_count", "final_count")},
    )
    if (
        imported_built["records"] != frozen["records"]
        or imported_built["exclusions"] != frozen["exclusions"]
    ):
        raise ValueError("authoritative source import replay mismatch")
    rebuilt = build_cohort(
        rows, affine=True, **{k: cohort["config"][k] for k in ("seed", "dev_count", "final_count")}
    )
    expected = [
        {k: g[k] for k in ("group_id", "members", "representative", "witnesses", "partition")}
        for g in cohort["groups"]
    ]
    if rebuilt["groups"] != expected:
        raise ValueError("release grouping replay mismatch")
    prefixes = set(read_ref(cohort_root, cohort["prefix_membership"]))
    required = {
        digest(frozen["records"][member]["values"][:20])
        for g in cohort["groups"]
        if g["partition"] != "reserved"
        for member in g["members"]
    }
    if prefixes != required:
        raise ValueError("reserved prefix membership mismatch")
    structures = None
    if archive is not None:
        if not required <= archive.holdouts.reserved_prefixes:
            raise ValueError("archive admission omitted release prefixes")
        view = archive.training_view(snapshot)
        if not view:
            raise ValueError("release training archive is empty")
        structures = structural_report([v["source"] for _, v in view], archive.holdouts.seed)
        if any(g["split"] != "train" for g in structures["groups"].values()):
            raise ValueError("development/final structure leaked into training archive")
    report = dict(
        version="release-source-audit/v1",
        cohort_sha256=cohort["cohort_id"],
        grouping_replayed=True,
        metadata_verified=True,
        source_manifest_sha256=source_report["manifest_sha256"],
        all_member_prefixes=len(required),
        structural_split=structures,
        status="PASS" if structures is not None else "pending",
        missing=[]
        if structures is not None
        else ["frozen admitted training archive structural audit"],
    )
    report["report_sha256"] = digest(report)
    return report


def prepare_release(manifest, config, output):
    root = Path(output)
    prepare_indexed_source(manifest, root / "source")
    freeze_cohort(root / "source", Path(config), root / "cohort")
    from oeis_learn.tracking.role_coordinator import copy_view

    inputs = load_json(Path(manifest).read_bytes())
    files = {"manifest.json": Path(manifest)}
    for record in inputs["records"]:
        for name in ("terms", "metadata"):
            ref = record[name]
            file = ArtifactPath(Path(manifest).parent, ref["path"]).as_path()
            if ref["path"] == "manifest.json":
                raise ValueError("source file conflicts with reserved manifest path")
            files[ref["path"]] = file
    copy_view(root / "cohort/source-inputs", files)
    report = audit_release(root / "cohort")
    write_json(root, "release-source-audit.json", report)
    return report
