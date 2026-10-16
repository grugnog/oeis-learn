"""Prepare a measurement capability bundle with development-only evaluator data."""

from pathlib import Path
from oeis_learn.experiments.artifacts import (
    ArtifactPath,
    atomic_write,
    load_json,
    compute_file_hash,
    compute_canonical_digest as digest,
)
from oeis_learn.evaluation.foundation_cohort import load_cohort, read_ref
from oeis_learn.evaluation.experiment_runner import implementation_identity
from oeis_learn.experiments.decisions import frozen_protocol
from oeis_learn.experiments.pilot_models import pilot_profile


def prepare_measurement(
    archive, cohort, g7, g8, evidence_root, image, output, *, baseline_report, baseline_checkpoint
):
    """Preparation replays software gates. Execution additionally requires real hardware gates."""
    from oeis_learn.evaluation.repair_gate import require_g7
    from oeis_learn.evaluation.learning_gate import require_g8
    from oeis_learn.evaluation.release_source import audit_release
    from oeis_learn.evaluation.foundation_cohort import write_json
    from oeis_learn.data.structural_splits import structural_signature
    import re

    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("resolved local image identity required")
    g8_path = g8
    g7, g8 = load_json(Path(g7).read_bytes()), load_json(Path(g8).read_bytes())
    require_g7(g7, evidence_root=evidence_root)
    require_g8(g8, evidence_root=Path(g8_path).parent, diagnostic=True)
    from oeis_learn.evaluation.baseline_gate import require_baseline
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint

    baseline = load_json(Path(baseline_report).read_bytes())
    checkpoint = load_foundation_checkpoint(baseline_checkpoint, device="cpu", include_payload=True)
    require_baseline(baseline, checkpoint, evidence_root=Path(baseline_report).parent)
    containment = read_ref(Path(baseline_report).parent, baseline["checks"]["G2"]["containment"])
    if containment["image"] != image:
        raise ValueError("baseline and measured image differ")
    snapshot = archive.freeze()
    audit = audit_release(cohort, archive=archive, snapshot=snapshot)
    if audit["status"] != "PASS":
        raise ValueError("real release source and structural split audit required")
    rows = [
        dict(key=key, **value)
        for key, value in archive.training_view(snapshot)
        if value["entry"]["origin"] == "generic_sample"
    ]
    # One generic prompt per output group; source diversity is retained separately.
    distinct = {}
    for row in rows:
        distinct.setdefault(digest(row["entry"]["conditioning_values"]), row)
    rows = list(distinct.values())[:128]
    if len(rows) != 128:
        raise ValueError(f"declared engineering sample shortfall: {len(rows)}/128")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    assets = {}

    def save(name, value):
        ref = write_json(root, "assets/" + name + ".json", value)
        assets[name] = ref

    training = [
        dict(key=key, **value)
        for key, value in archive.training_view(snapshot)
        if value["entry"]["origin"] == "generic_sample"
    ][:10000]
    from oeis_learn.data.structural_splits import split_for_structure

    validation = [
        dict(key=key, **archive.get(key))
        for key in snapshot["entries"]
        if archive.get(key)["entry"]["origin"] == "generic_sample"
        and split_for_structure(archive.get(key)["entry"]["ir_sha256"], archive.holdouts.seed)
        == "development"
    ][:256]
    save(
        "examples",
        dict(
            version="measurement-examples/v1",
            snapshot=snapshot,
            rows=rows,
            training_rows=training,
            validation_rows=validation,
            holdouts=dict(
                seed=archive.holdouts.seed,
                reserved_structures=sorted(archive.holdouts.reserved_structures),
                reserved_compositions=[sorted(s) for s in archive.holdouts.reserved_compositions],
                reserved_parameter_bits=archive.holdouts.reserved_parameter_bits,
                reserved_prefixes=sorted(archive.holdouts.reserved_prefixes),
            ),
            structures=len({structural_signature(row["source"]) for row in training}),
        ),
    )
    data = load_cohort(cohort)
    development = []
    for group in data["groups"]:
        if group["partition"] == "development":
            development.append(
                dict(group_id=group["group_id"], values=read_ref(cohort, group["truth"])["values"])
            )
    if not development:
        raise ValueError("development evaluator denominator empty")
    save(
        "development",
        dict(
            version="development-only/v1",
            cohort_sha256=data["cohort_id"],
            groups=development,
            source_audit=audit,
        ),
    )
    save("model_profile", pilot_profile())
    save("baseline", baseline)
    save("baseline_checkpoint", {"name": Path(baseline_checkpoint).name})
    save("gate7", g7)
    save("gate8", g8)
    # Gate artifact references are copied exactly; no repository/truth mount enters roles.
    evidence = []
    for label, report, original in (
        ("g7", g7, Path(evidence_root)),
        ("g8", g8, Path(g8_path).parent),
    ):
        from oeis_learn.evaluation.learning_gate import evidence_references

        refs = evidence_references(report) if label == "g8" else list(report["checks"].values())
        for ref in refs:
            file = ArtifactPath(original, ref["path"]).as_path()
            name = "assets/gates/" + label + "/" + ref["path"]
            target = ArtifactPath(root, name)
            atomic_write(target, file.read_bytes())
            evidence.append(dict(path=name, sha256=compute_file_hash(target.as_path())))
    baseline_root = Path(baseline_report).parent
    refs = [r for check in baseline["checks"].values() for r in check.values()]
    for ref in refs:
        file = ArtifactPath(baseline_root, ref["path"]).as_path()
        target = ArtifactPath(root, "assets/gates/baseline/" + ref["path"])
        atomic_write(target, file.read_bytes())
        evidence.append(
            dict(
                path="assets/gates/baseline/" + ref["path"],
                sha256=compute_file_hash(target.as_path()),
            )
        )
    from oeis_learn.tracking.training_checkpoint import SIDECARS

    checkpoint_path = Path(baseline_checkpoint)
    names = {checkpoint_path.name, checkpoint_path.with_suffix(".json").name, "config.yaml"} | {
        name + ".json" for name in SIDECARS
    }
    for name in names:
        file = checkpoint_path.parent / name
        target = ArtifactPath(root, "assets/gates/baseline/checkpoints/" + name)
        atomic_write(target, file.read_bytes())
        evidence.append(
            dict(
                path="assets/gates/baseline/checkpoints/" + name,
                sha256=compute_file_hash(target.as_path()),
            )
        )
    save("gate_evidence", evidence)
    assets["image"] = image
    sources = [row["source"] for row in rows]
    manifest = dict(
        version="measurement/v2",
        mode="qualified_device",
        protocol=frozen_protocol(),
        identities=dict(
            code=implementation_identity(),
            runtime=digest([image, g7["runtime"], g8["runtime"]]),
            data=digest(sources),
            model=digest(pilot_profile()),
        ),
        sources=sources,
        assets=assets,
    )
    write_json(root, "manifest-input.json", manifest)
    return manifest


def load_assets(manifest, root, *, hardware=False, replay_gates=True):
    from oeis_learn.evaluation.repair_gate import require_g7
    from oeis_learn.evaluation.learning_gate import require_g8
    from oeis_learn.data.sampler_factory import sampler as GenericSampler
    from oeis_learn.data.program_admission import require_sample

    root = Path(root)
    assets = manifest["assets"]
    if set(assets) != {
        "examples",
        "development",
        "model_profile",
        "gate7",
        "gate8",
        "gate_evidence",
        "image",
        "baseline",
        "baseline_checkpoint",
    }:
        raise ValueError("closed measurement capability assets required")
    values = {}
    for name in set(assets) - {"image"}:
        ref = assets[name]
        if set(ref) != {"path", "sha256"} or not ref["path"].startswith("assets/"):
            raise ValueError("confined measurement asset required")
        file = ArtifactPath(root, ref["path"]).as_path()
        if compute_file_hash(file) != ref["sha256"]:
            raise ValueError("measurement asset digest mismatch")
        values[name] = load_json(file.read_bytes())
    for ref in values["gate_evidence"]:
        file = ArtifactPath(root, ref["path"]).as_path()
        if not ref["path"].startswith("assets/gates/") or compute_file_hash(file) != ref["sha256"]:
            raise ValueError("gate evidence asset mismatch")
    if manifest["identities"]["runtime"] != digest(
        [assets["image"], values["gate7"]["runtime"], values["gate8"]["runtime"]]
    ):
        raise ValueError("image/gate runtime binding mismatch")
    if replay_gates:
        from oeis_learn.evaluation.baseline_gate import require_baseline
        from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint

        name = values["baseline_checkpoint"]["name"]
        if Path(name).name != name:
            raise ValueError("confined checkpoint filename required")
        checkpoint = load_foundation_checkpoint(
            root / "assets/gates/baseline/checkpoints" / name, device="cpu", include_payload=True
        )
        require_baseline(
            values["baseline"], checkpoint, evidence_root=root / "assets/gates/baseline"
        )
        require_g7(values["gate7"], evidence_root=root / "assets/gates/g7")
        require_g8(values["gate8"], evidence_root=root / "assets/gates/g8", diagnostic=not hardware)
    examples = values["examples"]
    if (
        set(examples)
        != {
            "version",
            "snapshot",
            "rows",
            "structures",
            "training_rows",
            "validation_rows",
            "holdouts",
        }
        or examples["version"] != "measurement-examples/v1"
    ):
        raise ValueError("immutable generic measurement fixture required")
    if (
        len(examples["rows"]) != 128
        or [r["source"] for r in examples["rows"]] != manifest["sources"]
    ):
        raise ValueError("engineering sources/128 prompts mismatch")
    for row in examples["rows"] + examples["training_rows"] + examples["validation_rows"]:
        packet = {k: row[k] for k in ("entry", "source", "tokens", "execution", "provenance")}
        if digest(packet) != row["key"] or row["key"] not in examples["snapshot"]["entries"]:
            raise ValueError("frozen archive measurement mismatch")
        if row["entry"]["origin"] != "generic_sample":
            raise ValueError("only provenance-valid generic training fixtures allowed")
        provenance = row["provenance"]
        if "structure" in provenance:
            from oeis_learn.data.generic_programs import QualifiedGenericSampler

            draw = QualifiedGenericSampler(provenance["config"]).sample_variant(
                provenance["sample"]["sample_counter"], provenance["structure"]["parameter_counter"]
            )
            if draw != provenance["structure"]:
                raise ValueError("structural draw mismatch")
        else:
            require_sample(provenance["sample"], GenericSampler(provenance["config"]))
        if provenance["sample"]["canonical_source"] != row["source"]:
            raise ValueError("generic fixture source mismatch")
    from oeis_learn.data.structural_splits import StructuralHoldouts, split_for_structure

    h = examples["holdouts"]
    holdouts = StructuralHoldouts(
        seed=h["seed"],
        reserved_structures=frozenset(h["reserved_structures"]),
        reserved_compositions=tuple(frozenset(s) for s in h["reserved_compositions"]),
        reserved_parameter_bits=h["reserved_parameter_bits"],
        reserved_prefixes=frozenset(h["reserved_prefixes"]),
    )
    from oeis_learn.discovery.program_archive import ProgramArchive

    verifier = object.__new__(ProgramArchive)
    verifier.holdouts, verifier.context_limit = holdouts, 1024
    for row in examples["rows"] + examples["training_rows"] + examples["validation_rows"]:
        verifier.validate_packet(
            {k: row[k] for k in ("entry", "source", "tokens", "execution", "provenance")},
            row["key"],
        )
    for row in examples["training_rows"]:
        if not holdouts.decision(
            row["source"], [int(v) for v in row["entry"]["conditioning_values"]]
        )[0]:
            raise ValueError("training fixture leaks a holdout")
    for row in examples["validation_rows"]:
        if split_for_structure(row["entry"]["ir_sha256"], holdouts.seed) != "development":
            raise ValueError("synthetic validation split mismatch")
    if set(r["entry"]["ir_sha256"] for r in examples["training_rows"]) & set(
        r["entry"]["ir_sha256"] for r in examples["validation_rows"]
    ):
        raise ValueError("synthetic structural split overlap")
    dev = values["development"]
    if (
        set(dev) != {"version", "cohort_sha256", "groups", "source_audit"}
        or dev["version"] != "development-only/v1"
    ):
        raise ValueError("development-only evaluator capability required; final truth forbidden")
    if not dev["groups"] or any(
        set(g) != {"group_id", "values"} or len(g["values"]) != 100 for g in dev["groups"]
    ):
        raise ValueError("complete fixed development denominator required")
    if len({g["group_id"] for g in dev["groups"]}) != len(dev["groups"]):
        raise ValueError("duplicate development group")
    if values["model_profile"] != pilot_profile() or manifest["identities"]["model"] != digest(
        pilot_profile()
    ):
        raise ValueError("baseline model profile mismatch")
    return values
