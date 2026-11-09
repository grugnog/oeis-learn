"""Typed filesystem projection with inode checks; a hash never grants permission."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from oeis_learn.pipeline_validation.identity import confined

from .contracts import load, reference, verify_refs

# Types, rather than filename conventions, determine role permission.
CHANNELS = {
    "sources": {
        "source",
        "groups",
        "exposure",
        "cohort",
        "coverage_witness",
        "generation_policy",
        "reference_tool",
    },
    "training": {
        "training_request",
        "mechanical_shard",
        "admitted_target",
        "reward_probe",
        "full_state",
        "library",
    },
    "visible": {"visible_prompt"},
    "model": {"checkpoint", "recipe", "library", "oeis_generation"},
    "candidates": {"candidate"},
    "admission_truth": {"training_continuation"},
    "score_truth": {"development_truth", "confirmation_truth"},
    "assessment_truth": {"assessment_truth"},
    "selection": {"locked_selection"},
    "search": {"search_space", "development_scalar"},
    "archive": {
        "verified_training_archive",
        "typed_training_source",
        "component_policy",
    },
    "dossier": {"sealed_dossier", "review_evidence"},
    "public": {"public_evidence"},
}


@dataclass(frozen=True)
class Artifact:
    """Custodian-bound descriptor; envelope/contents both checked before copying."""

    kind: str
    role: str
    ref: dict
    identity: str


def describe(root, path):
    value = load(confined(root, path), maximum=512 << 20)
    if (
        set(value) != {"version", "identity", "kind", "role", "data"}
        or value["version"] != "discovery-channel/v1"
    ):
        raise ValueError("typed channel envelope required")
    from oeis_learn.pipeline_validation.identity import seal

    if seal(value) != value:
        raise ValueError("channel identity differs")
    return Artifact(value["kind"], value["role"], reference(root, path), value["identity"])


def project(root, destination, operation, artifacts, *, registry, forbidden=(), authorized=None):
    """Build exact separate channel roots. Partial projections are removed on failure."""
    if operation not in registry:
        raise ValueError("unregistered operation")
    spec = registry[operation]
    if set(artifacts) != set(spec.channels):
        raise ValueError("exact operation channels required")
    destination = confined(root, destination, exists=False)
    if destination.exists():
        raise ValueError("fresh projection required")
    all_artifacts = [a for channel in artifacts.values() for a in channel]
    refs = [a.ref for a in all_artifacts]
    nested = []
    archive_children = {}
    for descriptor in all_artifacts:
        if descriptor.kind in {
            "checkpoint",
            "recipe",
            "training_request",
            "verified_training_archive",
            "typed_training_source",
            "library",
        }:
            data = load(confined(root, descriptor.ref["path"]))["data"]
            _validate_channel(descriptor.kind, data)
            if descriptor.kind == "training_request" and data["artifact"]["bytes"] > 256 << 20:
                raise ValueError("bounded training package required")
            nested.append(data["artifact"])
            if descriptor.kind == "verified_training_archive":
                from .contracts import validate

                cut = validate("archive", load(confined(root, data["artifact"]["path"])))
                base = confined(root, data["artifact"]["path"]).parent
                archive_children[descriptor.identity] = []
                for child in cut["payload"]["entries"]:
                    candidate = reference(root, confined(base, child["path"]))
                    if any(candidate[k] != child[k] for k in ("bytes", "sha256")):
                        raise ValueError("original archive source bytes differ")
                    nested.append(candidate)
                    archive_children[descriptor.identity].append(candidate)
            if descriptor.kind == "training_request":
                package = load(confined(root, data["artifact"]["path"]), maximum=256 << 20)
                if package.get("weights") is not None:
                    nested.append(package["weights"])
                if package.get("origin") is not None:
                    from .campaign import initializer_origin

                    _, closure = initializer_origin(
                        root,
                        dict(
                            mode="weights_only",
                            artifact=package["weights"],
                            origin=package["origin"],
                        ),
                        track="synthetic_strict",
                        purpose=load(
                            confined(
                                root,
                                load(confined(root, package["origin"]["path"]))["campaign"]["path"],
                            )
                        )["purpose"],
                    )
                    nested.extend(closure)
    verify_refs(root, refs)
    # Metadata is not a substitute for verifying the actual mounted weights.
    unique_nested = {r["path"]: r for r in nested}
    if any(unique_nested[r["path"]] != r for r in nested):
        raise ValueError("conflicting nested model artifact")
    verify_refs(root, list(unique_nested.values()), max_bytes=64 << 30)
    if authorized is not None:
        for descriptor in all_artifacts:
            if descriptor.ref not in authorized:
                raise ValueError("artifact is not permitted by frozen role authority")
    excluded = set()
    for ref in forbidden:
        verify_refs(root, [ref])
        info = confined(root, ref["path"]).stat()
        excluded.add((info.st_dev, info.st_ino))
    for ref in nested:
        info = confined(root, ref["path"]).stat()
        if (info.st_dev, info.st_ino) in excluded:
            raise ValueError("nested model artifact aliases private truth")
    for channel, items in artifacts.items():
        for descriptor in items:
            source = confined(root, descriptor.ref["path"])
            info = source.stat()
            if (info.st_dev, info.st_ino) in excluded:
                raise ValueError("same-file truth leakage")
            bound = describe(root, descriptor.ref["path"])
            if (
                bound != descriptor
                or descriptor.kind not in CHANNELS[channel]
                or descriptor.role not in {spec.role, "public"}
            ):
                raise ValueError("wrong role/type/identity")
            _validate_channel(descriptor.kind, load(source)["data"])
            if (
                source == destination
                or source in destination.parents
                or destination in source.parents
            ):
                raise ValueError("overlapping input/output")
    from .storage import preflight, copy_file

    preflight(
        sum(r["bytes"] for r in refs) + sum(r["bytes"] for r in unique_nested.values()),
        sum(len(Path(r["path"]).parts) + 1 for r in [*refs, *unique_nested.values()])
        + len(artifacts)
        + 1,
    )
    destination.mkdir(parents=True, exist_ok=False)
    try:
        result = {}
        for channel, items in artifacts.items():
            folder = destination / channel
            folder.mkdir()
            result[channel] = folder
            for item in items:
                # Flatten to digest filename; never preserve a private host path.
                target = folder / (item.identity.removeprefix("sha256:") + ".json")
                copy_file(confined(root, item.ref["path"]), target)
                target.chmod(0o444)
                if item.kind in {
                    "checkpoint",
                    "recipe",
                    "training_request",
                    "verified_training_archive",
                    "typed_training_source",
                    "library",
                }:
                    ref = load(confined(root, item.ref["path"]))["data"]["artifact"]
                    raw = confined(folder, ref["path"], exists=False)
                    if raw.exists():
                        if item.kind not in {"verified_training_archive", "typed_training_source"}:
                            raise ValueError("duplicate projected payload path")
                        verify_refs(folder, [ref], max_bytes=256 << 20)
                    else:
                        raw.parent.mkdir(parents=True, exist_ok=True)
                        copy_file(confined(root, ref["path"]), raw)
                        raw.chmod(0o444)
                    if item.kind == "verified_training_archive":
                        for child in archive_children[item.identity]:
                            copied = confined(folder, child["path"], exists=False)
                            if copied.exists():
                                verify_refs(folder, [child], max_bytes=256 << 20)
                            else:
                                copied.parent.mkdir(parents=True, exist_ok=True)
                                copy_file(confined(root, child["path"]), copied)
                                copied.chmod(0o444)
                    if item.kind == "training_request":
                        package = load(confined(root, ref["path"]), maximum=256 << 20)
                        if package.get("weights") is not None:
                            weights = package["weights"]
                            target_weights = confined(folder, weights["path"], exists=False)
                            if target_weights.exists():
                                raise ValueError("duplicate weights-only projection path")
                            target_weights.parent.mkdir(parents=True, exist_ok=True)
                            copy_file(confined(root, weights["path"]), target_weights)
                            target_weights.chmod(0o444)
                        if package.get("origin") is not None:
                            for child in unique_nested.values():
                                destination_child = confined(folder, child["path"], exists=False)
                                if destination_child.exists():
                                    verify_refs(folder, [child], max_bytes=64 << 30)
                                    continue
                                destination_child.parent.mkdir(parents=True, exist_ok=True)
                                copy_file(confined(root, child["path"]), destination_child)
                                destination_child.chmod(0o444)
            for child in folder.rglob("*"):
                if child.is_dir():
                    child.chmod(0o555)
            folder.chmod(0o555)
        return result
    except BaseException:
        remove_projection(destination)
        raise


def _validate_channel(kind, data):
    if kind == "component_policy":
        from .component_mining import validate_policy
        from .contracts import _check, schema

        if not isinstance(data, dict) or set(data) - {"generation_policies"} != {
            "policy",
            "previous",
        }:
            raise ValueError("closed component publication request required")
        validate_policy(data["policy"])
        policies = data.get("generation_policies", {})
        if not isinstance(policies, dict) or len(policies) > 64:
            raise ValueError("bounded frozen mechanical source policies required")
        from .generator import validate_policy as validate_generation
        from .contracts import digest

        for identity, law in policies.items():
            if identity != digest(validate_generation(law)):
                raise ValueError("mechanical source law differs from its identity")
        if data["previous"] is not None:
            _check(schema("common")["$defs"]["hash"], data["previous"], schema("common"))
    if kind == "oeis_generation":
        if (
            set(data) != {"methods", "forced"}
            or data["methods"] != ["correct"]
            or (data["forced"] is not None and not isinstance(data["forced"], dict))
        ):
            raise ValueError("closed one-portfolio OEIS generation policy required")
    if kind in {"cohort", "exposure"}:
        from .contracts import validate

        validate(kind, data)
    if kind == "coverage_witness" and (
        not isinstance(data, dict)
        or set(data)
        != {
            "task",
            "source",
            "indices",
            "values",
            "operator_class",
            "missing_operations",
            "created_at",
        }
    ):
        raise ValueError("closed coverage witness")
    if kind == "visible_prompt":
        from .cohorts import validate_prompt

        validate_prompt(data)
    if kind == "development_scalar":
        if not isinstance(data, dict) or set(data) != {"trial", "value", "coordinate"}:
            raise ValueError("scalar history only")
        from oeis_learn.pipeline_validation.identity import number

        number(data["value"])
        number(data["coordinate"])
    if kind == "training_request":
        if (
            not isinstance(data, dict)
            or set(data) != {"membership", "artifact", "compatibility", "profile"}
            or data["membership"] != "training"
            or data["profile"] != "bigint-body/v1"
        ):
            raise ValueError("closed learner-only training request required")
        from .contracts import _check, schema

        _check(schema("common")["$defs"]["artifact"], data["artifact"], schema("common"))
        _check(schema("common")["$defs"]["hash"], data["compatibility"], schema("common"))
    if kind in {"checkpoint", "recipe", "library", "full_state"}:
        if not isinstance(data, dict) or not {"artifact", "compatibility", "profile"} <= set(
            data
        ) <= {"artifact", "compatibility", "profile", "control"}:
            raise ValueError("closed model/state artifact metadata required")
        if "control" in data and (
            kind != "checkpoint"
            or data["control"] not in {"retained_D", "earlier_checkpoint", "random"}
        ):
            raise ValueError("registered checkpoint control required")
        from .contracts import _check, schema

        _check(schema("common")["$defs"]["artifact"], data["artifact"], schema("common"))
        _check(schema("common")["$defs"]["hash"], data["compatibility"], schema("common"))
        if data["profile"] != (
            "progressive-component-search/v1" if kind == "library" else "bigint-body/v1"
        ):
            raise ValueError("unregistered model language")
    if kind in {
        "mechanical_shard",
        "admitted_target",
        "reward_probe",
        "verified_training_archive",
        "typed_training_source",
    }:
        if (
            not isinstance(data, dict)
            or set(data) != {"membership", "artifact", "profile", "compatibility"}
            or data["membership"] != "training"
        ):
            raise ValueError("closed training membership metadata required")
        from .contracts import _check, schema

        _check(schema("common")["$defs"]["artifact"], data["artifact"], schema("common"))
        _check(schema("common")["$defs"]["hash"], data["compatibility"], schema("common"))
        if data["profile"] != "bigint-body/v1":
            raise ValueError("unregistered training profile")


def remove_projection(path):
    path = Path(path)
    if path.exists():
        for folder in path.rglob("*"):
            if folder.is_dir():
                folder.chmod(0o755)
        path.chmod(0o755)
        shutil.rmtree(path)
