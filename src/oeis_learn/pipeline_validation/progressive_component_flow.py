"""Owned component outputs enter learning only at its frozen task boundary."""

import json

from oeis_learn.generalization.progressive.component_search import validate_input
from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    seal,
    verify_refs,
)


def library_input(root, data):
    if data["profile"] != "progressive-component-search/v1":
        raise ValueError("expanded component search profile required")
    return validate_input(json.loads(verify_refs(root, [data["artifact"]], max_bytes=256 << 20)[0]))


def admission_request(root, values):
    library = library_input(root, next(v["data"] for v in values if v["kind"] == "library"))
    member = next(v["data"] for v in values if v["kind"] == "training_continuation")
    proposal = next(v for v in values if v["kind"] == "component_proposal")
    return [
        proposal,
        {"kind": "component_training_request", "data": {"member": member, "library": library}},
    ]


def training_input(root, package, values, task):
    from oeis_learn.generalization.progressive.component_distillation import (
        RULE,
        freeze_distillation,
    )

    from .progressive_learner import compatibility

    libraries = [v["data"] for v in values if v["kind"] == "library"]
    targets = [
        json.loads(verify_refs(root, [v["data"]["artifact"]], max_bytes=4 << 20)[0])
        for v in values
        if v["kind"] == "admitted_target"
    ]
    if package.get("component_rules") != RULE or "component_distillation" in package:
        raise ValueError("original frozen component update rule required")
    if task["payload"]["arm"] == "no_library":
        if libraries or targets:
            raise ValueError("no-library control cannot consume library targets")
        return seal(
            dict(package, component_distillation=freeze_distillation(None, [], arm="no_library"))
        )
    if task["payload"]["arm"] != "library" or len(libraries) != 1 or not 1 <= len(targets) <= 64:
        raise ValueError("one library and finite expanded target cut required")
    if libraries[0]["compatibility"] != compatibility(package):
        raise ValueError("component learner/library compatibility differs")
    library = library_input(root, libraries[0])
    if (
        library["library"]["purpose"] != task["purpose"]
        or library["library"]["track"] != task["track"]
    ):
        raise ValueError("component learner/library provenance differs")
    if any(t["purpose"] != task["purpose"] for t in targets):
        raise ValueError("component target purpose differs")
    result = seal(
        dict(package, component_distillation=freeze_distillation(library, targets, arm="library"))
    )
    if compatibility(result) != compatibility(package):
        raise ValueError("component cut changed frozen learning policy")
    return result


def generated_proposals(output, candidates, prompts=(), *, deadline=None):
    import time

    from oeis_learn.synthesis.reference import evaluate

    by_id = {p["id"]: p for p in prompts}
    members = {}
    for member, record in candidates["records"].items():
        rows = record.get("costs", {}).get("components", {}).get("proposals", [])
        eligible = []
        for r in rows:
            if r.get("status") != "eligible":
                continue
            prompt = by_id.get(member)
            if prompt is not None:
                bound = (
                    min(time.monotonic() + 4, deadline)
                    if deadline is not None
                    else time.monotonic() + 4
                )
                observed = evaluate(r["source"], prompt["indices"], deadline=bound)
                if observed.outcome != "ok" or list(observed.values) != prompt["values"]:
                    continue
            eligible.append(r)
        if eligible:
            # Visible neural scores only; no continuation truth enters selection.
            chosen = min(
                eligible,
                key=lambda r: (-r["model_score"], r["source"], r["position"], r["variant"]),
            )
            members[member] = {
                k: chosen[k]
                for k in (
                    "source",
                    "expression",
                    "position",
                    "variant",
                    "component",
                    "expanded_nodes",
                    "expanded_bytes",
                    "expanded_tokens",
                    "model_score",
                    "status",
                )
            }
    atomic_write(
        output / "component-proposals.json",
        seal({"version": "progressive-component-proposals/v1", "members": members}),
    )
