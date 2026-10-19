"""Monotone assistance lineage and observed overlap; no data import side effects."""

from oeis_learn.synthesis.contracts import identity
from oeis_learn.synthesis.state import require_hash
import math


def _cost(cost):
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ValueError("recorded finite nonnegative compute required")


def validate_assistance(manifest):
    fields = {
        "version",
        "review",
        "sources",
        "demonstrations",
        "recipients",
        "excluded_groups",
        "exposed_groups",
        "historical_approval",
    }
    if (
        set(manifest) != fields
        or manifest["version"] != "assistance/v1"
        or tuple(manifest["recipients"]) != ("assisted",)
    ):
        raise ValueError("reviewed assisted-only manifest required")
    require_hash(manifest["review"])
    if not manifest["sources"] or not manifest["demonstrations"]:
        raise ValueError("source/license and verified demonstrations required")
    for source, license, evidence in manifest["sources"]:
        if not source or not license:
            raise ValueError("missing source/license")
        require_hash(evidence)
    for demo, verification in manifest["demonstrations"]:
        require_hash(demo)
        require_hash(verification)
    for h in (*manifest["excluded_groups"], *manifest["exposed_groups"]):
        require_hash(h)
    if manifest["historical_approval"] is not None:
        require_hash(manifest["historical_approval"])
    return identity(manifest)


def _seal(record):
    return dict(record, sha256=identity(record))


def validate_lineage(record):
    fields = {
        "track",
        "root",
        "foundation",
        "foundation_cost",
        "incremental_cost",
        "parent",
        "assistance",
        "exposed_groups",
        "excluded_groups",
        "origin",
        "sha256",
    }
    if set(record) != fields or record["sha256"] != identity(
        {k: v for k, v in record.items() if k != "sha256"}
    ):
        raise ValueError("lineage identity mismatch")
    if record["track"] not in {"generic_only", "assisted"}:
        raise ValueError("unknown track")
    for k in ("root", "foundation"):
        require_hash(record[k])
    if record["parent"] is not None:
        require_hash(record["parent"])
    for h in (*record["assistance"], *record["exposed_groups"], *record["excluded_groups"]):
        require_hash(h)
    if record["track"] == "generic_only" and (
        record["assistance"] or record["origin"] not in {"random_generic", "generic_training"}
    ):
        raise ValueError("strict lineage contamination")
    _cost(record["foundation_cost"])
    _cost(record["incremental_cost"])
    return record


def root_lineage(*, seed, cost, checkpoint_sha256, recipe_sha256):
    _cost(cost)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random initialization seed required")
    require_hash(checkpoint_sha256)
    require_hash(recipe_sha256)
    root = identity(["random-generic/v1", seed, recipe_sha256])
    foundation = checkpoint_sha256
    return _seal(
        dict(
            track="generic_only",
            root=root,
            foundation=foundation,
            foundation_cost=cost,
            incremental_cost=0,
            parent=None,
            assistance=(),
            exposed_groups=(),
            excluded_groups=(),
            origin="random_generic",
        )
    )


def descendant(parent, *, track, cost, manifest=None, origin="generic_training"):
    validate_lineage(parent)
    _cost(cost)
    if origin not in {
        "generic_training",
        "named_demonstration",
        "historical_checkpoint",
        "legacy_replay",
    }:
        raise ValueError("unknown source origin")
    assisted = parent["track"] == "assisted" or manifest is not None or origin != "generic_training"
    if track not in {"generic_only", "assisted"} or assisted and track != "assisted":
        raise ValueError("assistance is monotone; no strict recipient")
    if (
        origin != "generic_training" or track == "assisted" and parent["track"] != "assisted"
    ) and manifest is None:
        raise ValueError("explicit reviewed assistance required")
    assistance = set(parent["assistance"])
    exposed = set(parent["exposed_groups"])
    excluded = set(parent["excluded_groups"])
    if manifest is not None:
        assistance.add(validate_assistance(manifest))
        exposed.update(manifest["exposed_groups"])
        excluded.update(manifest["excluded_groups"])
        if (
            origin in {"historical_checkpoint", "legacy_replay"}
            and not manifest["historical_approval"]
        ):
            raise ValueError("historical migration approval required")
    return _seal(
        dict(
            track=track,
            root=parent["root"],
            foundation=parent["foundation"],
            foundation_cost=parent["foundation_cost"],
            incremental_cost=parent["incremental_cost"] + cost,
            parent=parent["sha256"],
            assistance=tuple(sorted(assistance)),
            exposed_groups=tuple(sorted(exposed)),
            excluded_groups=tuple(sorted(excluded)),
            origin=origin,
        )
    )


def fresh_cohort(groups, lineage):
    validate_lineage(lineage)
    if set(groups) & (set(lineage["exposed_groups"]) | set(lineage["excluded_groups"])):
        raise ValueError("overlap/exposed groups require diagnostic status or new cohort")
    for h in groups:
        require_hash(h)
    return identity(sorted(groups))


def validate_ancestry(head, records):
    """Require complete parent records; a relabelled checkpoint is not a new root."""
    seen = set()
    node = head
    while True:
        validate_lineage(node)
        if node["sha256"] in seen:
            raise ValueError("cyclic ancestry")
        seen.add(node["sha256"])
        if node["parent"] is None:
            if (
                node["origin"] != "random_generic"
                or node["track"] != "generic_only"
                or node["assistance"]
                or node["incremental_cost"] != 0
            ):
                raise ValueError("unreviewed/nonrandom strict root")
            return seen
        parent = records.get(node["parent"])
        if parent is None:
            raise ValueError("incomplete ancestry")
        validate_lineage(parent)
        if (
            parent["sha256"] != node["parent"]
            or parent["track"] == "assisted"
            and node["track"] != "assisted"
            or any(node[k] != parent[k] for k in ("root", "foundation", "foundation_cost"))
            or node["incremental_cost"] < parent["incremental_cost"]
            or any(
                not set(parent[k]) <= set(node[k])
                for k in ("assistance", "exposed_groups", "excluded_groups")
            )
        ):
            raise ValueError("ancestry contamination or erased exposure/cost")
        node = parent
