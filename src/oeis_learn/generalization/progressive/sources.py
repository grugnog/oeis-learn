"""Exact finite-source metadata and outcome-blind relation grouping."""

from .contracts import _check, schema, seal, digest
from oeis_learn.synthesis.contracts import canonical_integer


def freeze_source(metadata):
    value = seal(dict(version="progressive-source/v1", **metadata))
    return validate_source(value)


def validate_source(value):
    doc = schema("source")
    _check(doc, value, doc)
    if seal(value) != value:
        raise ValueError("source seal differs")
    if (
        value["access"] != "available"
        or not value["use_conditions"].strip()
        or value["offset"] != "0"
        or value["finite_support"] != 9
        or value["representation"] != "exact_integer_totals"
        or value["objects"] != "permutations"
        or value["indexing"] != "collection-size"
        or value["transformation"] not in {"first_moment", "second_moment"}
        or value["code"] is not None
        or value["track"] != "external_assisted"
    ):
        raise ValueError("unsuitable/unresolved source for combinatorial-short/v1")
    for n in value["values"]:
        canonical_integer(n)
    return value


def groups(sources):
    """Union related statistic/map identities and exact finite duplicates across sources."""
    if len(sources) > 4096:
        raise ValueError("source group budget")
    keys = []
    for s in sources:
        validate_source(s)
        keys.append(set(s["relations"]) | {"values:" + digest(s["values"])})
    pools = []
    for i, k in enumerate(keys):
        matches = [p for p in pools if p[1] & k]
        members = {str(i)}
        for p in matches:
            members |= p[0]
            k |= p[1]
            pools.remove(p)
        pools.append((members, k))
    return sorted([sorted(p[0], key=int) for p in pools], key=lambda p: int(p[0]))


def pilot_support(sources, roles):
    if len(sources) != len(roles) or any(r not in {"training", "evaluation"} for r in roles):
        raise ValueError("frozen source-role roster required")
    grouped = groups(sources)
    counts = {"training": 0, "evaluation": 0}
    for g in grouped:
        membership = {roles[int(i)] for i in g}
        if len(membership) != 1:
            raise ValueError("related sources cross roles")
        if len(g) >= 2:
            counts[next(iter(membership))] += 1
    deficits = {r: max(0, 4 - n) for r, n in counts.items()}
    return seal(
        dict(
            version="progressive-source-feasibility/v1",
            groups=counts,
            deficits=deficits,
            status="ELIGIBLE" if not any(deficits.values()) else "SOURCE_DEFICIT",
            primary_scaling_eligible=False,
            track="external_assisted",
        )
    )
