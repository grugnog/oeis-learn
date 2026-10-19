"""Deterministic structure/cost alternatives, separate from training selection."""

import math
from oeis_learn.synthesis.state import require_hash


def retain_alternatives(rows, cap=8):
    if type(cap) is not int or not 1 <= cap <= 8:
        raise ValueError("archive cap must be 1..8")
    seen = {}
    fields = {
        "id",
        "structure",
        "source",
        "cost",
        "verified",
        "verification",
        "track",
        "profile",
        "assignment",
    }
    for row in rows:
        if (
            set(row) != fields
            or row["verified"] is not True
            or row["track"] not in {"generic_only", "assisted"}
            or row["profile"] != "bigint-body/v1"
        ):
            raise ValueError("independently verified archive record required")
        require_hash(row["verification"])
        if (
            type(row["cost"]) not in (int, float)
            or not math.isfinite(row["cost"])
            or row["cost"] < 0
        ):
            raise ValueError("measured cost required")
        if row["id"] in seen and seen[row["id"]] != row:
            raise ValueError("conflicting archive identity")
        seen[row["id"]] = row
    key = lambda r: (len(r["source"].encode()), r["cost"], r["id"])
    groups = {}
    for row in seen.values():
        groups.setdefault(row["structure"], []).append(row)
    front = []
    dominated = []
    for rows in groups.values():
        for r in rows:
            if any(
                len(q["source"].encode()) <= len(r["source"].encode())
                and q["cost"] <= r["cost"]
                and (len(q["source"].encode()) < len(r["source"].encode()) or q["cost"] < r["cost"])
                for q in rows
            ):
                dominated.append(r)
            else:
                front.append(r)
    representatives = sorted(
        (min((r for r in front if r["structure"] == g), key=key) for g in groups), key=key
    )
    retained = representatives[:cap]
    selected = {r["id"] for r in retained}
    for r in sorted(front, key=key):
        if len(retained) == cap:
            break
        if r["id"] not in selected:
            retained.append(r)
            selected.add(r["id"])
    omissions = [
        dict(
            id=r["id"],
            reason="dominated_within_structure" if r in dominated else "frozen_capacity",
            structure=r["structure"],
            verification=r["verification"],
        )
        for r in sorted(seen.values(), key=key)
        if r["id"] not in selected
    ]
    import copy

    return dict(
        policy="structure-pareto/v1",
        retained=copy.deepcopy(sorted(retained, key=key)),
        omitted=omissions,
    )


def training_representative(archive):
    if not archive["retained"]:
        return None
    import copy

    return copy.deepcopy(
        min(archive["retained"], key=lambda r: (len(r["source"].encode()), r["cost"], r["id"]))
    )
