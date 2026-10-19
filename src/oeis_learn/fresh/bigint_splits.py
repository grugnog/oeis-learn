"""Matched target eligibility and conservative observed-behavior overlap groups."""

from collections import defaultdict
from oeis_learn.synthesis.contracts import identity, canonical_integer
from oeis_learn.decoder.bigint_codec import encode_body


def paired_tokens(row):
    try:
        return dict(
            concrete=encode_body(row["source"]), parameterized=encode_body(row["parameter_source"])
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("paired exclusion: either target exceeds codec limits") from exc


def overlap_groups(rows):
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate example identity")
    parent = {k: k for k in ids}
    seen = {}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for row in rows:
        values = row["values"]
        indices = row["indices"]
        if len(values) != 100 or len(indices) != 100:
            raise ValueError("complete 100-index truth required")
        ns = list(map(canonical_integer, indices))
        for value in values:
            canonical_integer(value, bits=16384)
        if any(a >= b for a, b in zip(ns, ns[1:])):
            raise ValueError("indices must increase")
        # Every observed 20-term window: catches offsets anywhere in the horizon.
        # Relative index deltas prevent fabricating overlap across unequal gaps.
        keys = []
        for i in range(81):
            keys.append(identity([values[i : i + 20], [n - ns[i] for n in ns[i : i + 20]]]))
        for key in keys:
            if key in seen:
                a, b = root(row["id"]), root(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = row["id"]
    members = defaultdict(list)
    for k in ids:
        members[root(k)].append(k)
    group_ids = {k: identity(sorted(v)) for k, v in members.items()}
    return {k: group_ids[root(k)] for k in ids}


def behavior(row):
    ns = [canonical_integer(n) for n in row["indices"]]
    vs = [canonical_integer(v, bits=16384) for v in row["values"]]
    constant = len(set(vs)) == 1
    offset = len({v - n for n, v in zip(ns, vs)}) == 1
    delta = ns[1] - ns[0]
    change = vs[1] - vs[0]
    affine = all((v - vs[0]) * delta == change * (n - ns[0]) for n, v in zip(ns, vs))
    return constant, offset, not affine


def _select(rows, count, groups, used, *, tiny=False, facts=None):
    facts = facts or {r["id"]: behavior(r) for r in rows}
    selected = []
    constant = offset = 0
    for row in sorted(rows, key=lambda r: (not facts[r["id"]][2], r.get("stage", ""), r["id"])):
        group = groups[row["id"]]
        c, o, _ = facts[row["id"]]
        if (
            group in used
            or constant + c > (count // 10 if not tiny else count)
            or offset + o > (8 if tiny else count // 10)
        ):
            continue
        selected.append(row)
        used.add(group)
        constant += c
        offset += o
        if len(selected) == count:
            break
    return selected


def diverse_tiny(rows):
    groups = overlap_groups(rows)
    facts = {r["id"]: behavior(r) for r in rows}
    selected = []
    used = set()
    families = set()
    offsets = 0
    ordered = sorted(rows, key=lambda r: (not facts[r["id"]][2], r["id"]))
    for distinct in (True, False):
        for row in ordered:
            if groups[row["id"]] in used or distinct and row["family"] in families:
                continue
            if offsets + facts[row["id"]][1] > 8:
                continue
            selected.append(row)
            used.add(groups[row["id"]])
            families.add(row["family"])
            offsets += facts[row["id"]][1]
            if distinct and len(families) >= 4 or len(selected) == 32:
                break
        if len(selected) == 32:
            break
    if (
        len(selected) != 32
        or len({r["family"] for r in selected}) < 4
        or sum(facts[r["id"]][2] for r in selected) < 16
    ):
        raise ValueError(
            f"diverse32 shortfall: count={len(selected)}, structures={len({r['family'] for r in selected})}, nonaffine={sum(facts[r['id']][2] for r in selected)}"
        )
    return selected


def choose_splits(rows, counts=(512, 128, 128)):
    if tuple(counts) != (512, 128, 128):
        raise ValueError("split sizes are frozen at 512/128/128")
    eligible = []
    excluded = []
    for row in rows:
        try:
            paired_tokens(row)
        except ValueError as exc:
            excluded.append(dict(id=row["id"], reason=str(exc)))
            continue
        eligible.append(row)
    groups = overlap_groups(eligible)
    facts = {r["id"]: behavior(r) for r in eligible}
    families = defaultdict(list)
    for row in eligible:
        families[row["family"]].append(row)
    # Retain only fully sampled families with at least 16 distinct variant IDs.
    families = {k: v for k, v in families.items() if len({r.get("variant") for r in v}) >= 16}
    keys = sorted(families, key=lambda k: identity(["structural-reservation/v1", k]))
    # Whole-family structural reserve; deterministic retries are data-only, never model-selected.
    structural_keys = set(keys[::4])
    used = set()
    structural = _select(
        [r for k in structural_keys for r in families[k]], 128, groups, used, facts=facts
    )
    train = []
    parameter = []
    # Four train variants plus one held-out variant per selected family gives
    # an exact, reproducible 512/128 pairing without deleting holdout families.
    for k in keys:
        if k in structural_keys or len(parameter) == 128:
            continue
        trial = set(used)
        batch = []
        for row in sorted(families[k], key=lambda r: r["id"]):
            if groups[row["id"]] in trial:
                continue
            target = train + batch if len(batch) < 4 else parameter
            limit = 51 if len(batch) < 4 else 12
            c, o, _ = facts[row["id"]]
            if (
                sum(facts[r["id"]][0] for r in target) + c > limit
                or sum(facts[r["id"]][1] for r in target) + o > limit
            ):
                continue
            batch.append(row)
            trial.add(groups[row["id"]])
            if len(batch) == 5:
                break
        if len(batch) != 5:
            continue
        train.extend(batch[:4])
        parameter.append(batch[4])
        used = trial
    result = dict(
        training=train, parameter_development=parameter, structural_development=structural
    )
    missing = {
        name: want - len(result[name])
        for name, want in zip(result, counts)
        if len(result[name]) != want
    }
    if missing:
        raise ValueError(f"split shortfall: {missing}; paired exclusions={len(excluded)}")
    return dict(
        splits=result,
        groups=groups,
        excluded=excluded,
        identity=identity({k: [r["id"] for r in v] for k, v in result.items()}),
        limitations="observed 20-term shifted windows only; no complete semantic equivalence claim",
    )
