"""Outcome-blind family reservation and conventional difficulty calibration for round028."""

from collections import Counter, defaultdict
from fractions import Fraction

from oeis_learn.pipeline_validation.progressive_research_census import Union
from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex

from .contracts import digest, seal
from .prepared_panels import composition

BANDS = ("easy", "middle", "hard", "unknown")
RESERVATIONS = (
    "development",
    "replication",
    "development_retention",
    "replication_retention",
    "confirmation",
    "assessment",
)
PARAMETER = {"parameter", "retention"}


def fresh_groups(candidates, historical, *, indexes):
    """Close historical and prospective links before discarding any bridge row."""
    if len({r["id"] for r in candidates}) != len(candidates):
        raise ValueError("duplicate prospective source member")
    rows = [*candidates, *historical]
    relations = [
        composition(r, allow_parameters=True) if r.get("source") else r.get("composition")
        for r in rows
    ]
    # Parameter variants can transfer a structure only when both sides allow it.
    protected_structures = {r.get("structure") for r in rows if r["stratum"] not in PARAMETER}
    protected_compositions = {c for r, c in zip(rows, relations) if r["stratum"] not in PARAMETER}
    union = Union(len(candidates))
    first, blocked = {}, set()
    for i, (row, relation) in enumerate(zip(rows, relations)):
        indexed = set(ExposureIndex.keys(row))
        keys = {item for item in indexed if item[0] not in {"structure", "family", "group"}}
        if row.get("structure") in protected_structures and row.get("structure") is not None:
            keys.add(("structure", row["structure"]))
        if relation and relation in protected_compositions:
            keys.add(("composition", relation))
        for value in [row.get("group"), row.get("family"), *row.get("declared_groups", ())]:
            if value:
                keys.add(("declared", value))
                if i < len(candidates):
                    indexed.update((kind, digest(value)) for kind in ("family", "group"))
        if i >= len(candidates):
            # Every historical member is excluded. Keep only keys that intersect a
            # prospective component rather than retaining the entire old index again.
            blocked.update(first[k] for k in keys if k in first)
            continue
        for key in keys:
            union.join(i, first.setdefault(key, i))
        for index in indexes:
            matches = set(index.matching_keys(indexed))
            if row["stratum"] == "parameter":
                matches.discard("structure")
            if matches:
                blocked.add(i)
                break
    dirty = {union.root(i) for i in blocked}
    result = defaultdict(list)
    for i, row in enumerate(candidates):
        owner = union.root(i)
        if owner not in dirty:
            result[owner].append(row)
    groups = {}
    for members in result.values():
        ordered = sorted(members, key=lambda r: r["id"])
        groups[digest([r["id"] for r in ordered])] = ordered
    eligible = sum(map(len, groups.values()))
    return groups, {
        "candidates": len(candidates),
        "historical_rows": len(historical),
        "eligible": eligible,
        "excluded": len(candidates) - eligible,
        "components": len(groups),
    }


def descriptor(row):
    dimensions = row.get("difficulty") or {}
    if (
        row["stratum"] == "oeis"
        or not row.get("operations")
        or not row.get("source")
        or any(
            type(dimensions.get(k)) is not int or dimensions[k] <= 0
            for k in ("depth", "nodes", "work")
        )
    ):
        return None
    from oeis_learn.synthesis.reference_parser import parse

    program = parse(row["source"])
    root = program.result[0] if isinstance(program.result, tuple) else "read"
    shape = [*[s[0] for s in program.statements], "return:" + root]
    return dict(
        language="bigint-body/v1",
        shape=shape,
        operations=sorted(set(row["operations"])),
        **{k: dimensions[k] for k in ("depth", "nodes", "work")},
    )


def bin_key(value):
    if value is None or any(
        type(value.get(k)) is not int or value[k] <= 0 for k in ("depth", "nodes", "work")
    ):
        return None
    return digest(
        [
            value["language"],
            value["shape"],
            sorted(set(value["operations"])),
            *[value[k].bit_length() - 1 for k in ("nodes", "depth", "work")],
        ]
    )


def calibrate_bands(observations):
    """One frozen witness per independent calibration family; whole bins stay together."""
    families = set()
    bins = defaultdict(list)
    for row in observations:
        family, key = row["family"], bin_key(row["descriptor"])
        if (
            family in families
            or key is None
            or row["denominator"] != 3
            or any(type(row[k]) is not int for k in ("exact", "measured", "denominator"))
            or not 0 <= row["exact"] <= row["measured"] <= row["denominator"]
        ):
            raise ValueError(
                "one witnessed calibration family and three conventional dispositions required"
            )
        families.add(family)
        bins[key].append(row)

    def order(key):
        rows = bins[key]
        n = len(rows)
        return (
            -Fraction(sum(r["exact"] for r in rows), 3 * n),
            -Fraction(sum(r["measured"] for r in rows), 3 * n),
            *[
                Fraction(sum(r["descriptor"][k] for r in rows), n)
                for k in ("nodes", "depth", "work")
            ],
            key,
        )

    result = {}
    prior = 0
    for key in sorted(bins, key=order):
        size = len(bins[key])
        tertile = min(2, (3 * (2 * prior + size - 1)) // (2 * len(families)))
        result[key] = BANDS[tertile]
        prior += size
    return seal(
        {
            "version": "round028-difficulty-map/v1",
            "families": len(families),
            "bins": result,
            "observations": digest(sorted(observations, key=lambda r: r["family"])),
        }
    )


def difficulty_band(value, mapping):
    return mapping["bins"].get(bin_key(value), "unknown")


def calibration_families(groups, seed, *, count=48):
    """Draw once from synthetic, non-parameter families before any comparative or calibration result."""
    eligible = {}
    for family, rows in groups.items():
        if any(r["stratum"] not in {"composition", "unseen_structure"} for r in rows):
            continue
        witnessed = [
            r for r in rows if r.get("native_verified") is True and descriptor(r) is not None
        ]
        if witnessed:
            eligible[family] = min(
                witnessed, key=lambda r: digest([seed, "calibration-case", r["id"]])
            )
    ordered = sorted(eligible, key=lambda f: (digest([seed, "calibration-family", f]), f))
    return {f: eligible[f] for f in ordered[:count]}


def family_options(groups):
    """Keep parameter variants in one statistical family, allowing distinct components in different roles."""
    components = sorted(groups)
    union = Union(len(components))
    first = {}
    for i, component in enumerate(components):
        for row in groups[component]:
            if row["stratum"] == "parameter" and row.get("structure"):
                union.join(i, first.setdefault(row["structure"], i))
    parameter_groups = defaultdict(list)
    for i, component in enumerate(components):
        parameter_groups[union.root(i)].append(component)
    aliases = {c: digest(members) for members in parameter_groups.values() for c in members}
    options = defaultdict(lambda: defaultdict(list))
    for component in components:
        for row in groups[component]:
            family = aliases[component] if row["stratum"] == "parameter" else component
            options[row["stratum"]][family].append((component, row))
    return options


def allocate_roles(groups, *, seed, per_stratum=32):
    """Deal one family per role per round, redistributing band quotas when supply is exhausted."""
    if type(per_stratum) is not int or per_stratum < 1:
        raise ValueError("positive prospective quota required")
    options = family_options(groups)
    used = set()
    owners = {}
    panels = {p: [] for p in RESERVATIONS}
    deficits = {}
    for stratum in ("composition", "unseen_structure", "oeis", "parameter"):
        roles = [p for p in RESERVATIONS if stratum == "parameter" or not p.endswith("_retention")]
        choices = options[stratum]
        members = {p: [] for p in roles}
        family_count = {p: Counter() for p in roles}
        ranks = {}
        for p in roles:
            buckets = {b: [] for b in BANDS}
            for family, pairs in choices.items():
                ordered = sorted(
                    pairs, key=lambda item: (digest([seed, "member", item[1]["id"]]), item[1]["id"])
                )
                band = ordered[0][1]["band"]
                buckets[band].append((digest([seed, p, stratum, family]), family, ordered))
            ranks[p] = {b: iter(sorted(pool)) for b, pool in buckets.items()}

        for _ in range(per_stratum):
            for p in roles:
                remaining = per_stratum - len(members[p])
                if not remaining:
                    continue
                # Equal largest-remainder quotas. The fixed band order breaks equal remainders.
                for band in sorted(BANDS, key=lambda b: (family_count[p][b], BANDS.index(b))):
                    for _, family, pairs in ranks[p][band]:
                        available = [
                            (c, r)
                            for c, r in pairs
                            if r["id"] not in used and owners.get(c, p) == p
                        ]
                        if not available:
                            continue
                        for component, row in available[: min(2, remaining)]:
                            owners[component] = p
                            used.add(row["id"])
                            members[p].append(
                                {
                                    "id": row["id"],
                                    "group": family,
                                    "component": component,
                                    "band": row["band"],
                                    "stratum": "retention" if p.endswith("_retention") else stratum,
                                }
                            )
                        family_count[p][band] += 1
                        break
                    else:
                        continue
                    break
        for p, selected in members.items():
            panels[p].extend(selected)
            if len(selected) < per_stratum:
                deficits[p + "/" + stratum] = {
                    "missing_cases": per_stratum - len(selected),
                    "missing_families": max(
                        0, (per_stratum + 1) // 2 - len({r["group"] for r in selected})
                    ),
                }
    return {"panels": panels, "deficits": deficits}


def freeze_panels(groups, samples, observed, *, seed, per_stratum=32):
    if set(samples) != {r["family"] for r in observed}:
        raise ValueError("complete original calibration family outcomes required")
    if any(descriptor(samples[r["family"]]) != r["descriptor"] for r in observed):
        raise ValueError("calibration observation belongs to a different descriptor")
    bands = calibrate_bands(observed)
    available = {
        g: [dict(r, band=difficulty_band(descriptor(r), bands)) for r in rows]
        for g, rows in groups.items()
        if g not in samples
    }
    return seal(
        dict(
            version="round028-panel-reservation/v1",
            seed=seed,
            calibration=bands["identity"],
            **allocate_roles(available, seed=seed, per_stratum=per_stratum),
        )
    )


def reserve_velocity_families(candidates, historical, *, indexes, quotas, seed):
    """Deal one independent family per turn before any role takes a second member."""
    from oeis_learn.generalization.budgets import integer

    roles = {"development", "replication", "assessment", "development_retention",
             "replication_retention", "calibration", "practice", "teacher"}
    if not set(quotas) <= roles:
        raise ValueError("unknown family reservation")
    groups, census = fresh_groups(candidates, historical, indexes=indexes)
    options = family_options(groups)
    panels = {r: [] for r in quotas}
    owners, used = {}, set()
    deficits = {}
    for stratum in sorted({s for limits in quotas.values() for s in limits}):
        ordering = sorted(options[stratum], key=lambda f: (digest([seed, stratum, f]), f))
        picked = {role: Counter() for role in quotas}
        for turn in range(max(integer(q.get(stratum, 0)) for q in quotas.values())):
            for role in sorted(quotas):
                if turn >= quotas[role].get(stratum, 0):
                    continue
                for family in sorted(ordering, key=picked[role].__getitem__):
                    if owners.get(family, role) != role:
                        continue
                    members = sorted(options[stratum][family],
                                     key=lambda m: digest([seed, stratum, family, m[1]["id"]]))
                    available = [(c, r) for c, r in members if r["id"] not in used
                                 and owners.get(c, role) == role]
                    if not available:
                        continue
                    component, row = available[0]
                    owners[family] = owners[component] = role
                    used.add(row["id"])
                    picked[role][family] += 1
                    panels[role].append(dict(task=row["id"], family=family, component=component,
                                             stratum=stratum, difficulty=row.get("band", "unknown")
                                             if descriptor(row) else "unknown"))
                    break
        for role, q in quotas.items():
            short = q.get(stratum, 0) - sum(picked[role].values())
            if short:
                deficits.setdefault(role, {})[stratum] = short
    return seal(dict(version="round029-family-reservations/v1", seed=seed, panels=panels,
                     deficits=deficits, quotas=quotas, **census))
