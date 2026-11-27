"""Paired descriptive measurements in percentage points for fixed round029 rosters."""

from collections import Counter, defaultdict
from fractions import Fraction
from math import isfinite

from .contracts import _check, schema, seal


def case_outcome_receipt(case, *, outcome_ref, choice_ref, scorer_ref, truth_ref, status):
    """Bind a private score (or unresolved opportunity) independently of its use at a look."""
    hashes = schema("common")["$defs"]["hash"]
    for ref in (outcome_ref, scorer_ref, truth_ref):
        _check(hashes, ref, {})
    if choice_ref is not None:
        _check(hashes, choice_ref, {})
    if not isinstance(case, str) or not case or status not in {"solve", "miss", "unresolved"}:
        raise ValueError("invalid case outcome receipt")
    if status == "solve" and choice_ref is None:
        raise ValueError("a verified solve requires a sealed choice")
    return seal(dict(version="round029-case-outcome/v1", case=case, outcome_ref=outcome_ref,
                     choice_ref=choice_ref, scorer_ref=scorer_ref, truth_ref=truth_ref, status=status))


def _frozen_rows(rows, looks, names):
    from .graph import velocity_look

    if not names or not all(name in looks for name in names):
        raise ValueError("registered look required for every contrast coordinate")
    members = None
    coordinates = []
    for name in names:
        look = looks[name]
        # Reopen the actual registered graph record, including its order and profile.
        development = [r for r in look["members"] if r["stratum"] != "retention"]
        retention = [r for r in look["members"] if r["stratum"] == "retention"]
        coord = {field: look[field] for field in ("method", "seed", "look", "profile")}
        _check(schema("common")["$defs"]["hash"], look["registration"], {})
        expected = velocity_look(registration=look["registration"], **coord, development=development,
                                 retention=retention, extension=True)
        if type(coord["seed"]) is not int or coord["seed"] < 0 or expected != look:
            raise ValueError("invalid frozen look or roster")
        if coordinates and coord["profile"] != coordinates[0]["profile"]:
            raise ValueError("compared looks have different inference profiles")
        if members is not None and members != look["members"]:
            raise ValueError("compared looks have different frozen rosters")
        members = look["members"]
        coordinates.append(coord)
    roster = {r["task"]: (r["stratum"], r["group"]) for r in members}
    remaining = set(roster)
    families = defaultdict(list)
    known = {}
    for row in rows:
        case = row["case"]
        if case not in remaining or roster[case] != (row["stratum"], row["family"]):
            raise ValueError("case does not match the unique frozen roster")
        remaining.remove(case)
        for name in names:
            obs = row["observations"].get(name)
            if not isinstance(obs, dict) or type(obs.get("closed")) is not bool or type(obs.get("complete")) is not bool:
                raise ValueError("explicit case observation and completion required on frozen roster")
            receipt = obs.get("receipt")
            evidence_fields = ("outcome_ref", "choice_ref", "truth_ref", "scorer_ref")
            if not isinstance(receipt, dict) or not all(k in receipt for k in evidence_fields):
                raise ValueError("sealed per-case outcome receipt required")
            evidence = {k: receipt[k] for k in evidence_fields}
            binding = case_outcome_receipt(case, status=obs["status"], **evidence)
            if receipt != binding or known.setdefault(receipt["outcome_ref"], binding["identity"]) != binding["identity"]:
                raise ValueError("inconsistent case, choice, scorer or truth on outcome receipt")
        families[roster[case]].append(row)
    if remaining:
        raise ValueError("missing case on frozen roster")
    return families, coordinates


def contrast(rows, coefficients, *, stratum, looks):
    """Compare only the complete authenticated rosters of the registered looks."""
    strata = ("composition", "unseen_structure") if stratum == "structural" else (stratum,)
    names = sorted(coefficients)
    all_families, coordinates = _frozen_rows(rows, looks, names)
    families = {k: v for k, v in sorted(all_families.items()) if k[0] in strata}
    support = Counter(s for s, _ in families)
    denominator = sum(len(members) for members in families.values())
    supported = all(support[s] for s in strata)
    unknown = defaultdict(Fraction)
    paired = Counter(
        {
            k: 0
            for k in (
                "gains",
                "losses",
                "both",
                "neither",
                "unresolved_overlay",
                "first_solve_families",
                "last_solve_families",
            )
        }
    )
    pair = sorted(coefficients, key=coefficients.get)
    binary = [coefficients[k] for k in pair] == [-1, 1]
    observed = Fraction()
    counts = {name: Counter() for name in coefficients}
    joint = []
    for (s, family), members in families.items():
        patterns = Counter()
        weight = Fraction(100, len(strata) * support[s] * len(members))
        solved = Counter()
        for row in members:
            obs = row["observations"]
            classes = {}
            pattern = []
            for name in names:
                coefficient = coefficients[name]
                receipt = obs[name]["receipt"]["identity"]
                equality = classes.setdefault(receipt, len(classes))
                pattern.append((obs[name]["status"], obs[name]["complete"], obs[name]["closed"], equality))
                status = obs[name]["status"]
                counts[name][status] += 1
                counts[name]["complete"] += obs[name]["complete"]
                counts[name]["closed"] += obs[name]["closed"]
                if status == "solve":
                    observed += weight * coefficient
                    solved[name] += 1
                elif status == "unresolved":
                    unknown[receipt] += weight * coefficient
            patterns[tuple(pattern)] += 1
            if binary:
                a, b = [obs[k]["status"] for k in pair]
                paired["unresolved_overlay"] += "unresolved" in (a, b)
                key = {
                    (False, True): "gains",
                    (True, False): "losses",
                    (True, True): "both",
                    (False, False): "neither",
                }[a == "solve", b == "solve"]
                paired[key] += 1
        cells = []
        for pattern, count in sorted(patterns.items()):
            outcomes = [dict(status=status, complete=complete, closed=closed, equality=equality)
                        for status, complete, closed, equality in pattern]
            cells.append(dict(count=count, outcomes=outcomes))
        joint.append(dict(stratum=s, family=family, frozen_size=len(members), patterns=cells))
        if binary:
            paired["first_solve_families"] += not solved[pair[0]] and bool(solved[pair[1]])
            paired["last_solve_families"] += bool(solved[pair[0]]) and not solved[pair[1]]
    return dict(
        observed_pp=float(observed) if supported else None,
        lower_pp=float(observed + sum(min(0, c) for c in unknown.values())) if supported else None,
        upper_pp=float(observed + sum(max(0, c) for c in unknown.values())) if supported else None,
        **({} if supported else {"reason": "missing stratum"}),
        denominator=denominator,
        family_counts=dict(support),
        coordinates=coordinates,
        families=joint,
        outcomes={k: dict(v) for k, v in counts.items()},
        **(paired if binary else {}),
    )


def rates(delta_pp, tokens, charged_us):
    def rate(den, scale):
        if den is None or delta_pp is None or den == 0:
            return None
        if den < 0 or not isfinite(den):
            raise ValueError("invalid additional cost")
        return delta_pp * scale / den

    return dict(pp_per_million=rate(tokens, 1_000_000), pp_per_hour=rate(charged_us, 3_600_000_000))


def affordable_look(looks, axis, budget, *, overlap):
    if not overlap[0] <= budget <= overlap[1]:
        return None
    eligible = [
        row
        for row in looks
        if row["verified"] and row.get(axis) is not None and row[axis] <= budget
    ]

    def coordinate(row):
        tokens = row.get("useful_tokens", 0)
        return row[axis], row.get("look", tokens), tokens

    return max(eligible, key=coordinate, default=None)


def factorial(levels):
    def block(ids):
        a, b, c, d = [levels.get(k) for k in ids]
        if any(v is None for v in (a, b, c, d)):
            return None
        return dict(
            first=[b - a, d - c],
            second=[c - a, d - b],
            first_mean=((b - a) + (d - c)) / 2,
            second_mean=((c - a) + (d - b)) / 2,
            interaction=(d - c) - (b - a),
        )

    return dict(
        composition_matching=block(("00", "01", "02", "03")),
        batch_decay=block(("00", "04", "05", "06")),
    )


def sampling_interval(rows, coefficients, *, stratum, seed, looks, scale=1, draws=10000):
    """Resample each linked family once per draw across all arms, looks and strata."""
    import random

    rng = random.Random(seed)
    strata = ("composition", "unseen_structure") if stratum == "structural" else (stratum,)
    result = contrast(rows, coefficients, stratum=stratum, looks=looks)
    coefficients_in_order = [coefficients[name] for name in sorted(coefficients)]
    by_group = defaultdict(dict)
    for family in result["families"]:
        net_solves = sum(pattern["count"] * sum(c for c, outcome in zip(coefficients_in_order, pattern["outcomes"])
                                                if outcome["status"] == "solve")
                         for pattern in family["patterns"])
        by_group[family["family"]][family["stratum"]] = 100 * net_solves / family["frozen_size"]
    contributions = [by_group[g] for g in sorted(by_group)]
    values = []
    for _ in range(draws):
        totals, counts = Counter(), Counter()
        for entry in rng.choices(contributions, k=len(contributions)):
            for s, value in entry.items():
                totals[s] += value
                counts[s] += 1
        if all(counts[s] for s in strata):
            values.append(scale * sum(totals[s] / counts[s] for s in strata) / len(strata))
    values.sort()
    return dict(
        lower=values[int(0.025 * (len(values) - 1))] if values else None,
        upper=values[int(0.975 * (len(values) - 1))] if values else None,
        draws=draws,
        supported_draws=len(values),
        seed=seed,
    )
