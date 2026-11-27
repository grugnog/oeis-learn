"""Paired descriptive measurements in percentage points for fixed round029 rosters."""
from collections import Counter, defaultdict
from fractions import Fraction
from math import isfinite


def contrast(rows, coefficients, *, stratum):
    """Combine unresolved receipt coefficients before assigning adverse/favorable outcomes."""
    strata = ("composition", "unseen_structure") if stratum == "structural" else (stratum,)
    families = defaultdict(list)
    identities = set()
    known = {}
    for row in rows:
        if row["stratum"] not in strata:
            continue
        if row["case"] in identities:
            raise ValueError("duplicate case in frozen contrast")
        identities.add(row["case"])
        families[row["stratum"], row["family"]].append(row)
        for name in coefficients:
            obs = row["observations"][name]
            if (obs["status"] not in {"solve", "miss", "unresolved"} or type(obs["closed"]) is not bool
                    or type(obs["complete"]) is not bool or not obs["receipt"]
                    or known.setdefault(obs["receipt"], obs["status"]) != obs["status"]):
                raise ValueError("inconsistent sealed case outcome")
    support = Counter(s for s,_ in families)
    if any(not support[s] for s in strata):
        return dict(observed_pp=None, lower_pp=None, upper_pp=None, reason="missing stratum",
                    family_counts=dict(support), denominator=len(identities))
    unknown = defaultdict(Fraction)
    paired = Counter({k:0 for k in ("gains","losses","both","neither","unresolved_overlay",
                                    "first_solve_families","last_solve_families")})
    pair = sorted(coefficients, key=coefficients.get)
    binary = [coefficients[k] for k in pair] == [-1,1]
    observed = Fraction()
    counts = {name:Counter() for name in coefficients}
    for (s, _), members in families.items():
        weight = Fraction(100, len(strata)*support[s]*len(members))
        solved = Counter()
        for row in members:
            obs = row["observations"]
            for name, coefficient in coefficients.items():
                status = obs[name]["status"]
                counts[name][status] += 1
                counts[name]["complete"] += obs[name]["complete"]
                counts[name]["closed"] += obs[name]["closed"]
                if status == "solve":
                    observed += weight * coefficient
                    solved[name] += 1
                elif status == "unresolved":
                    unknown[obs[name]["receipt"]] += weight * coefficient
            if binary:
                a,b = [obs[k]["status"] for k in pair]
                paired["unresolved_overlay"] += "unresolved" in (a,b)
                key = { (False, True):"gains", (True, False):"losses",
                        (True, True):"both", (False,False):"neither"}[a=="solve", b=="solve"]
                paired[key] += 1
        if binary:
            paired["first_solve_families"] += not solved[pair[0]] and bool(solved[pair[1]])
            paired["last_solve_families"] += bool(solved[pair[0]]) and not solved[pair[1]]
    return dict(observed_pp=float(observed),
                lower_pp=float(observed+sum(min(0,c) for c in unknown.values())),
                upper_pp=float(observed+sum(max(0,c) for c in unknown.values())),
                denominator=len(identities), family_counts=dict(support),
                outcomes={k:dict(v) for k,v in counts.items()}, **(paired if binary else {}))


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
    eligible = [row for row in looks if row["verified"] and row.get(axis) is not None
                and row[axis] <= budget]
    return max(eligible, key=lambda row:row[axis], default=None)


def factorial(levels):
    def block(ids):
        a,b,c,d = [levels.get(k) for k in ids]
        if any(v is None for v in (a,b,c,d)):
            return None
        return dict(first=[b-a,d-c], second=[c-a,d-b],
                    first_mean=((b-a)+(d-c))/2, second_mean=((c-a)+(d-b))/2,
                    interaction=(d-c)-(b-a))
    return dict(composition_matching=block(("00","01","02","03")),
                batch_decay=block(("00","04","05","06")))


def sampling_interval(rows, coefficients, *, stratum, seed, scale=1, draws=10000):
    """Resample each linked family once per draw across all arms, looks and strata."""
    import random

    rng = random.Random(seed)
    strata = ("composition","unseen_structure") if stratum == "structural" else (stratum,)
    by_group = defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row["stratum"] in strata:
            by_group[row["family"]][row["stratum"]].append(row)
    groups = sorted(by_group)
    contributions = [{s:contrast(members, coefficients, stratum=s)["observed_pp"]
                      for s,members in by_group[g].items()} for g in groups]
    values = []
    for _ in range(draws):
        totals, counts = Counter(), Counter()
        for entry in rng.choices(contributions, k=len(groups)):
            for s, value in entry.items():
                totals[s] += value
                counts[s] += 1
        if all(counts[s] for s in strata):
            values.append(scale * sum(totals[s]/counts[s] for s in strata) / len(strata))
    values.sort()
    return dict(lower=values[int(.025*(len(values)-1))] if values else None,
                upper=values[int(.975*(len(values)-1))] if values else None,
                draws=draws, supported_draws=len(values), seed=seed)
