"""Paired family uncertainty conditional on measured seeds, never seed-population CI.

The reducer accepts only outcome projections, not private targets. Installed
callers must supply the frozen roster/seed roster so missing results are retained
as unresolved. Without a separate roster, rows must be an exact paired rectangle;
that diagnostic convenience cannot certify that a campaign roster was supplied.
No function here issues a decision record or authorizes a subsequent action.
"""

import math
import random
from collections import Counter, defaultdict
from fractions import Fraction

from .cohorts import STRATA
from .targets import defaults

PRIMARY = ("composition", "unseen_structure")
ARMS = ("baseline", "candidate")
FIELDS = {"seed", "arm", "stratum", "group", "task", "outcome"}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("bounded exact frozen seed required")
    return value


def _membership(row):
    if row["stratum"] not in (*STRATA, "retention"):
        raise ValueError("registered stratum required")
    for key in ("group", "task"):
        if type(row[key]) is not str or not 1 <= len(row[key]) <= 1024:
            raise ValueError("bounded group/task identity required")
    return row["stratum"], row["group"]


def _rows(rows):
    rows = list(rows)
    seen = set()
    memberships = {}
    for row in rows:
        if not isinstance(row, dict) or not FIELDS <= row.keys() <= FIELDS | {"complete"}:
            raise ValueError("closed outcome projection required")
        _seed(row["seed"])
        if row["arm"] not in ARMS or row["outcome"] not in {"success", "failure", "unresolved"}:
            raise ValueError("registered arm and outcome required")
        member = _membership(row)
        if memberships.setdefault(row["task"], member) != member:
            raise ValueError("task membership changed between arms/seeds")
        key = row["seed"], row["arm"], row["task"]
        if key in seen:
            raise ValueError("duplicate outcome")
        seen.add(key)
        complete = row.get("complete", row["outcome"] != "unresolved")
        if (
            type(complete) is not bool
            or (row["outcome"] == "failure" and not complete)
            or (row["outcome"] == "unresolved" and complete)
        ):
            raise ValueError("explicit scientific completion disagrees with outcome")
    return rows


def _summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    n = len(rows)
    success = sum(r["outcome"] == "success" for r in rows)
    unresolved = sum(r["outcome"] == "unresolved" for r in rows)
    complete = sum(r.get("complete", r["outcome"] != "unresolved") for r in rows)
    lower = sum(
        (Fraction(sum(r["outcome"] == "success" for r in g), len(g)) for g in groups.values()),
        Fraction(),
    ) / len(groups)
    upper = sum(
        (Fraction(sum(r["outcome"] != "failure" for r in g), len(g)) for g in groups.values()),
        Fraction(),
    ) / len(groups)
    return {
        "tasks": n,
        "groups": len(groups),
        "successes": success,
        "completed_failures": n - success - unresolved,
        "unresolved": unresolved,
        "complete": complete,
        "all_task_yield": float(Fraction(success, n)),
        "family_macro": float(lower),
        "completion": float(Fraction(complete, n)),
        "success_bounds": (float(Fraction(success, n)), float(Fraction(success + unresolved, n))),
        "family_macro_bounds": (float(lower), float(upper)),
    }


def summarize(rows):
    """All tasks stay in the denominator; each family has equal macro weight."""
    rows = _rows(rows)
    if not rows or len({(r["seed"], r["arm"], r["stratum"]) for r in rows}) != 1:
        raise ValueError("one nonempty arm/seed/stratum cell required")
    return _summary(rows)


def _policy():
    return defaults()["payload"]["statistics"]


def _settings(replicates, comparisons, seed):
    policy = _policy()
    if type(replicates) is not int or replicates != policy["replicates"]:
        raise ValueError("exactly 10,000 frozen replicates required")
    if type(comparisons) is not int or not 1 <= comparisons <= policy["max_comparisons"]:
        raise ValueError("comparison count exceeds frozen quantile resolution")
    _seed(seed)
    return policy


def _values(values):
    values = list(values)
    if len(values) != 10000 or any(
        type(v) not in (int, float, Fraction) or not -1 <= v <= 1 or not math.isfinite(v)
        for v in values
    ):
        raise ValueError("10,000 finite bounded difference replicates required")
    return sorted(values)


def decision_bounds(values, *, comparisons=1, upper_values=None):
    """One-based floor/ceil order statistics, with exact integer index arithmetic.

    Pass pessimistic replicates as values and optimistic replicates as
    upper_values. Using the pessimistic samples for both bounds would incorrectly
    claim no material gain in the presence of candidate censoring.
    """
    _settings(10000, comparisons, 0)
    lower = _values(values)
    upper = lower if upper_values is None else _values(upper_values)
    b, denominator = len(lower), 20 * comparisons
    lo = max(1, b // denominator)
    hi = min(b, (b * (denominator - 1) + denominator - 1) // denominator)
    return float(lower[lo - 1]), float(upper[hi - 1])


def nominal_interval(values):
    """Observed-yield nominal 95% interval: floor(0.025B), ceil(0.975B)."""
    values = _values(values)
    b = len(values)
    return float(values[max(1, b // 40) - 1]), float(values[min(b, (39 * b + 39) // 40) - 1])


def _components(cells, links):
    nodes = {f"{s}/{g}" for s in PRIMARY for g in cells.get(s, {})}
    parent = {node: node for node in nodes}

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    seen = set()
    for link in links:
        if not isinstance(link, (tuple, list)) or len(link) != 2:
            raise ValueError("two frozen cross-stratum group identities required")
        a, b = link
        if not isinstance(a, str) or not isinstance(b, str) or a not in nodes or b not in nodes:
            raise ValueError("unknown linked primary group")
        if a.split("/", 1)[0] == b.split("/", 1)[0]:
            raise ValueError("cross-stratum link required")
        pair = tuple(sorted((a, b)))
        if pair in seen:
            raise ValueError("duplicate group link")
        seen.add(pair)
        ra, rb = sorted((root(a), root(b)))
        parent[rb] = ra
    return {node: root(node) for node in sorted(nodes)}


def _resample(cells, links, *, seed, replicates):
    """Exact paired cluster bootstrap; small draws are diagnostic-only test seams.

    Unlinked strata use separate multinomial draws of their family groups.
    Linked strata draw once from the union of dependency components, using each
    component's integer multiplicity for every arm, seed and linked stratum.
    Within each stratum weights are renormalized, then the two strata receive
    exactly half weight. Empty-stratum draws are redrawn at most 128 times;
    exhaustion fails closed. Canonical ordering makes input order irrelevant.
    """
    _seed(seed)
    if type(replicates) is not int or not 1 <= replicates <= 10000:
        raise ValueError("bounded diagnostic replicate count required")
    components = _components(cells, links)
    if any(not cells.get(s) for s in PRIMARY):
        raise ValueError("both primary strata required for bootstrap")
    blocks = [PRIMARY] if links else [(s,) for s in PRIMARY]
    prepared = []
    for block in blocks:
        units = sorted({components[f"{s}/{g}"] for s in block for g in cells[s]})
        positions = {unit: i for i, unit in enumerate(units)}
        strata = []
        for s in block:
            rows = [
                (positions[components[f"{s}/{g}"]], tuple(Fraction(v) for v in cells[s][g]))
                for g in sorted(cells[s])
            ]
            scale = math.lcm(*(v.denominator for _, vs in rows for v in vs))
            strata.append([(i, tuple(int(v * scale) for v in vs)) for i, vs in rows])
            strata[-1] = (scale, strata[-1])
        prepared.append((len(units), strata))
    rng = random.Random(seed)
    result = []
    for _ in range(replicates):
        totals = [Fraction(), Fraction(), Fraction()]
        for size, strata in prepared:
            for _retry in range(128):
                weights = Counter(rng.randrange(size) for _ in range(size))
                if all(sum(weights[i] for i, _ in rows) for _, rows in strata):
                    break
            else:
                raise ValueError("bounded linked bootstrap could not populate every stratum")
            for scale, rows in strata:
                denominator = sum(weights[i] for i, _ in rows) * scale
                for j in range(3):
                    totals[j] += (
                        Fraction(sum(weights[i] * vs[j] for i, vs in rows), denominator) / 2
                    )
        result.append(tuple(totals))
    return result


def _paired(rows, roster, seeds):
    rows = _rows(rows)
    if roster is None:
        if not rows:
            raise ValueError("nonempty frozen roster required")
        roster = list(
            {r["task"]: {k: r[k] for k in ("task", "stratum", "group")} for r in rows}.values()
        )
        explicit = False
    else:
        explicit = True
        roster = list(roster)
        if seeds is None:
            raise ValueError("explicit frozen roster also requires frozen seeds")
    members = {}
    for member in roster:
        if not isinstance(member, dict) or set(member) != {"task", "group", "stratum"}:
            raise ValueError("closed frozen roster member required")
        _membership(member)
        if member["task"] in members:
            raise ValueError("duplicate frozen roster member")
        members[member["task"]] = member
    if not members:
        raise ValueError("nonempty frozen roster required")
    seeds = sorted({r["seed"] for r in rows}) if seeds is None else list(seeds)
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("unique frozen seeds required")
    seeds = sorted(_seed(s) for s in seeds)
    found = {}
    for row in rows:
        if (
            row["seed"] not in seeds
            or row["task"] not in members
            or any(row[k] != members[row["task"]][k] for k in ("stratum", "group"))
        ):
            raise ValueError("outcome does not match frozen roster/seeds")
        found[row["seed"], row["arm"], row["task"]] = row
    aligned = []
    for s in seeds:
        for arm in ARMS:
            for task in sorted(members):
                row = found.get((s, arm, task))
                if row is None:
                    if not explicit:
                        raise ValueError("paired arm/seed membership missing; supply frozen roster")
                    row = dict(members[task], seed=s, arm=arm, outcome="unresolved", complete=False)
                aligned.append(row)
    return aligned, seeds, explicit


def compare(rows, *, seed, replicates=10000, comparisons=1, links=(), roster=None, seeds=None):
    """Fixed primary endpoint and both conservative bounds, without a gate verdict."""
    policy = _settings(replicates, comparisons, seed)
    rows, seeds, explicit = _paired(rows, roster, seeds)
    links = list(links)
    grouped = defaultdict(list)
    families = defaultdict(list)
    for row in rows:
        key = row["arm"], row["seed"], row["stratum"]
        grouped[key].append(row)
        families[key + (row["group"],)].append(row)
    cells = {arm: {s: {} for s in seeds} for arm in ARMS}
    for (arm, s, stratum), selected in sorted(grouped.items()):
        cells[arm][s][stratum] = _summary(selected)
    vectors = {stratum: {} for stratum in PRIMARY}
    seed_vectors = {s: {} for s in seeds}
    for stratum in PRIMARY:
        groups = sorted({r["group"] for r in rows if r["stratum"] == stratum})
        for group in groups:
            paired = []
            for s in seeds:
                a, b = (families[arm, s, stratum, group] for arm in ARMS)
                n = len(a)
                sa, sb = (sum(r["outcome"] == "success" for r in arm) for arm in (a, b))
                ua, ub = (sum(r["outcome"] == "unresolved" for r in arm) for arm in (a, b))
                vector = (
                    Fraction(sb - sa - ua, n),
                    Fraction(sb - sa, n),
                    Fraction(sb + ub - sa, n),
                )
                paired.append(vector)
                seed_vectors[s].setdefault(stratum, []).append(vector)
            vectors[stratum][group] = tuple(
                sum((v[j] for v in paired), Fraction()) / len(seeds) for j in range(3)
            )
    components = _components(vectors, links)
    support = {s: len({components[f"{s}/{g}"] for g in vectors[s]}) for s in PRIMARY}
    deficits = [
        f"{s}: {n} independent groups < {policy['minimum_groups']}"
        for s, n in support.items()
        if n < policy["minimum_groups"]
    ]
    if len(seeds) < 2:
        deficits.append("fewer than two observed paired training seeds")
    effects = {}
    bounds = []
    for s in seeds:
        if all(seed_vectors[s].get(stratum) for stratum in PRIMARY):
            vector = tuple(
                sum(
                    (
                        sum((v[j] for v in seed_vectors[s][stratum]), Fraction())
                        / len(seed_vectors[s][stratum])
                        for stratum in PRIMARY
                    ),
                    Fraction(),
                )
                / 2
                for j in range(3)
            )
            effects[s] = vector[1]
            bounds.append(vector)
        else:
            effects[s] = None
    mean = (
        sum(effects.values(), Fraction()) / len(seeds)
        if all(v is not None for v in effects.values())
        else None
    )
    gain_bounds = (
        (
            float(sum(v[0] for v in bounds) / len(seeds)),
            float(sum(v[2] for v in bounds) / len(seeds)),
        )
        if len(bounds) == len(seeds)
        else None
    )
    retention = {}
    for s in seeds:
        a, b = (cells[arm][s].get("retention") for arm in ARMS)
        retention[s] = (
            float(
                Fraction(a["successes"] + a["unresolved"], a["tasks"])
                - Fraction(b["successes"], b["tasks"])
            )
            if a is not None
            else None
        )
    lower = upper = nominal = None
    if not deficits:
        samples = _resample(vectors, links, seed=seed, replicates=replicates)
        lower, upper = decision_bounds(
            [v[0] for v in samples], upper_values=[v[2] for v in samples], comparisons=comparisons
        )
        nominal = nominal_interval([v[1] for v in samples])
    return {
        "mean_gain": None if mean is None else float(mean),
        "seed_effects": {s: None if v is None else float(v) for s, v in effects.items()},
        "seed_range": (float(min(effects.values())), float(max(effects.values())))
        if mean is not None
        else None,
        "gain_bounds": gain_bounds,
        "lower": lower,
        "upper": upper,
        "nominal_interval": nominal,
        "cells": cells,
        "retention_losses": retention,
        "primary_weights": dict.fromkeys(PRIMARY, 0.5),
        "support": support,
        "minimum_groups": min(support.values()),
        "support_status": "INCOMPLETE" if deficits else "SUFFICIENT",
        "support_deficits": deficits,
        "uncertainty_scope": "family_sampling_conditional_on_observed_seeds",
        "roster_supplied": explicit,
        "replicates": replicates,
        "bootstrap_seed": seed,
        "comparisons": comparisons,
        "statistics_policy": policy,
        "links": sorted(tuple(sorted(link)) for link in links),
    }
