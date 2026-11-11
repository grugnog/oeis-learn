"""Outcome-blind, bounded synthetic operating-characteristic experiments.

Assumptions are scenarios, never estimates fitted to real confirmation outcomes.
This CPU diagnostic does not qualify learning, discover data or change thresholds.
"""

import itertools
import random
from collections import Counter

from .contracts import digest
from .decisions import COST_POLICY, recompute
from .metrics import _seed

FIELDS = {
    "effect_ppm",
    "censor_ppm",
    "family_dependence_ppm",
    "family_sizes",
    "seed_disagreement",
    "comparisons",
    "claim",
    "groups",
}


def _scenario(s):
    if not isinstance(s, dict) or set(s) != FIELDS:
        raise ValueError("closed synthetic scenario required")
    for key, values in [
        ("effect_ppm", (0, 20000, 50000)),
        ("censor_ppm", (0, 20000, 50000, 100000)),
        ("family_dependence_ppm", (0, 750000)),
    ]:
        if type(s[key]) is not int or s[key] not in values:
            raise ValueError("registered synthetic assumption required")
    if (
        type(s["seed_disagreement"]) is not bool
        or s["claim"] not in ("capability", "efficiency")
        or type(s["comparisons"]) is not int
        or not 1 <= s["comparisons"] <= 500
        or type(s["groups"]) is not int
        or not 1 <= s["groups"] <= 64
        or not isinstance(s["family_sizes"], list)
        or not 1 <= len(s["family_sizes"]) <= 8
        or any(type(n) is not int or not 1 <= n <= 64 for n in s["family_sizes"])
    ):
        raise ValueError("bounded prospective scenario required")
    return s


def default_design(*, seed, repeats=8):
    _seed(seed)
    scenarios = []
    for effect, censor, dependent, disagree, claim in itertools.product(
        (0, 20000, 50000),
        (0, 20000, 50000, 100000),
        (0, 750000),
        (False, True),
        ("capability", "efficiency"),
    ):
        scenarios.append(
            {
                "effect_ppm": effect,
                "censor_ppm": censor,
                "family_dependence_ppm": dependent,
                "family_sizes": [4, 16],
                "seed_disagreement": disagree,
                "comparisons": 4 if claim == "efficiency" else 1,
                "claim": claim,
                "groups": 16,
            }
        )
    return {"seed": seed, "repeats": repeats, "scenarios": scenarios}


def simulate(scenario, *, seed):
    s = _scenario(scenario)
    rng = random.Random(_seed(seed))
    rows, roster = [], []
    for stratum in ("composition", "unseen_structure", "retention"):
        groups = range(s["groups"]) if stratum != "retention" else range(1)
        for g in groups:
            n = s["family_sizes"][g % len(s["family_sizes"])] if stratum != "retention" else 20
            tasks = [f"synthetic/{stratum}/{g}/{i}" for i in range(n)]
            roster.extend({"task": t, "group": str(g), "stratum": stratum} for t in tasks)
            for run in (1, 2):
                common = rng.random()
                for task in tasks:
                    u = common if rng.random() < s["family_dependence_ppm"] / 1e6 else rng.random()
                    shift = s["effect_ppm"] / 1e6
                    if s["seed_disagreement"]:
                        shift += 0.04 if run == 1 else -0.04
                    for arm in ("baseline", "candidate"):
                        outcome = (
                            "success"
                            if u < 0.3 + (shift if arm == "candidate" else 0)
                            else "failure"
                        )
                        if stratum == "retention":
                            outcome = "success"
                        elif rng.random() < s["censor_ppm"] / 1e6:
                            outcome = "unresolved"
                        rows.append(
                            {
                                "seed": run,
                                "arm": arm,
                                "stratum": stratum,
                                "group": str(g),
                                "task": task,
                                "outcome": outcome,
                            }
                        )
    return rows, roster, [1, 2]


def validate_design(design):
    if not isinstance(design, dict) or set(design) != {"seed", "repeats", "scenarios"}:
        raise ValueError("only prospective synthetic assumptions permitted")
    _seed(design["seed"])
    if (
        type(design["repeats"]) is not int
        or not 1 <= design["repeats"] <= 100
        or not isinstance(design["scenarios"], list)
        or not 1 <= len(design["scenarios"]) <= 256
    ):
        raise ValueError("finite sensitivity reservation required")
    for s in design["scenarios"]:
        _scenario(s)
    return design


def repeat_verdict(design, ordinal, repeat):
    """One original scenario/repeat, preserving its preregistered random stream."""
    s = design["scenarios"][ordinal]
    seed = int(
        digest({"design": design["seed"], "scenario": ordinal, "repeat": repeat})[7:15], 16
    )
    rows, roster, seeds = simulate(s, seed=seed)
    _, verdict = recompute(
        rows,
        roster,
        seeds,
        {
            "claim": s["claim"],
            "bootstrap_seed": seed,
            "comparisons": s["comparisons"],
            "links": [],
            "costs": dict(COST_POLICY) if s["claim"] == "efficiency" else None,
        },
        costs={"baseline": 100, "candidate": 80} if s["claim"] == "efficiency" else None,
    )
    return verdict


def run(design):
    design = validate_design(design)
    results = []
    for ordinal, s in enumerate(design["scenarios"]):
        counts = Counter()
        for repeat in range(design["repeats"]):
            verdict = repeat_verdict(design, ordinal, repeat)
            counts[verdict] += 1
        promoted = counts["CONFIRMED_GAIN"] + counts["CONFIRMED_EFFICIENCY"]
        # Efficiency's synthetic positive truth is the assumed 20% cost saving
        # and no population capability loss; a zero-effect efficiency adoption
        # is consequently not a false capability claim.
        positive = s["claim"] == "efficiency" or s["effect_ppm"] >= 20000
        results.append(
            {
                "assumptions": s,
                "verdict_counts": dict(sorted(counts.items())),
                "false_advancement_rate": promoted / design["repeats"] if not positive else None,
                "missed_gain_rate": 1 - promoted / design["repeats"] if positive else None,
                "inconclusive_rate": counts["INCONCLUSIVE"] / design["repeats"],
                "incomplete_rate": counts["INCOMPLETE"] / design["repeats"],
            }
        )
    return {
        "version": "discovery-sensitivity/v1",
        "kind": "DIAGNOSTIC_ONLY",
        "design": digest(design),
        "no_confirmation_inputs": True,
        "scenarios": results,
        "aspiration": "80% only for named scenarios; no guarantee at 2pp boundary",
        "assumptions": [
            "paired common-uniform Bernoulli tasks at baseline probability 0.3",
            "family-shared mixture controls within-family dependence; families independent",
            "fixed two measured seeds; optional opposing 4pp seed shifts",
            "independent arm censoring; exact retained familiar panel; assumed 20% efficiency cost saving",
            "conditional family bootstrap; no claim about actual-model statistical power",
        ],
    }
