"""Predeclared fitted-portfolio transfer gate and descriptive family bootstrap."""

import random
import math

SEEDS = (20261016, 20261017, 20261018)


def _reports(rows):
    if len(rows) != 3 or {r["seed"] for r in rows} != set(SEEDS):
        raise ValueError("all three paired seeds required")
    result = sorted(rows, key=lambda r: r["seed"])
    for r in result:
        if (
            r.get("metric") != "continuation_accuracy_100"
            or r.get("portfolio") != "exact_equal_budget"
            or r["denominator"] != 128
            or any(type(r[k]) is not int for k in ("success", "complete", "denominator"))
            or not 0 <= r["success"] <= r["complete"] <= 128
        ):
            raise ValueError("full-horizon fixed-denominator fitted portfolio required")
    return result


def promotion(candidate, controls, parameter, literal):
    candidate = _reports(candidate)
    parameter = _reports(parameter)
    literal = _reports(literal)
    if set(controls) != {"shuffled", "absent", "template"}:
        raise ValueError("all comparators required")
    controls = {k: _reports(v) for k, v in controls.items()}
    all_rows = candidate + parameter + literal + [r for rows in controls.values() for r in rows]
    coverage = all(
        r["complete"] * 100 >= 95 * 128
        and r.get("verified_admissions") is True
        and r.get("disagreement") is False
        for r in all_rows
    )
    differences = {
        k: [a["success"] - b["success"] for a, b in zip(candidate, v)] for k, v in controls.items()
    }
    advantages = all(sum(d >= 8 for d in ds) >= 2 and sum(ds) >= 24 for ds in differences.values())
    regression = sum(p["success"] - l["success"] for p, l in zip(parameter, literal)) >= -12
    return dict(
        promote=coverage and advantages and regression,
        coverage=coverage,
        parameter_regression_ok=regression,
        paired_success_differences=differences,
        mean_percentage_points={k: sum(v) / 3 / 128 * 100 for k, v in differences.items()},
        seed_spread={k: [min(v), max(v)] for k, v in differences.items()},
        automatic_run_authorized=False,
    )


def memorization_gate(success, complete, *, termination_reliable):
    return (
        type(success) is int
        and type(complete) is int
        and 31 <= success <= 32
        and complete == 32
        and termination_reliable is True
    )


def family_bootstrap(families, *, seed):
    if not families or type(seed) is not int:
        raise ValueError("frozen family observations and seed required")
    keys = sorted(families)
    if any(
        not families[k]
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or not -1 <= x <= 1
            for x in families[k]
        )
        for k in keys
    ):
        raise ValueError("paired prompt differences required")
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        chosen = [families[rng.choice(keys)] for _ in keys]
        draws.append(sum(sum(v) for v in chosen) / sum(len(v) for v in chosen))
    draws.sort()
    return dict(
        resamples=2000,
        seed=seed,
        interval_95=[draws[49], draws[1949]],
        families=len(keys),
        interpretation="descriptive family-cluster interval; report three-seed spread separately",
    )
