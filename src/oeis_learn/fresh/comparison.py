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


def compare_runs(roots, output):
    """Assemble immutable per-seed observations; use the qualified gate unchanged."""
    from pathlib import Path
    from copy import deepcopy
    from .common import read, write, digest

    arms = {}
    compatibility = None
    costs = []
    observations = {}
    for root in roots:
        root = Path(root)
        run = read(root / "run.json")
        status = read(root / "status.json")
        report = read(root / "comparison.json")
        if run.get("version") != "bigint-run/v1" or status["status"] != "completed":
            raise ValueError("comparison requires completed bigint run/child reports")
        recipe = deepcopy(run["recipe"])
        seed = recipe["model"].pop("seed")
        view = recipe["learning"].pop("target_view")
        recipe.pop("identity")
        # Initialization seed and representation targets vary in a matched pair.
        key = digest(dict(recipe=recipe, data=report["data"], split=run["split_sha256"]))
        if compatibility is None:
            compatibility = key
        if key != compatibility:
            raise ValueError(
                "mismatched data, numerical/input, resource, portfolio or provenance arms"
            )
        if (view, seed) in arms:
            raise ValueError("duplicate representation/seed arm")
        arms[view, seed] = report
        costs.append(
            dict(
                view=view,
                seed=seed,
                lineage=read(root / "lineage.json") if (root / "lineage.json").exists() else None,
                budget=read(root / "budget.json") if (root / "budget.json").exists() else None,
            )
        )
        observations[view, seed] = report.get("observations", {})
    missing = [
        dict(view=view, seed=seed)
        for view in ("literal", "parameterized")
        for seed in SEEDS
        if (view, seed) not in arms
    ]
    result = dict(
        version="bigint-comparison/v1",
        compatibility=compatibility,
        missing_seeds=missing,
        costs=costs,
        automatic_run_authorized=False,
        qualification="pending" if missing else "evaluated",
        decision=None,
        intervals={},
    )
    if not missing:

        def rows(view, split, mode):
            result = []
            for seed in SEEDS:
                report = arms[view, seed]["splits"]["0:" + split]
                result.append(
                    dict(
                        report["splits"][mode],
                        seed=seed,
                        metric=report["metric"],
                        portfolio=report["portfolio"],
                    )
                )
            return result

        result["decision"] = promotion(
            rows("parameterized", "structural_development", "equal_budget"),
            {
                mode: rows("parameterized", "structural_development", mode)
                for mode in ("shuffled", "absent", "template")
            },
            rows("parameterized", "parameter_development", "equal_budget"),
            rows("literal", "parameter_development", "equal_budget"),
        )
        for split in ("parameter_development", "structural_development"):
            families = {}
            for seed in SEEDS:
                left = observations["literal", seed].get("0:" + split, {})
                right = observations["parameterized", seed].get("0:" + split, {})
                if set(left) != set(right) or len(left) != 128:
                    raise ValueError("paired full128 prompt observations required for intervals")
                for prompt in left:
                    if left[prompt]["family"] != right[prompt]["family"]:
                        raise ValueError("paired family mismatch")
                    families.setdefault(left[prompt]["family"], []).append(
                        int(right[prompt]["outcomes"]["equal_budget"]["success"])
                        - int(left[prompt]["outcomes"]["equal_budget"]["success"])
                    )
            result["intervals"][split] = family_bootstrap(families, seed=SEEDS[0])
    result["identity"] = digest(result)
    write(Path(output) / "comparison.json", result, immutable=True)
    return result
