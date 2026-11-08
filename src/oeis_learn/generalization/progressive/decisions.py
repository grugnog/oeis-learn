"""Recomputed evidence verdicts, separately validated frozen next actions."""

import math

from .contracts import digest, record, validate
from .metrics import compare
from .targets import validate_action, validate_policy


def classify(
    *,
    mean_gain,
    seed_effects,
    completion_by_arm_seed_stratum,
    retention_losses,
    minimum_groups,
    lower,
    upper,
    integrity,
    claim,
    cost_reduction=None,
    stratum_lowers=None,
):
    """Pure diagnostic classification. Only publish/verify produce authority."""
    if integrity is not True:
        return "INVALID"
    values = [
        mean_gain,
        lower,
        upper,
        *seed_effects,
        *completion_by_arm_seed_stratum,
        *retention_losses,
    ]
    if claim not in {"capability", "efficiency"} or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in values
    ):
        return "INVALID"
    if (
        type(minimum_groups) is not int
        or minimum_groups < 16
        or len(seed_effects) < 2
        or len(retention_losses) != len(seed_effects)
        or len(completion_by_arm_seed_stratum) != 4 * len(seed_effects)
        or any(not 0 <= x <= 1 for x in completion_by_arm_seed_stratum)
        or min(completion_by_arm_seed_stratum) < 0.95
    ):
        return "INCOMPLETE"
    if lower > upper or not -1 <= lower <= upper <= 1:
        return "INVALID"
    if upper < 0:
        return "REGRESSION"
    retained = max(retention_losses) <= 0.02
    if (
        claim == "capability"
        and retained
        and mean_gain >= 0.02
        and min(seed_effects) > 0
        and lower > 0
    ):
        return "CONFIRMED_GAIN"
    if claim == "efficiency" and retained:
        bounds = [lower] if stratum_lowers is None else list(stratum_lowers)
        if (
            type(cost_reduction) in (int, float)
            and math.isfinite(cost_reduction)
            and cost_reduction >= 0.2
            and bounds
            and all(type(x) in (int, float) and math.isfinite(x) and x >= -0.01 for x in bounds)
        ):
            return "CONFIRMED_EFFICIENCY"
    if upper < 0.02:
        return "NO_MATERIAL_GAIN"
    return "INCONCLUSIVE"


def _stratum_bounds(rows, roster, seeds, *, bootstrap_seed, comparisons, links):
    """Preserve the registered joint clusters even for marginal claim bounds."""
    from collections import defaultdict
    from fractions import Fraction

    from .metrics import PRIMARY, _paired, _resample, decision_bounds

    aligned, seeds, _ = _paired(rows, roster, seeds)
    groups = defaultdict(list)
    for row in aligned:
        if row["stratum"] in PRIMARY:
            groups[row["stratum"], row["group"], row["seed"], row["arm"]].append(row)
    vectors = {s: {} for s in PRIMARY}
    for stratum, group in sorted(
        {(r["stratum"], r["group"]) for r in aligned if r["stratum"] in PRIMARY}
    ):
        effects = []
        for seed in seeds:
            a, b = (groups[stratum, group, seed, arm] for arm in ("baseline", "candidate"))
            effects.append(
                Fraction(
                    sum(r["outcome"] == "success" for r in b)
                    - sum(r["outcome"] != "failure" for r in a),
                    len(a),
                )
            )
        value = sum(effects) / len(seeds)
        vectors[stratum][group] = (value, value, value)
    result = {}
    for stratum in PRIMARY:
        isolated = {
            s: {g: v if s == stratum else (0, 0, 0) for g, v in vectors[s].items()} for s in PRIMARY
        }
        draws = _resample(isolated, links, seed=bootstrap_seed, replicates=10000)
        result[stratum] = 2 * decision_bounds([d[0] for d in draws], comparisons=comparisons)[0]
    return result


COST_POLICY = {
    "metric": "total_active_us",
    "scope": "preparation_search_training",
    "amortization": "none",
}


def recompute(rows, roster, seeds, design, *, costs=None):
    """Design is frozen independently of results; no supplied summary is trusted."""
    if set(design) != {"claim", "bootstrap_seed", "comparisons", "links", "costs"}:
        raise ValueError("closed registered comparison design required")
    if design["claim"] not in {"capability", "efficiency"}:
        raise ValueError("registered claim required")
    if design["claim"] == "efficiency":
        if design["costs"] != COST_POLICY:
            raise ValueError("prospectively registered total-cost scope required")
        if costs is not None and (
            not isinstance(costs, dict)
            or set(costs) != {"baseline", "candidate"}
            or any(type(v) is not int or v <= 0 for v in costs.values())
        ):
            raise ValueError("positive measured total cost microseconds required")
    elif design["costs"] is not None or costs is not None:
        raise ValueError("capability design cannot retroactively select efficiency")
    rows, roster, seeds = list(rows), list(roster), list(seeds)
    metrics = compare(
        rows,
        roster=roster,
        seeds=seeds,
        seed=design["bootstrap_seed"],
        comparisons=design["comparisons"],
        links=design["links"],
    )
    if metrics["support_status"] != "SUFFICIENT" or any(
        v is None for v in metrics["retention_losses"].values()
    ):
        return metrics, "INCOMPLETE"
    completion = [
        metrics["cells"][a][s][t]["completion"]
        for a in ("baseline", "candidate")
        for s in seeds
        for t in ("composition", "unseen_structure")
    ]
    cost_reduction, stratum_lowers = None, None
    if design["claim"] == "efficiency":
        if design["costs"] != COST_POLICY:
            raise ValueError("prospectively registered total-cost scope required")
        if costs is None:
            return metrics, "INCOMPLETE"
        if (
            not isinstance(costs, dict)
            or set(costs) != {"baseline", "candidate"}
            or any(type(v) is not int or v <= 0 for v in costs.values())
        ):
            raise ValueError("positive measured total cost microseconds required")
        cost_reduction = (costs["baseline"] - costs["candidate"]) / costs["baseline"]
        bounds = _stratum_bounds(
            rows,
            roster,
            seeds,
            bootstrap_seed=design["bootstrap_seed"],
            comparisons=design["comparisons"],
            links=design["links"],
        )
        stratum_lowers = list(bounds.values())
        metrics["efficiency_stratum_lowers"] = bounds
    elif design["costs"] is not None or costs is not None:
        raise ValueError("capability design cannot retroactively select efficiency")
    verdict = classify(
        mean_gain=metrics["mean_gain"],
        seed_effects=list(metrics["seed_effects"].values()),
        completion_by_arm_seed_stratum=completion,
        retention_losses=list(metrics["retention_losses"].values()),
        minimum_groups=metrics["minimum_groups"],
        lower=metrics["lower"],
        upper=metrics["upper"],
        integrity=True,
        claim=design["claim"],
        cost_reduction=cost_reduction,
        stratum_lowers=stratum_lowers,
    )
    return metrics, verdict


def publish(
    rows,
    roster,
    seeds,
    design,
    *,
    policy,
    target_id,
    action,
    remaining,
    created_at,
    purpose,
    track,
    trigger=None,
    visited=(),
    costs=None,
    confirmation=None,
):
    validate_policy(policy)
    if not policy["payload"]["selected"]:
        raise ValueError("selected pre-outcome policy required")
    target = next((t for t in policy["payload"]["targets"] if t["id"] == target_id), None)
    if target is None or digest(design) not in target["evidence"]:
        raise ValueError("comparison design must be registered in the frozen target policy")
    fixed = {"gain": "20000", "completion": "950000", "retention": "20000"}
    if any(
        t["id"] in fixed and t["threshold"] != fixed[t["id"]] for t in policy["payload"]["targets"]
    ):
        raise ValueError("outcome-driven threshold change invalid")
    if confirmation is None:
        metrics, verdict = recompute(rows, roster, seeds, design, costs=costs)
    else:
        if costs is not None:
            raise ValueError("third-seed route is a capability confirmation")
        from .contingencies import confirm

        metrics, verdict = confirm(
            rows,
            roster,
            seeds,
            design,
            confirmation=confirmation,
            policy=policy,
            remaining=remaining,
        )
    validate_action(
        policy,
        target_id,
        verdict=verdict,
        action=action,
        remaining=remaining,
        trigger=trigger,
        visited=visited,
    )
    # Bind canonical input projections; integer seed keys in metrics are converted
    # only for digesting, never accepted as independently supplied authority.
    import json

    normalized = json.loads(json.dumps(metrics))
    payload = {
        "outcomes": digest(rows),
        "roster": digest(roster),
        "seeds": list(seeds),
        "design": digest(design),
        "costs": digest(costs) if costs is not None else None,
        "policy": policy["identity"],
        "metrics": digest(normalized),
        "evidence_verdict": verdict,
        "next_action": action,
        "target_id": target_id,
        "trigger": trigger,
    }
    if confirmation is not None:
        payload["confirmation"] = digest(confirmation)
    return record("decision", payload, created_at=created_at, purpose=purpose, track=track)


def verify(decision, rows, roster, seeds, design, **kwargs):
    validate("decision", decision)
    expected = publish(
        rows,
        roster,
        seeds,
        design,
        created_at=decision["created_at"],
        purpose=decision["purpose"],
        track=decision["track"],
        **kwargs,
    )
    if expected != decision:
        raise ValueError("decision differs from independently recomputed outcomes/action")
    return expected
