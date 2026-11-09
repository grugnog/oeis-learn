"""Finite prospective diagnostic actions and the all-three confirmation rule."""

import math

from oeis_learn.pipeline_validation.identity import seal

from .trials import integer


def third_seed(*, effects, completion, retention, lower, original_verdict, registered, comparisons):
    integer(comparisons, 2)
    if (
        registered is not True
        or len(effects) != 3
        or len(completion) not in {3, 12}
        or len(retention) != 3
    ):
        raise ValueError("registered extra look with all three paired seeds required")
    if original_verdict not in {"INCONCLUSIVE", "NO_MATERIAL_GAIN", "INCOMPLETE"}:
        raise ValueError("original failed decision must be preserved")
    if any(
        type(v) not in (int, float) or not math.isfinite(v)
        for v in [*effects, *completion, *retention, lower]
    ):
        raise ValueError("finite all-seed evidence required")
    if (
        any(not -1 <= v <= 1 for v in effects + retention)
        or any(not 0 <= v <= 1 for v in completion)
        or not -1 <= lower <= 1
    ):
        raise ValueError("invalid effect/completion bounds")
    passed = (
        sum(effects) / 3 >= 0.02
        and sum(v > 0 for v in effects) >= 2
        and min(effects) >= -0.02
        and min(completion) >= 0.95
        and max(retention) <= 0.02
        and lower > 0
    )
    return seal(
        {
            "version": "progressive-third-seed/v1",
            "original_verdict": original_verdict,
            "effects": effects,
            "completion": completion,
            "retention": retention,
            "comparisons": comparisons,
            "evidence_verdict": "CONFIRMED_GAIN" if passed else "INCONCLUSIVE",
            "uncertainty_scope": "family sampling conditional on all three seeds",
        }
    )


def validate_branch(branch, *, evidence, remaining, visited, exposed_panels):
    fields = {
        "id",
        "failure_class",
        "trigger",
        "design",
        "panel",
        "task",
        "reservation",
        "maximum_visits",
        "terminal",
    }
    if (
        not isinstance(branch, dict)
        or set(branch) != fields
        or any(not isinstance(branch[k], str) or not branch[k] for k in fields - {"maximum_visits"})
    ):
        raise ValueError("concrete closed diagnostic branch required")
    if (
        type(branch["maximum_visits"]) is not int
        or branch["maximum_visits"] != 1
        or branch["terminal"] not in {"RETAIN_BASELINE", "DEFER_BRANCH", "STOP_AFFECTED"}
    ):
        raise ValueError("one finite episode and terminal action required")
    if branch["trigger"] == "training_loss" or evidence.get(branch["trigger"]) is not True:
        raise ValueError("held-out/runtime trigger required beyond training loss")
    if branch["failure_class"] in visited or len(visited) >= 4 or branch["panel"] in exposed_panels:
        raise ValueError("recursive/exhausted diagnostic or reused panel")
    if remaining.get(branch["reservation"], 0) <= 0:
        raise ValueError("no frozen diagnostic reservation")
    return branch


def confirm(rows, roster, seeds, design, *, confirmation, policy, remaining):
    """Recompute a registered extra look, preserving the original two-seed proof.

    Scalar third_seed is diagnostic convenience only. This entry point derives
    every bound, completion cell and retention loss from the paired outcomes.
    """
    from .contracts import digest
    from .decisions import recompute, verify

    if set(confirmation) != {"registration", "original"}:
        raise ValueError("closed third-seed confirmation required")
    registration, original = confirmation["registration"], confirmation["original"]
    if (
        set(registration)
        != {
            "version",
            "branch",
            "original_seeds",
            "third_seed",
            "original_design",
            "confirmation_design",
        }
        or registration["version"] != "progressive-third-seed-design/v1"
    ):
        raise ValueError("frozen third-seed design required")
    if set(original) != {"decision", "rows", "roster", "seeds", "design", "kwargs"} or set(
        original["kwargs"]
    ) - {"target_id", "action", "trigger", "costs"}:
        raise ValueError("original closed two-seed evidence required")
    branch = next(
        (b for b in policy["payload"]["diagnostics"] if b["id"] == registration["branch"]), None
    )
    if (
        branch is None
        or branch["signal"] != "seed_disagreement"
        or branch["design"] != digest(registration)
    ):
        raise ValueError("unregistered third-seed branch")
    first = registration["original_seeds"]
    if (
        len(first) != 2
        or len(set(first)) != 2
        or registration["third_seed"] in first
        or list(original["seeds"]) != first
        or list(seeds) != [*first, registration["third_seed"]]
    ):
        raise ValueError("all three fixed seeds required; no best-two selection")
    if (
        digest(original["design"]) != registration["original_design"]
        or digest(design) != registration["confirmation_design"]
        or (design["comparisons"] != original["design"]["comparisons"] or design["comparisons"] < 2)
        or design["claim"] != "capability"
    ):
        raise ValueError("registered designs and extra look must count")
    verify(
        original["decision"],
        original["rows"],
        original["roster"],
        first,
        original["design"],
        policy=policy,
        remaining={**remaining, branch["id"]: branch["reservation"]},
        **original["kwargs"],
    )
    if (
        original["decision"]["payload"]["next_action"] != "EXPLORE_RESERVED"
        or original["decision"]["payload"]["trigger"] != branch["trigger"]
    ):
        raise ValueError("original failed decision must select this reserved diagnostic")
    if roster != original["roster"] or sorted(
        (r for r in rows if r["seed"] in first), key=lambda r: (r["seed"], r["arm"], r["task"])
    ) != sorted(original["rows"], key=lambda r: (r["seed"], r["arm"], r["task"])):
        raise ValueError("original outcomes/roster cannot be replaced after the extra look")
    metrics, ordinary = recompute(rows, roster, seeds, design)
    if ordinary in {"INVALID", "INCOMPLETE", "REGRESSION"}:
        return metrics, ordinary
    result = third_seed(
        effects=list(metrics["seed_effects"].values()),
        completion=[
            metrics["cells"][a][s][t]["completion"]
            for a in ("baseline", "candidate")
            for s in seeds
            for t in ("composition", "unseen_structure")
        ],
        retention=list(metrics["retention_losses"].values()),
        lower=metrics["lower"],
        original_verdict=original["decision"]["payload"]["evidence_verdict"],
        registered=True,
        comparisons=design["comparisons"],
    )
    return metrics, result["evidence_verdict"]


def validate_trigger(branch, *, metrics, remaining, visited=()):
    """Recompute scientific triggers; a requested label is never evidence."""
    effects = list(metrics.get("seed_effects", {}).values())
    effects = [v for v in effects if v is not None]
    signal = branch["signal"]
    if signal == "seed_disagreement":
        observed = len(effects) >= 2 and min(effects) <= 0 < max(effects)
    elif signal == "support_deficit":
        observed = metrics.get("support_status") == "INCOMPLETE"
    elif signal == "heldout_yield":
        observed = metrics.get("mean_gain") is not None and (
            metrics["mean_gain"] < 0.02
            or any(v <= 0 for v in effects)
            or metrics.get("lower") is None
            or metrics["lower"] <= 0
        )
    else:
        # Until an original runtime producer is supplied through the frozen
        # study boundary, neither a string nor training loss can establish it.
        observed = False
    legacy = dict(
        id=branch["id"],
        failure_class=branch["failure_class"],
        trigger=branch["trigger"],
        design=branch["design"],
        panel=branch["panel"],
        task=branch["task_slots"][0],
        reservation=branch["id"],
        maximum_visits=branch["maximum_visits"],
        terminal=branch["terminal_action"],
    )
    return validate_branch(
        legacy,
        evidence={branch["trigger"]: observed},
        remaining={k: v["elapsed_us"] for k, v in remaining.items()},
        visited=visited,
        exposed_panels=(),
    )
