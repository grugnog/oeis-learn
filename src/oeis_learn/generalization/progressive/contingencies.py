"""Finite prospective diagnostic actions and the all-three confirmation rule."""

import math

from oeis_learn.pipeline_validation.identity import seal

from .trials import integer


def third_seed(*, effects, completion, retention, lower, original_verdict, registered, comparisons):
    integer(comparisons, 2)
    if registered is not True or len(effects) != 3 or len(completion) != 3 or len(retention) != 3:
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
