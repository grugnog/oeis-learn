"""Finite prequalified trial alternatives share one immutable task entitlement."""

import copy
from .tuning import SPACE, SHAPES


def resolve(plan, index):
    if type(index) is not int or not 0 <= index < len(plan.get("trial_choices", [])):
        raise ValueError("frozen finite trial choice required")
    choice = plan["trial_choices"][index]
    params = choice["parameters"]
    if (
        set(params) - {"shape"} != set(SPACE)
        or any(params[k] not in SPACE[k] for k in SPACE)
        or params.get("shape", "reference") not in SHAPES
    ):
        raise ValueError("registered trial search space required")
    result = copy.deepcopy(plan)
    result["trial_choice"] = index
    result["initializer"]["compatibility"] = choice["compatibility"]
    result["policy"] = choice["policy"]
    if "runtime" in choice:
        result["trial_runtime"] = choice["runtime"]
    if "iteration_budget" in choice:
        result["iteration_budget"] = choice["iteration_budget"]
    return result


def variants(plan):
    return (
        [resolve(plan, i) for i in range(len(plan["trial_choices"]))]
        if "trial_choices" in plan
        else [plan]
    )
