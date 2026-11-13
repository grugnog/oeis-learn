"""Finite prequalified trial alternatives share one immutable task entitlement."""

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
    result = dict(plan, initializer=dict(plan["initializer"]))
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


def materialize(package, plan):
    """Expand a finite frozen parameter choice over unchanged training-only data."""
    if "trial_choice" not in plan:
        return package
    from .models import resolve_profile
    from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy
    from oeis_learn.pipeline_validation.progressive_learner import compatibility
    from .shared_support import seal_package, shared
    package=shared(package)

    params = plan["trial_choices"][plan["trial_choice"]]["parameters"]
    profile = resolve_profile(
        package["profile"]["base_recipe"],
        shape=params.get("shape", "reference"),
        seed=plan["seed"],
        dropout=params["dropout"],
    )
    confirmation = plan["trial"]["phase"] == "confirmation"
    horizon = 400000000 if confirmation else package["update_policy"]["schedule_horizon_tokens"]
    policy = resolve_update_policy(
        profile,
        schedule_horizon_tokens=horizon,
        lr_multiplier=params["lr_multiplier"],
        schedule=params["schedule"],
        effective_batch=params["effective_batch"],
        weight_decay=params["weight_decay"],
        microbatch_size=min(package["update_policy"]["microbatch_size"], params["effective_batch"]),
        warmup_tokens=1250000 if confirmation and params["schedule"] == "warmup_cosine" else None,
    )
    from .prepared_support import apply_batch_cadence
    result = seal_package(apply_batch_cadence(dict(package, profile=profile, update_policy=policy)))
    if compatibility(result) != plan["initializer"]["compatibility"]:
        raise ValueError("finite trial choice differs from frozen learning compatibility")
    return result
