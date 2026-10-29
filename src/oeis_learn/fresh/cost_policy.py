"""Complete scheduled screen cost; failure requires a newly registered recipe."""

import math


def screen_cost(
    *,
    training_p95,
    teacher_forcing_p95,
    generation_p95,
    fit_p95,
    reference_p95,
    hidden_scoring_p95,
    checkpoint_p95,
    updates=3000,
    cadence=500,
    budget=7200,
    generation_batch_p95=None,
    generation_batch_size=1,
    scope=None,
):
    costs = (
        training_p95,
        teacher_forcing_p95,
        generation_p95,
        fit_p95,
        reference_p95,
        hidden_scoring_p95,
        checkpoint_p95,
    )
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in costs):
        raise ValueError("measured finite nonnegative p95 costs required")
    if (
        type(updates) is not int
        or type(cadence) is not int
        or updates <= 0
        or cadence <= 0
        or type(budget) not in (int, float)
        or not math.isfinite(budget)
        or budget <= 0
    ):
        raise ValueError("invalid registered schedule")
    scheduled = updates // cadence

    batch_costs = dict(
        generation_p95=generation_p95,
        generation_batch_p95=generation_batch_p95,
        generation_batch_size=generation_batch_size,
    )
    if scope is not None:
        return scoped_cost(
            dict(
                zip(
                    (
                        "training_p95",
                        "teacher_forcing_p95",
                        "generation_p95",
                        "fit_p95",
                        "reference_p95",
                        "hidden_scoring_p95",
                        "checkpoint_p95",
                    ),
                    costs,
                ),
                generation_batch_p95=generation_batch_p95,
                generation_batch_size=generation_batch_size,
            ),
            scope,
            updates=updates,
            cadence=cadence,
            budget=budget,
        )

    phases = dict(
        training=updates * training_p95,
        periodic_synthesis=scheduled
        * (generation_seconds(96, batch_costs) + 96 * (fit_p95 + reference_p95)),
        full_development_loss=(scheduled + 1) * 256 * teacher_forcing_p95,
        end_screen_generation=32 * generation_seconds(256, batch_costs),
        end_screen_fitting=256 * 40 * (fit_p95 + reference_p95),
        # raw, same-candidate rescue, equal-budget, shuffled, absent, template
        end_screen_hidden=256 * 6 * hidden_scoring_p95,
        checkpoint=(scheduled + 1) * checkpoint_p95,
    )
    total = sum(phases.values())
    return dict(
        phases=phases,
        projected_seconds=total,
        budget=budget,
        reserve_fraction=0.2,
        within_budget=total <= 0.8 * budget,
        scheduled_validations=scheduled,
        end_screen_prompts=256,
        omitted_prompts=0,
        generation_batch_size=generation_batch_size,
    )


def workload(scope):
    """Frozen command workloads; scope is admission authority, not a label."""
    if type(scope) is not str or scope not in ("diverse32", "transfer-screen"):
        raise ValueError("unknown cost scope")
    return dict(
        periodic_prompts=32 if scope == "diverse32" else 96,
        end_screen_prompts=32 if scope == "diverse32" else 256,
        end_groups=[32] if scope == "diverse32" else [128, 128],
        teacher_forcing_prompts=256,
        periodic_attempts=1,
        attempts=8,
        controls=["correct", "shuffled", "absent", "template"],
        generation_rounds=24,  # Three neural controls; templates never decode.
        fitting_rounds=32,  # Three neural portfolios plus eight templates.
        raw_rounds=3,  # One raw greedy check per neural control.
        hidden_rounds=6,  # raw/rescue/equal-budget/shuffled/absent/template
        checkpoints_per_validation=2,  # validation save and role-final save
    )


def periodic_cost(costs, scope):
    work = workload(scope)
    n = work["periodic_prompts"]
    return dict(
        generation=generation_seconds(n, costs),
        fitting=n * (costs["fit_p95"] + costs["reference_p95"]),
        # A measured 100-index production+reference check bounds the raw20 check.
        raw=n * costs["hidden_scoring_p95"],
        loss=work["teacher_forcing_prompts"] * costs["teacher_forcing_p95"],
        checkpoint=work["checkpoints_per_validation"] * costs["checkpoint_p95"],
    )


def training_role_seconds(costs, scope, updates):
    if type(updates) is not int or updates <= 0:
        raise ValueError("positive update reservation required")
    return (
        math.ceil(
            (updates * costs["training_p95"] + sum(periodic_cost(costs, scope).values())) / 0.8
        )
        + 5
    )


def end_role_seconds(costs, scope, prompts, role):
    work = workload(scope)
    if prompts not in work["end_groups"] or type(prompts) is not int:
        raise ValueError("end-of-run prompt count differs from cost scope")
    if role == "generation":
        raw = (
            work["generation_rounds"] * generation_seconds(prompts, costs)
            + prompts * work["fitting_rounds"] * (costs["fit_p95"] + costs["reference_p95"])
            + prompts * work["raw_rounds"] * costs["hidden_scoring_p95"]
        )
    elif role == "score":
        raw = prompts * work["hidden_rounds"] * costs["hidden_scoring_p95"]
    else:
        raise ValueError("unknown end-of-run role")
    return math.ceil(raw / 0.8) + 5


def scoped_cost(costs, scope, *, updates, cadence, budget):
    work = workload(scope)
    scheduled = math.ceil(updates / cadence)
    periodic = periodic_cost(costs, scope)
    n = work["end_screen_prompts"]
    phases = dict(
        training=updates * costs["training_p95"],
        periodic_synthesis=scheduled
        * (periodic["generation"] + periodic["fitting"] + periodic["raw"]),
        full_development_loss=scheduled * periodic["loss"],
        end_screen_generation=sum(
            work["generation_rounds"] * generation_seconds(group, costs)
            for group in work["end_groups"]
        ),
        end_screen_fitting=n * work["fitting_rounds"] * (costs["fit_p95"] + costs["reference_p95"]),
        end_screen_raw=n * work["raw_rounds"] * costs["hidden_scoring_p95"],
        end_screen_hidden=n * work["hidden_rounds"] * costs["hidden_scoring_p95"],
        checkpoint=scheduled * periodic["checkpoint"],
    )
    reservations = [
        training_role_seconds(costs, scope, min(cadence, updates - i))
        for i in range(0, updates, cadence)
    ]
    reservations += [
        end_role_seconds(costs, scope, group, role)
        for group in work["end_groups"]
        for role in ("generation", "score")
    ]
    total = sum(phases.values())
    return dict(
        version="scoped-cost/v1",
        scope=scope,
        workload=work,
        phases=phases,
        p95=costs,
        projected_seconds=total,
        budget=budget,
        reserve_fraction=0.2,
        within_budget=total <= 0.8 * budget and sum(reservations) <= budget,
        scheduled_validations=scheduled,
        cadence=cadence,
        updates=updates,
        end_screen_prompts=n,
        omitted_prompts=0,
        generation_batch_size=costs["generation_batch_size"],
        role_reservations=reservations,
        reserved_seconds=sum(reservations),
    )


def profile_identity(recipe):
    """Keep v1 hashes unchanged; opt-in policies are part of measured behavior."""
    from .common import digest

    payload = {k: recipe[k] for k in ('execution', 'inputs', 'model', 'fitting', 'search')}
    if 'pipeline' in recipe:
        payload['pipeline'] = recipe['pipeline']
    return digest(payload)


def schedule_identity(recipe, scope):
    from .common import digest

    workload(scope)
    payload = dict(
            scope=scope,
            data=recipe["data"],
            evaluation=recipe["evaluation"],
            budgets=recipe["budgets"],
            learning={k: v for k, v in recipe["learning"].items() if k != "cost_preflight"},
        )
    if 'pipeline' in recipe:
        payload['pipeline'] = recipe['pipeline']
    return digest(payload)


def validate_preflight(recipe, *, scope, image, source_sha256, shape_sha256):
    """Old receipts remain readable, but cannot authorize a new scoped run."""
    from .common import digest

    receipt = recipe["learning"]["cost_preflight"]
    if not receipt or receipt.get("scope") != scope:
        raise ValueError("matching scope-specific cost preflight required before training")
    if (
        receipt["image"] != image
        or receipt["source_sha256"] != source_sha256
        or receipt["shape_sha256"] != shape_sha256
        or receipt["schedule_sha256"] != schedule_identity(recipe, scope)
        or receipt["profile_sha256"]
        != profile_identity(recipe)
        or receipt["scoring_data"] is not False
    ):
        raise ValueError("cost preflight source/image/shape/profile/schedule mismatch")
    forecast = screen_cost(
        **receipt["costs"],
        scope=scope,
        updates=recipe["learning"]["max_updates"],
        cadence=recipe["learning"]["validation_every"],
        budget=recipe["budgets"]["training_seconds"],
    )
    if not forecast["within_budget"]:
        raise ValueError("scoped projection exceeds80%; preregister revised recipe")
    return forecast


def prepared_shape(inputs):
    """Identical host/measurement shape binding; read public prepared records only."""
    from pathlib import Path
    from .common import read, file_hash

    inputs = Path(inputs)
    prepared, examples = read(inputs / "prepared.json"), read(inputs / "examples.json")
    rows = [r for rs in examples["views"]["concrete"].values() for r in rs]
    return dict(
        prepared_sha256=file_hash(inputs / "prepared.json"),
        split_sha256=prepared["split_sha256"],
        counts=prepared["counts"],
        tiny_count=len(examples["tiny_ids"]),
        exclusions=prepared["exclusions"],
        native_admitted=prepared["native_admitted"],
        stages={s: sum(r["stage"] == s for r in rows) for s in "ABCD"},
        max_input_bits=max(abs(int(v)).bit_length() for r in rows for v in r["values"]),
        max_tokens=max(
            len(r["tokens"])
            for view in examples["views"].values()
            for rs in view.values()
            for r in rs
        ),
    )


def generation_seconds(prompts, costs):
    """Charge one decode round, including singleton upper bounds for tail rows."""
    if type(prompts) is not int or prompts < 0:
        raise ValueError("nonnegative exact prompt count required")
    generation_p95 = costs["generation_p95"]
    generation_batch_p95 = costs.get("generation_batch_p95")
    generation_batch_size = costs.get("generation_batch_size", 1)
    if (
        type(generation_p95) not in (int, float)
        or not math.isfinite(generation_p95)
        or generation_p95 < 0
    ):
        raise ValueError("measured finite singleton cost required")
    if (
        type(generation_batch_size) is not int
        or generation_batch_size not in (1, 8)
        or (generation_batch_p95 is None) != (generation_batch_size == 1)
        or (
            generation_batch_p95 is not None
            and (
                type(generation_batch_p95) not in (int, float)
                or not math.isfinite(generation_batch_p95)
                or generation_batch_p95 < 0
            )
        )
    ):
        raise ValueError("complete measured batch8 latency or retained singleton costs required")
    if generation_batch_p95 is None:
        return prompts * generation_p95
    full, tail = divmod(prompts, generation_batch_size)
    return full * generation_batch_p95 + tail * generation_p95


def pipeline_forecast(costs,scope,*,updates,cadence,budget):
    """New exact-scope receipt. Old v1 readers/identities are deliberately unchanged.

    Missing startup/export/cleanup and material workload observations are unknown,
    never zero. Observations use conservative calibrated envelopes, not an invented
    population p95 from three repetitions.
    """
    from oeis_learn.pipeline_validation.identity import seal, number
    workload(scope)
    required={'training_p95','teacher_forcing_p95','generation_p95','fit_p95',
              'reference_p95','hidden_scoring_p95','checkpoint_p95','startup','export','cleanup'}
    missing=sorted(k for k in required if costs.get(k) is None)
    if missing:
        return seal(dict(version='pipeline-scoped-cost/v1',scope=scope,missing=missing,
            projected_seconds=None,within_budget=False,budget=budget,reason='unknown complete-schedule components'))
    if set(costs)-required-{'generation_batch_p95','generation_batch_size'}:
        raise ValueError('unknown forecast component')
    for key,value in costs.items():
        if value is not None:
            number(value)
    base={k:v for k,v in costs.items() if k not in {'startup','export','cleanup'}}
    base.setdefault('generation_batch_p95',None);base.setdefault('generation_batch_size',1)
    result=screen_cost(**base,scope=scope,updates=updates,cadence=cadence,budget=budget)
    # Startup applies to every bounded learning/generation/scoring role, not once
    # for an entire many-role study. Export/cleanup likewise charge each role.
    roles=len(result['role_reservations'])
    overhead=roles*sum(costs[k] for k in ('startup','export','cleanup'))
    total=result['projected_seconds']+overhead
    return seal(dict(version='pipeline-scoped-cost/v1',scope=scope,workload=result['workload'],
        schedule=dict(updates=updates,cadence=cadence),components=result['phases'],role_count=roles,
        overhead_seconds=overhead,projected_seconds=total,budget=budget,missing=[],
        within_budget=total<=.8*budget and result['reserved_seconds']+overhead<=budget,
        interpretation='calibrated conservative component envelopes; no population p95 claim'))
