"""Explicit seed/stream lineage and useful positive exposure for round029 only."""

from .campaign import velocity_policy
from .contracts import _check, digest, schema, seal
from .metrics import _seed
from .training_rows import matched_batch
from .trials import integer


def recipe_bindings(registration, *, initialization, stream_seed, ordinary, broad):
    for identity in (registration, ordinary, *([broad] if broad is not None else [])):
        _check(schema("common")["$defs"]["hash"], identity, schema("common"))
    _seed(initialization)
    _seed(stream_seed)
    return {
        row["id"]: seal(
            dict(
                version="round029-trial/v1",
                registration=registration,
                recipe=row,
                initialization=initialization,
                stream_seed=stream_seed,
                stream_ref=ordinary if row["stream"] == "ordinary" else broad,
            )
        )
        for row in velocity_policy()["recipes"]
    }


def exposure(stream, rows, *, updates, batch, discarded=0):
    integer(updates)
    integer(discarded)
    if type(batch) is not int or batch not in (16, 32):
        raise ValueError("registered effective batch required")
    useful = last = 0
    chain = digest(None)
    for i in range(updates):
        ids = matched_batch(stream, i * batch, batch)
        last = sum(len(rows[key]["tokens"]) for key in ids)
        if not last:
            raise ValueError("empty positive targets")
        useful += last
        chain = digest([chain, ids])
    return dict(
        cursor=updates * batch,
        updates=updates,
        useful_tokens=useful,
        charged_tokens=useful + discarded,
        discarded_tokens=discarded,
        last_update_tokens=last,
        order_digest=chain,
    )


def crossed(cut, target):
    integer(target)
    return (
        cut["updates"] == 0
        if target == 0
        else (cut["useful_tokens"] - cut["last_update_tokens"] < target <= cut["useful_tokens"])
    )


def update_policy(profile, recipe_id, *, microbatch, matching_weight=None, horizon=4_000_000):
    from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy

    row = next((r for r in velocity_policy()["recipes"] if r["id"] == recipe_id), None)
    if row is None:
        raise ValueError("unregistered training recipe")
    return resolve_update_policy(
        profile,
        velocity_recipe=recipe_id,
        effective_batch=row["batch"],
        weight_decay=row["decay"],
        schedule_horizon_tokens=horizon,
        microbatch_size=microbatch,
        matching_weight=matching_weight,
    )
