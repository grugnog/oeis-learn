"""Explicit seed/stream lineage and useful positive exposure for round029 only."""

from .campaign import velocity_policy
from .contracts import _check, digest, schema, seal
from .metrics import _seed
from .training_rows import matched_batch
from .trials import integer


def recipe_bindings(registration, *, initialization, stream_seed, ordinary, broad, training_start=None):
    for identity in (registration, ordinary, *([broad] if broad is not None else [])):
        _check(schema("common")["$defs"]["hash"], identity, schema("common"))
    _seed(initialization)
    _seed(stream_seed)
    provenance = {}
    if training_start is not None:
        origin = validate_study_start(training_start, initialization, stream_seed, ordinary, broad)
        provenance = dict(origin=origin["identity"], initial_weights=origin["initial_weights"])
    return {
        row["id"]: seal(
            dict(
                version="round029-trial/v1",
                registration=registration,
                recipe=row,
                initialization=initialization,
                stream_seed=stream_seed,
                stream_ref=ordinary if row["stream"] == "ordinary" else broad,
                **provenance,
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


def validate_start(origin):
    from .training_rows import pair_seeds

    fields = {
        "version",
        "identity",
        "seed_authority",
        "pair",
        "initialization",
        "stream_seed",
        "ordinary",
        "broad",
        "initial_weights",
        "common_policy",
        "codec",
    }
    if (
        not isinstance(origin, dict)
        or set(origin) != fields
        or origin != seal(origin)
        or origin["version"] != "round029-initial-state/v1"
    ):
        raise ValueError("original sealed random initialization and streams required")
    for name in ("ordinary", "initial_weights", "common_policy", "codec"):
        _check(schema("common")["$defs"]["hash"], origin[name], {})
    if origin["broad"] is not None:
        _check(schema("common")["$defs"]["hash"], origin["broad"], {})
        if origin["broad"] == origin["ordinary"]:
            raise ValueError("the two treatments have distinct positive-stream receipts")
    seeds = pair_seeds(origin["seed_authority"], origin["pair"])
    if seeds != {
        "initialization": _seed(origin["initialization"]),
        "training_families": _seed(origin["stream_seed"]),
    }:
        raise ValueError("prepared initialization or stream seed differs from original authority")
    return origin


def select_start(inventory, preferred, fallback):
    """Select one seed-one start for the full grid, before any new comparative outcome."""
    origin = validate_start(preferred) if preferred is not None else None
    stream = inventory.get("stream") or {}
    controls = [r for r in inventory["cuts"] if r["method"] == "00"]
    if origin is not None:
        seeds = {
            "initialization": origin["initialization"],
            "training_families": origin["stream_seed"],
        }
        reproduced = False
        for cut in controls:
            if (
                cut.get("initial") != origin["initial_weights"]
                or cut.get("codec") != origin["codec"]
                or cut.get("policy") != origin["common_policy"]
                or not cut.get("lineage") == stream.get("identity") == origin["ordinary"]
                or not cut.get("seeds") == stream.get("seeds") == seeds
            ):
                continue
            if (
                cut["look"] == 0
                and cut.get("updates") == cut.get("useful_tokens") == cut.get("cursor") == 0
            ):
                if cut.get("weights"):
                    return seal(
                        dict(
                            mode="reuse_ordinary_start",
                            origin=origin,
                            control_weights=cut["weights"],
                        )
                    )
                reproduced = True
            elif cut["look"] in (1_000_000, 4_000_000) and cut.get("weights"):
                reproduced = True
        if reproduced:
            # A later optimizer cut can corroborate the random initialization. Its trained
            # weights are never supplied to a treatment starting at zero.
            return seal(
                dict(mode="reconstruct_ordinary_start", origin=origin, control_weights=None)
            )
    if fallback is None:
        raise ValueError(
            "a prepared fresh shared initialization is required for the affected comparison"
        )
    fresh = validate_start(fallback)
    known = {r.get("initial") for r in controls}
    old_seeds = {(r.get("seeds") or {}).get("initialization") for r in controls}
    if (
        fresh["initial_weights"] in known
        or fresh["initialization"] in old_seeds
        or origin
        and fresh["initialization"] == origin["initialization"]
    ):
        raise ValueError("unmatched historical initialization requires fresh shared random weights")
    return seal(dict(mode="fresh_shared_control", origin=fresh, control_weights=None))


def validate_study_start(start, initialization, stream_seed, ordinary, broad):
    if (
        not isinstance(start, dict)
        or set(start) != {"mode", "origin", "control_weights", "identity"}
        or start != seal(start)
        or start["mode"]
        not in {"reuse_ordinary_start", "reconstruct_ordinary_start", "fresh_shared_control"}
    ):
        raise ValueError("sealed training start and ordinary control disposition required")
    origin = validate_start(start["origin"])
    if (initialization, stream_seed, ordinary, broad) != tuple(
        origin[k] for k in ("initialization", "stream_seed", "ordinary", "broad")
    ):
        raise ValueError("study seeds and streams differ from the prepared common start")
    if start["mode"] == "reuse_ordinary_start":
        _check(schema("common")["$defs"]["hash"], start["control_weights"], {})
    elif start["control_weights"] is not None:
        raise ValueError("only the original zero cut can supply zero-look weights")
    return origin
