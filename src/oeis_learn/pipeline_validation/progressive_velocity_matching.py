"""Freeze one training-only calibration before comparative looks and bind it to both matching arms."""

import math

from oeis_learn.generalization.budgets import integer
from oeis_learn.generalization.progressive.campaign import validate_velocity_registration
from oeis_learn.generalization.progressive.contracts import _check, digest, load, schema, seal
from oeis_learn.generalization.progressive.shared_support import validated_rows
from oeis_learn.generalization.progressive.training_rows import matched_batch, matching_pairs
from oeis_learn.generalization.progressive.velocity_trials import update_policy
from .identity import confined
from .preparation_recovery import retain


def check_projection(request):
    from oeis_learn.decoder.bigint_codec import validate_body
    from oeis_learn.synthesis.contracts import canonical_integer

    if (
        set(request)
        != {
            "version",
            "plan",
            "pair_proof",
            "profile",
            "order",
            "rows",
            "donors",
            "initial_weights",
            "identity",
        }
        or request != seal(request)
        or request["version"] != "round029-matching-projection/v1"
    ):
        raise ValueError("closed original training projection required")
    for k in ("plan", "pair_proof", "initial_weights"):
        _check(schema("common")["$defs"]["hash"], request[k], {})
    order, rows, donors = request["order"], request["rows"], request["donors"]
    if (
        not order
        or len(order) % 32
        or set(order) != rows.keys()
        or not donors.keys() <= rows.keys()
    ):
        raise ValueError("complete ordinary calibration batches required")
    for k, row in rows.items():
        _check(schema("common")["$defs"]["hash"], k, {})
        if set(row) != {"values", "tokens"}:
            raise ValueError(
                "only training prefixes and their positive decoder targets are permitted"
            )
        validate_body(row["tokens"])
        for values in [row["values"], *([donors[k]] if k in donors else [])]:
            if not isinstance(values, list) or len(values) != 20:
                raise ValueError("twenty exposed prefix values required")
            for v in values:
                canonical_integer(v)
    return request


def qualified_pairs(journal, stream, role):
    """Recover the registered stream's exact labels from its paid private qualification."""
    from . import progressive_velocity_broad_host as broad

    folder = journal.root.parent.parent
    fresh = (folder / broad.FRESH_MARKER).is_file()
    pointer = broad.FRESH_MARKER if fresh else broad.MARKER
    if not (folder / pointer).is_file():
        return None
    marker = load(confined(folder, pointer))
    area = confined(folder, "qualification/" + marker["grant"][7:])
    root = folder.parents[2]
    binding = load(confined(area, "binding.json"))
    source = confined(root, "run-state/discovery-tools/round028-source/" + binding["candidate"])
    receipt = broad.inventory(root, source, fresh=True) if fresh else broad.inventory(root, source)
    support = ((receipt.get("supply") or {}).get("matching") or {}).get(role)
    if support is None or support["stream"] != stream["identity"]:
        return None
    grant = load(confined(area, "grant.json"))
    native = broad.attempt(area, broad.coordinates(grant, "prepare"), grant)
    path = confined(native, "native/pairs.json")
    pairs = load(path, maximum=path.stat().st_size)[role]
    if pairs != seal(pairs) or pairs["identity"] != support["manifest"]:
        raise ValueError("stored training pairs differ from their paid workload")
    return pairs


def prepare_calibration(
    journal, registration, profile, stream, shards, *, batches, client=None, pair_manifest=None
):
    """Commit the outcome-blind ordinary slots before fitting labels or exposing their worker view."""
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.generalization.progressive.models import validate_profile

    reg = validate_velocity_registration(registration)
    origin = reg["training_start"]["origin"]
    rows = validated_rows(shards, native_verified=True)
    order = stream["order"]
    if (
        stream != seal(stream)
        or stream["identity"] != origin["ordinary"]
        or not order
        or profile != validate_profile(profile)
        or profile["model"]["seed"] != origin["initialization"]
        or not set(order) <= rows.keys()
        or digest([rows[k]["identity"] for k in order]) != stream["row_identities"]
    ):
        raise ValueError(
            "matching calibration requires the frozen ordinary stream and initialization"
        )
    batches = integer(batches, 1)
    # Cover at most one traversal of unique support. Complete its last logical batch just as the learner does.
    count = 32 * min(batches, (len(order) + 31) // 32)
    keys = matched_batch(stream, 0, count)
    plan = seal(
        dict(
            version="round029-matching-plan/v1",
            registration=reg["identity"],
            origin=origin["identity"],
            profile=profile["identity"],
            ordinary=stream["identity"],
            first_rows=digest(keys),
            batches=batches,
            source=digest(source_snapshot()),
            runtime=client.runtime_identity("bigint-body/v1") if client is not None else None,
            reference=client.reference_sha256 if client is not None else None,
        )
    )
    paid = qualified_pairs(journal, stream, "ordinary")
    if pair_manifest is not None and paid is not None and pair_manifest != paid:
        raise ValueError("calibration pairs differ from the priced ordinary workload")
    pair_manifest = paid if paid is not None else pair_manifest
    area = journal.root
    with journal.lock():
        if reg != load(confined(area, "velocity-registration.json")):
            raise ValueError("calibration belongs to another study")
        path = confined(area, "matching-plan.json", exists=False)
        projection = confined(area, "matching-projection.json", exists=False)
        if not path.is_file() and any(area.glob("look-*.json")):
            raise ValueError("matching calibration must be frozen before comparative outcomes")
        retain(path, plan)
        if not projection.exists():
            unique = list(dict.fromkeys(keys))
            saved = confined(area, "matching-pairs.json", exists=False)
            if saved.is_file() and pair_manifest is None:
                pair_manifest = load(saved, maximum=saved.stat().st_size)
            pairs = matching_pairs(
                shards, order, seed=origin["stream_seed"], client=client, recorded=pair_manifest
            )
            retain(area / "matching-pairs.json", pairs)
            projection_data = seal(
                dict(
                    version="round029-matching-projection/v1",
                    plan=plan["identity"],
                    pair_proof=pairs["identity"],
                    profile=profile,
                    initial_weights=origin["initial_weights"],
                    order=keys,
                    rows={
                        k: dict(values=rows[k]["values"][:20], tokens=rows[k]["tokens"])
                        for k in unique
                    },
                    donors={
                        k: rows[v["donor"]]["values"][:20]
                        for k, v in pairs["pairs"].items()
                        if k in unique
                    },
                )
            )
            retain(projection, check_projection(projection_data))
        request = check_projection(load(projection, maximum=projection.stat().st_size))
        pairs_path = confined(area, "matching-pairs.json")
        pairs = load(pairs_path, maximum=pairs_path.stat().st_size)
        if (
            pairs != seal(pairs)
            or (pair_manifest is not None and pairs != pair_manifest)
            or pairs["identity"] != request["pair_proof"]
            or pairs["seed"] != origin["stream_seed"]
            or pairs["order"] != digest(order)
            or request["plan"] != plan["identity"]
            or request["profile"] != profile
            or request["order"] != keys
            or request["initial_weights"] != origin["initial_weights"]
            or any(
                request["rows"][k] != dict(values=rows[k]["values"][:20], tokens=rows[k]["tokens"])
                for k in set(keys)
            )
            or request["donors"]
            != {
                k: rows[p["donor"]]["values"][:20]
                for k, p in pairs["pairs"].items()
                if k in request["rows"]
            }
        ):
            raise ValueError("retained matching labels or learner projection changed")
        return request


def _verified(observation, request):
    check_projection(request)
    fields = {
        "version",
        "identity",
        "request",
        "initial_weights",
        "positive_rows",
        "eligible_rows",
        "wins",
        "extra_passes",
        "median_positive",
        "median_auxiliary",
        "weight",
        "reason",
        "elapsed_us",
        "device",
    }
    order, donors = request["order"], request["donors"]
    support = sum(k in donors for k in order)
    passes = sum(any(k in donors for k in order[i : i + 32]) for i in range(0, len(order), 32))
    if (
        set(observation) != fields
        or observation != seal(observation)
        or observation["version"] != "round029-matching-observation/v1"
        or observation["request"] != request["identity"]
        or observation["initial_weights"] != request["initial_weights"]
        or integer(observation["positive_rows"], 1) != len(order)
        or integer(observation["eligible_rows"]) != support
        or integer(observation["wins"]) > support
        or integer(observation["extra_passes"]) != passes
        or not integer(observation["elapsed_us"], 1)
        or observation["device"] not in {"cpu", "cuda"}
    ):
        raise ValueError("matching observation conflicts with the original calibration projection")
    positive, auxiliary = observation["median_positive"], observation["median_auxiliary"]
    for value in (positive, auxiliary):
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("invalid measured loss")
    if not support and auxiliary is not None:
        raise ValueError("auxiliary loss requires eligible pairs")
    expected = (
        positive * 0.25 / auxiliary
        if support and positive is not None and auxiliary is not None
        else None
    )
    if expected is not None and (not math.isfinite(expected) or expected <= 0):
        expected = None
    reason = (
        None
        if expected is not None
        else (
            "nonfinite or nonpositive training calibration"
            if support
            else "no eligible training pairs"
        )
    )
    weight = observation["weight"]
    if (
        (weight is not None and type(weight) not in (int, float))
        or weight != expected
        or observation["reason"] != reason
    ):
        raise ValueError("matching weight differs from the measured quarter-loss rule")
    return observation


def commit_calibration(journal, request, observation):
    with journal.lock():
        area = journal.root
        plan = load(confined(area, "matching-plan.json"))
        path = confined(area, "matching-projection.json")
        if (
            plan != seal(plan)
            or plan["registration"] != journal.binding["study"]
            or request != load(path, maximum=path.stat().st_size)
            or request["plan"] != plan["identity"]
        ):
            raise ValueError("original matching registration and training-only projection required")
        final = confined(area, "matching-calibration.json", exists=False)
        if not final.exists() and any(area.glob("look-*.json")):
            raise ValueError("freeze the common matching weight before comparative outcomes")
        retain(final, _verified(observation, request))
        return observation


def registered_update_policy(journal, profile, method, *, microbatch):
    """Resolve either matching arm from the same paid observation in the original study."""
    from oeis_learn.generalization.progressive.training_rows import generator_common_policy
    from .progressive_training import resolve_update_policy

    binding = journal.velocity_trial(method)["binding"]
    reg = validate_velocity_registration(load(confined(journal.root, "velocity-registration.json")))
    origin = reg["training_start"]["origin"]
    policy = resolve_update_policy(
        profile,
        objective="ordinary",
        effective_batch=32,
        microbatch_size=microbatch,
        weight_decay=0,
        schedule_horizon_tokens=4_000_000,
    )
    if (
        profile["model"]["seed"] != reg["initialization"]
        or profile["codec"] != origin["codec"]
        or generator_common_policy(profile, policy) != origin["common_policy"]
    ):
        raise ValueError("learner profile differs from the prepared shared start")
    weight = None
    if binding["recipe"]["matching"]:
        plan = load(confined(journal.root, "matching-plan.json"))
        path = confined(journal.root, "matching-projection.json")
        request = load(path, maximum=path.stat().st_size)
        pairs_path = confined(journal.root, "matching-pairs.json")
        pairs = load(pairs_path, maximum=pairs_path.stat().st_size)
        if (
            pairs != seal(pairs)
            or pairs["identity"] != request["pair_proof"]
            or not pairs.get("coverage", {}).get("full")
        ):
            raise ValueError(
                "eligible matching treatment needs its full registered calibration scope"
            )
        if (
            plan != seal(plan)
            or plan["registration"] != reg["identity"]
            or plan["origin"] != origin["identity"]
            or request["plan"] != plan["identity"]
            or request["profile"] != profile
        ):
            raise ValueError("matching calibration belongs to another study, stream or model")
        obs = _verified(load(confined(journal.root, "matching-calibration.json")), request)
        if obs["weight"] is None:
            raise ValueError(obs["reason"])
        weight = obs["weight"]
    return update_policy(profile, method, microbatch=microbatch, matching_weight=weight)
