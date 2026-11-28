"""Custodian projections and independently checked inputs for installed round029 roles."""

import os
from pathlib import Path

from oeis_learn.experiments.source_snapshot import source_snapshot
from oeis_learn.generalization.progressive.contracts import _check, atomic_write, digest, load, schema, seal
from oeis_learn.generalization.progressive.projections import project, remove_projection
from oeis_learn.generalization.progressive.velocity_records import registered_worker, visible_prompt
from .identity import confined, validate
from .progressive_roles import ROUND029_ROLES, ROUND029_ROLE_POLICY


CHANNEL_KINDS = {
    "visible": {"round029_visible"}, "guide": {"round029_guide"},
    "admissions": {"round029_admission"}, "model": {"round029_model"},
    "training": {"round029_training"}, "skeleton": {"round029_skeleton"},
    "choices": {"round029_choice"}, "sealed_bank": {"round029_bank"},
    "score_truth": {"round029_truth"},
}


def validate_role_value(kind, data):
    from oeis_learn.decoder.bigint_codec import encode_body, validate_body
    from oeis_learn.synthesis.contracts import canonical_integer

    if not isinstance(data, dict):
        raise ValueError("closed role input object required")
    if kind == "round029_visible":
        return visible_prompt(data)
    rules = schema("common")["$defs"]

    def hash_ref(value):
        _check(rules["hash"], value, {})

    if kind == "round029_guide" and set(data) == {"guide_ref", "prompt"}:
        hash_ref(data["guide_ref"])
        visible_prompt(data["prompt"])
    elif kind == "round029_admission" and set(data) == {"ordinal", "source_ref"}:
        _check(rules["integer"], data["ordinal"], {})
        hash_ref(data["source_ref"])
    elif kind == "round029_training" and set(data) == {"prompt", "target", "family_ref"}:
        visible_prompt(data["prompt"])
        validate_body(data["target"])
        hash_ref(data["family_ref"])
    elif kind == "round029_model" and set(data) == {"weights_ref", "profile_ref"}:
        hash_ref(data["weights_ref"])
        hash_ref(data["profile_ref"])
    elif kind == "round029_skeleton" and set(data) == {"tokens"}:
        validate_body(data["tokens"])
    elif kind == "round029_truth" and set(data) == {"indices", "values"}:
        if any(not isinstance(v, list) or len(v) != 100 for v in data.values()):
            raise ValueError("100 indexed terms required by the private scorer")
        indices = [canonical_integer(n) for n in data["indices"]]
        if indices != list(range(indices[0], indices[0] + 100)):
            raise ValueError("consecutive original indices required")
        for value in data["values"]:
            canonical_integer(value)
    elif kind == "round029_choice" and set(data) == {"prompt_ref", "source", "closed", "complete"}:
        hash_ref(data["prompt_ref"])
        if type(data["closed"]) is not bool or type(data["complete"]) is not bool:
            raise ValueError("explicit choice and portfolio completion required")
        if data["source"] is not None:
            validate_body(encode_body(data["source"]))
    elif kind == "round029_bank" and set(data) == {"prompt_ref", "choice_ref", "sources", "closed"}:
        hash_ref(data["prompt_ref"])
        hash_ref(data["choice_ref"])
        if type(data["closed"]) is not bool or not isinstance(data["sources"], list):
            raise ValueError("explicit bank closure and ordered admissions required")
        for source in data["sources"]:
            validate_body(encode_body(source))
    else:
        raise ValueError("invalid typed role channel value")
    return data


def _check_request(request, worker):
    version = request.get("version")
    if version not in {"round029-live-input/v1", "round029-role-input/v1"}:
        raise ValueError("unknown round029 request type")
    validate(request, {"version", "operation", "source_manifest", "worker", "channels"}, version)
    operation = request["operation"]
    if operation not in ROUND029_ROLES or (version == "round029-live-input/v1" and operation not in {"search", "select"}):
        raise ValueError("registered live CPU operation required")
    registered_worker(request["worker"], worker)
    device = "hip" if ROUND029_ROLES[operation].gpu else "cpu"
    if worker["device"] != device or worker["role_policy"] != ROUND029_ROLE_POLICY:
        raise ValueError("registered device and role policy required")
    hashes = schema("common")["$defs"]["hash"]
    _check(hashes, request["source_manifest"], {})
    channels = request["channels"]
    if not isinstance(channels, dict) or set(channels) != set(ROUND029_ROLES[operation].channels):
        raise ValueError("incorrect registered channel set")
    for channel, identities in channels.items():
        if not isinstance(identities, list) or len(set(identities)) != len(identities):
            raise ValueError("unique ordered channel identities required")
        for identity in identities:
            _check(hashes, identity, {})
        if channel == "guide" and len(identities) > 1:
            raise ValueError("at most one frozen guide per request")
        if channel == "training" and not identities:
            raise ValueError("training-only rows required")
        if channel not in {"guide", "training", "admissions"} and len(identities) != 1:
            raise ValueError("one registered object per role channel required")
    return channels


def project_role_inputs(root, destination, *, request, current_worker, artifacts, authorized, forbidden=()):
    """The supervisor supplies the observed worker and the independent frozen artifact allowlist."""
    channels = _check_request(request, current_worker)
    if authorized is None or {c: [a.identity for a in items] for c, items in artifacts.items()} != channels:
        raise ValueError("live artifacts differ from frozen registration")
    operation = request["operation"]
    if any(a.role != ROUND029_ROLES[operation].role for items in artifacts.values() for a in items):
        raise ValueError("live artifact belongs to another role")
    dest = confined(root, destination, exists=False)
    dest.mkdir(parents=True, exist_ok=False)
    try:
        views = project(root, dest / "inputs", operation, artifacts, registry=ROUND029_ROLES,
                        channel_types=CHANNEL_KINDS, authorized=authorized, forbidden=forbidden)
        atomic_write(dest / "authority.json", seal(dict(version="round029-live-supervisor/v1",
                     request=request, current_worker=current_worker)))
        return views
    except BaseException:
        remove_projection(dest)
        raise


def role_mounts(projection, request):
    """Build file-only binds for the shared socket-free stage supervisor."""
    channels = _check_request(request, request["worker"])
    return [(confined(projection, "authority.json"), "/work/inputs/authority.json"), *[
        (confined(projection, f"inputs/{channel}/{ref[7:]}.json"), f"/work/inputs/channels/{channel}/{ref[7:]}.json")
        for channel, refs in channels.items() for ref in refs
    ]]


def open_role_inputs(operation, inputs, authority):
    """Reopen only registered files at the installed boundary before starting any role work."""
    control = validate(load(authority), {"version", "request", "current_worker"}, "round029-live-supervisor/v1")
    request = control["request"]
    # The controller inspected the candidate checkout and physical image. Independently
    # identify this installed package, CPU role and supervisor-provided container image.
    registered_worker(request["worker"], control["current_worker"])
    observed = dict(control["current_worker"], image=os.environ.get("OEIS_QUALIFICATION_IMAGE"),
                    device="hip" if ROUND029_ROLES[operation].gpu else "cpu", role_policy=ROUND029_ROLE_POLICY)
    channels = _check_request(request, observed)
    if request["operation"] != operation or request["source_manifest"] != digest(source_snapshot()):
        raise ValueError("installed source or operation differs from registered worker")
    root = Path(inputs)
    present = {p.name for p in root.iterdir()}
    required = {name for name, refs in channels.items() if refs}
    if root.is_symlink() or not required <= present <= set(channels):
        raise ValueError("unexpected installed live channel")
    values = {}
    for channel, identities in channels.items():
        if channel not in present:
            values[channel] = []
            continue
        folder = confined(root, channel)
        names = [identity[7:] + ".json" for identity in identities]
        if {p.name for p in folder.iterdir()} != set(names):
            raise ValueError("unexpected file in installed live channel")
        values[channel] = []
        for name, identity in zip(names, identities):
            value = validate(load(confined(folder, name)), {"version", "role", "kind", "data"}, "discovery-channel/v1")
            if (value["identity"] != identity or value["role"] != ROUND029_ROLES[operation].role
                    or value["kind"] not in CHANNEL_KINDS[channel]):
                raise ValueError("installed live envelope differs from registered channel")
            values[channel].append(validate_role_value(value["kind"], value["data"]))
    if ROUND029_ROLES[operation].gpu:
        from .progressive_qualification import require_hip
        require_hip()
    return request["identity"], values


def accept_role_inputs(operation, *, inputs, authority, output):
    request, values = open_role_inputs(operation, inputs, authority)
    accepted = seal(dict(version="round029-accepted-input/v1", scope=("live_input_projection" if operation in {"search", "select"} else "role_input_projection"),
                         operation=operation, request_ref=request,
                         channel_counts={k: len(v) for k, v in values.items()}))
    atomic_write(Path(output) / "accepted-input.json", accepted)
    return accepted


# Original versioned live requests remain readable by existing callers.
project_live_inputs = project_role_inputs
open_live_inputs = open_role_inputs
validate_live_value = validate_role_value
accept_live_inputs = accept_role_inputs
