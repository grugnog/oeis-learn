"""Custodian projection and installed acceptance of round029 visible-only CPU inputs."""

import os
from pathlib import Path

from oeis_learn.experiments.source_snapshot import source_snapshot
from oeis_learn.generalization.progressive.contracts import _check, atomic_write, digest, load, schema, seal
from oeis_learn.generalization.progressive.projections import project, remove_projection
from oeis_learn.generalization.progressive.velocity_records import registered_worker, visible_prompt
from .identity import confined, validate
from .progressive_roles import ROUND029_ROLES, ROUND029_ROLE_POLICY


LIVE_CHANNELS = {"visible": {"round029_visible"}, "guide": {"round029_guide"},
                 "admissions": {"round029_admission"}}


def validate_live_value(kind, data):
    if kind == "round029_visible":
        return visible_prompt(data)
    rules = schema("common")["$defs"]
    if kind == "round029_guide" and set(data) == {"guide_ref", "prompt"}:
        _check(rules["hash"], data["guide_ref"], {})
        visible_prompt(data["prompt"])
    elif kind == "round029_admission" and set(data) == {"ordinal", "source_ref"}:
        _check(rules["integer"], data["ordinal"], {})
        _check(rules["hash"], data["source_ref"], {})
    else:
        raise ValueError("invalid typed live channel value")
    return data


def _check_request(request, worker):
    validate(request, {"version", "operation", "source_manifest", "worker", "channels"}, "round029-live-input/v1")
    operation = request["operation"]
    if operation not in {"search", "select"}:
        raise ValueError("registered live CPU operation required")
    registered_worker(request["worker"], worker)
    if worker["device"] != "cpu" or worker["role_policy"] != ROUND029_ROLE_POLICY:
        raise ValueError("registered CPU worker policy required")
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
        if channel == "visible" and len(identities) != 1 or channel == "guide" and len(identities) > 1:
            raise ValueError("one registered case and at most one guide per request")
    return channels


def project_live_inputs(root, destination, *, request, current_worker, artifacts, authorized, forbidden=()):
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
                        channel_types=LIVE_CHANNELS, authorized=authorized, forbidden=forbidden)
        atomic_write(dest / "authority.json", seal(dict(version="round029-live-supervisor/v1",
                     request=request, current_worker=current_worker)))
        return views
    except BaseException:
        remove_projection(dest)
        raise


def open_live_inputs(operation, inputs, authority):
    """Reopen only registered files at the installed boundary before starting any role work."""
    control = validate(load(authority), {"version", "request", "current_worker"}, "round029-live-supervisor/v1")
    request = control["request"]
    # The controller inspected the candidate checkout and physical image. Independently
    # identify this installed package, CPU role and supervisor-provided container image.
    registered_worker(request["worker"], control["current_worker"])
    observed = dict(control["current_worker"], image=os.environ.get("OEIS_QUALIFICATION_IMAGE"),
                    device="cpu", role_policy=ROUND029_ROLE_POLICY)
    channels = _check_request(request, observed)
    if request["operation"] != operation or request["source_manifest"] != digest(source_snapshot()):
        raise ValueError("installed source or operation differs from registered worker")
    root = Path(inputs)
    if root.is_symlink() or {p.name for p in root.iterdir()} != set(channels):
        raise ValueError("unexpected installed live channel")
    values = {}
    for channel, identities in channels.items():
        folder = confined(root, channel)
        names = [identity[7:] + ".json" for identity in identities]
        if {p.name for p in folder.iterdir()} != set(names):
            raise ValueError("unexpected file in installed live channel")
        values[channel] = []
        for name, identity in zip(names, identities):
            value = validate(load(confined(folder, name)), {"version", "role", "kind", "data"}, "discovery-channel/v1")
            if (value["identity"] != identity or value["role"] != ROUND029_ROLES[operation].role
                    or value["kind"] not in LIVE_CHANNELS[channel]):
                raise ValueError("installed live envelope differs from registered channel")
            values[channel].append(validate_live_value(value["kind"], value["data"]))
    return request["identity"], values


def accept_live_inputs(operation, *, inputs, authority, output):
    request, values = open_live_inputs(operation, inputs, authority)
    accepted = seal(dict(version="round029-accepted-input/v1", scope="live_input_projection",
                         operation=operation, request_ref=request,
                         channel_counts={k: len(v) for k, v in values.items()}))
    atomic_write(Path(output) / "accepted-input.json", accepted)
    return accepted
