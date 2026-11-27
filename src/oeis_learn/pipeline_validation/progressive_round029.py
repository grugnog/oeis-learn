"""Standard-library host entry point for the round029 study on the physical donor."""

import fcntl
import os
from pathlib import Path

from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest, load
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.program_search_costing import price_wave
from oeis_learn.generalization.progressive.velocity_records import (
    SESSION,
    checked,
    physical_root_id,
    validate_grant,
    validate_result,
)
from .identity import confined
from .progressive_round028 import candidate_revision
from .progressive_velocity_donor import inventory


def result(action, status, reason, next_action, *, last=None, remaining=None, **data):
    header = validate_result(
        dict(
            version="round029-result/v1",
            action=action,
            session_id=SESSION,
            mechanical_status=status,
            scientific_verdict=None,
            last_committed_ref=last,
            remaining_authority=remaining,
            next_action=next_action,
            reason=reason,
        )
    )
    return {**header, **data}


def preflight(root):
    policy = velocity_policy()
    info = inventory(root)
    stat = os.statvfs(root)
    info["mount"] = dict(
        device=root.stat().st_dev,
        available_bytes=stat.f_bavail * stat.f_frsize,
        available_inodes=stat.f_favail,
        required_bytes=None,
        required_inodes=None,
        reason="pending measured peak and operator reserve for this scope",
    )
    info["prices"] = price_wave(
        {
            "common_probe": {"cold_start": 1, "common_probe": 1, "selected_scoring": 1},
            "wave": {
                "six_new_4M_arms": 1,
                "ordinary_control_shortfall": 1,
                "probes_and_endpoints": 1,
                "auxiliary_passes": 1,
                "publication": 1,
            },
            "followup": {"matched_oeis_measurement": 1, "retention_and_verification": 1},
        },
        {},
    )
    info["new_positive_tokens_before_overshoot_if_control_reusable"] = 24_000_000
    info["families"] = dict(
        count=None, reason="requires a reconciled census of donor and subsequent exposure"
    )
    info["matching_pairs"] = dict(count=None, reason="requires a training-only pair rehearsal")
    info["assessment"] = policy["assessment"]
    return result(
        "preflight",
        "INSPECTED",
        "Inventory read; candidate-specific measurements pending",
        "qualify",
        inventory=info,
        state_root_id=physical_root_id(root),
        qualification_policy=digest(policy),
    )


def inspect_status(folder):
    busy = False
    path = folder / "run.lock"
    if path.is_file():
        with confined(folder, "run.lock").open() as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                busy = True
    status = folder / "status.json"
    if status.exists():
        from .identity import seal

        value = load(confined(folder, "status.json"))
        if value != seal(value) or value.get("session_id") != SESSION:
            raise ValueError("saved status differs")
        return result(
            "status",
            "INSPECTED",
            value["reason"],
            value["next_action"],
            last=value.get("last_committed_ref"),
            remaining=value.get("remaining_authority"),
            owner_active=busy,
            progress=value,
        )
    return result(
        "status",
        "INSPECTED",
        "No committed round029 work",
        "preflight",
        owner_active=busy,
        stage="unregistered",
        recipe=None,
        seed=None,
        look=None,
        useful_tokens=0,
        charged_tokens=0,
        elapsed_us=None,
        telemetry="unavailable",
    )


def execute(args, source, *, host_factory=None):
    action = args.action
    try:
        funded = action in {"qualify", "run", "cpu"}
        if bool(args.authority) != funded:
            raise ValueError("only qualify, cpu and run require an operator grant")
        if args.archive or args.qualification_archive:
            if not (
                action == "verify"
                and args.archive
                and args.qualification_archive
                and not args.workspace
            ):
                raise ValueError(
                    "independent verify requires both the report and its pre-score archive"
                )
            return result(
                action,
                "BLOCKED",
                "Independent round029 archive verification is pending implementation",
                "status",
            ), 2
        root = Path(args.workspace)
        if not root.is_absolute() or root != root.resolve(strict=True):
            raise ValueError("absolute physical donor workspace required")
        folder = confined(root, "run-state/discovery/" + SESSION, exists=False)
        if action == "preflight":
            return preflight(root), 0
        if action == "status":
            return inspect_status(folder), 0
        if not funded:
            return result(
                action,
                "BLOCKED",
                "Private round029 publication replay is pending implementation",
                "status",
            ), 2
        grant = checked("grant", load(args.authority))
        authorizer = confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
        policy = velocity_policy()
        revision = candidate_revision(Path(source), root)
        from .progressive_velocity_qualification import qualification_parent

        parent = qualification_parent(revision) if action == "qualify" else None
        registration = folder / "accepted.json"
        if parent is None and registration.exists():
            parent = load(confined(folder, "accepted.json"))["identity"]
        validate_grant(
            grant, load(authorizer), root=physical_root_id(root), parent=parent, action=action
        )
        from .progressive_round028 import git

        if (
            git(root, "rev-parse", "HEAD") != policy["donor_revision"]
            or git(root, "symbolic-ref", "-q", "HEAD") is not None
        ):
            raise ValueError("qualified detached donor required for funded actions")
        folder.mkdir(parents=True, exist_ok=True)
        with RunStore.owner(folder):
            if action == "qualify":
                from .progressive_velocity_qualification import qualify

                return qualify(args, root, folder, source, revision, grant, host_factory)
            return result(
                action,
                "BLOCKED",
                "Accepted pre-score commitment for this scope required",
                "qualify",
            ), 2
    except Exception as exc:
        return result(
            action,
            "FAILED",
            str(exc) or type(exc).__name__,
            "inspect_inputs",
            first_error=type(exc).__name__,
        ), 1
