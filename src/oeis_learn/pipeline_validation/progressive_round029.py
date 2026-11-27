"""Standard-library host entry point for the round029 study on the physical donor."""

import fcntl
import os
from pathlib import Path

from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest, load
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.program_search_costing import original_probe_costs, price_wave
from oeis_learn.generalization.progressive.velocity_records import (
    SESSION,
    accepted_grant_parent,
    checked,
    physical_root_id,
    validate_grant,
    validate_result,
)
from .identity import confined
from .progressive_round028 import candidate_revision
from .progressive_velocity_donor import inventory
from .progressive_velocity_supply import supply_inventory


def result(action, status, reason, next_action, *, last=None, remaining=None, **data):
    return validate_result(
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
            **data,
        )
    )


def preflight(root, source=None):
    policy = velocity_policy()
    info = inventory(root)
    if source is not None:
        from .progressive_velocity_verify import donor_continuation

        folder = confined(root, "run-state/discovery/" + SESSION, exists=False)
        cut = next(r for r in info["cuts"] if (r["method"], r["look"]) == ("00", 4_000_000))
        try:
            parity = donor_continuation(root, folder, Path(source), cut)
            if parity is not None:
                cut.update(
                    continuation="verified_disposable_next_update",
                    parity=parity,
                    permitted_roles=["generate", "disposable_restore", "continue_ordinary"],
                )
        except (ValueError, OSError, KeyError) as exc:
            cut["continuation_reason"] = str(exc)
    from .progressive_velocity_reuse import add_look_costs, reuse_inventory

    add_look_costs(info["cuts"], info["original_clocks"])
    try:
        info["reuse"] = reuse_inventory(root, info["cuts"])
    except (ValueError, OSError, KeyError) as exc:
        info["reuse"] = dict(
            measurements=[],
            distinct_score_credits=0,
            research_dispatches=0,
            registration=None,
            reason=str(exc),
        )
    stat = os.statvfs(root)
    info["mount"] = dict(
        device=root.stat().st_dev,
        available_bytes=stat.f_bavail * stat.f_frsize,
        available_inodes=stat.f_favail,
        required_bytes=None,
        required_inodes=None,
        reason="pending measured peak and operator reserve for this scope",
    )
    from .progressive_research_resources import velocity_capacity

    try:
        info["capacity"] = velocity_capacity(root)
    except (ValueError, OSError, KeyError) as exc:
        info["capacity"] = dict(mounts=None, registration=None, reason=str(exc))
    for mount in info["capacity"]["mounts"] or ():
        if mount["device"] == info["mount"]["device"]:
            info["mount"].update(
                {
                    k: mount[k]
                    for k in (
                        "available_bytes",
                        "available_inodes",
                        "required_bytes",
                        "required_inodes",
                        "reserved_bytes",
                        "reserved_inodes",
                        "deficit_bytes",
                        "deficit_inodes",
                    )
                }
            )
            info["mount"][
                "reason"
            ] = "registered simultaneous owner peaks and physical mount reserve"
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
    info["prices"]["original_028_workloads"] = original_probe_costs(
        info["cuts"], info["historical_scores"], info["original_clocks"]
    )
    info["new_positive_tokens_before_overshoot_if_control_reusable"] = 24_000_000
    info.update(supply_inventory(root))
    info["assessment"] = policy["assessment"]
    ordinary = next(r for r in info["cuts"] if (r["method"], r["look"]) == ("00", 4_000_000))
    if any(m["deficit_bytes"] or m["deficit_inodes"] for m in info["capacity"]["mounts"] or ()):
        reason, next_action = (
            "Registered simultaneous writes exceed a physical mount reserve",
            "inspect_capacity",
        )
    elif ordinary["continuation"] == "requires_next_update_parity":
        reason, next_action = (
            "The ordinary 4M full state is available for one disposable next-update comparison",
            "qualify",
        )
    elif not ordinary["weights"]:
        reason, next_action = (
            "Original ordinary weights and training provenance are unavailable for the proposed reuse",
            "inspect_inputs",
        )
    else:
        reason, next_action = (
            "Price the common probe and a protected follow-up on exposed inputs",
            "qualify",
        )
    return result(
        "preflight",
        "INSPECTED",
        reason,
        next_action,
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
            from .progressive_velocity_verify import archive_result

            return archive_result(args.archive, args.qualification_archive)
        root = Path(args.workspace)
        if not root.is_absolute() or root != root.resolve(strict=True):
            raise ValueError("absolute physical donor workspace required")
        folder = confined(root, "run-state/discovery/" + SESSION, exists=False)
        if action == "preflight":
            return preflight(root, source), 0
        if action == "status":
            return inspect_status(folder), 0
        if not funded:
            from .progressive_velocity_verify import private_result

            return private_result(root, folder)
        grant = checked("grant", load(args.authority))
        authorizer = confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
        policy = velocity_policy()
        revision = candidate_revision(Path(source), root)
        from .progressive_velocity_qualification import qualification_parent

        from .progressive_velocity_donation import donation_parent

        claimed = grant["registration_or_parent_ref"]
        donation = action == "qualify" and claimed == donation_parent(revision)
        parent = None
        if action == "qualify":
            parent = (
                claimed
                if donation or claimed == qualification_parent(revision, children=True)
                else qualification_parent(revision)
            )
        registration = folder / "accepted.json"
        if parent is None and registration.exists():
            parent = accepted_grant_parent(load(confined(folder, "accepted.json")), grant["scope"])
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
                if donation:
                    from .progressive_velocity_donation import qualify
                else:
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
            first_error=dict(error=type(exc).__name__, stage=action, reason=str(exc) or type(exc).__name__),
            **({"verification_scope": "unverified_inputs"} if action == "verify" else {}),
        ), 1
