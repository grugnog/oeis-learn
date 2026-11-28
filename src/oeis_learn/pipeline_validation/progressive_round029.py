"""Standard-library host entry point for the round029 study on the physical donor."""

import fcntl
import os
import time
from pathlib import Path

from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest, load
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.program_search_costing import velocity_forecast
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


def preflight(root, source=None, *, free_disk_gib=5):
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
        info["reuse"] = reuse_inventory(root, info["cuts"], info["original_clocks"])
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
    info["new_positive_tokens_before_overshoot_if_control_reusable"] = 24_000_000
    info.update(supply_inventory(root))
    from .progressive_velocity_timing_host import timing_inventory

    measured = None
    try:
        measured = timing_inventory(root, source)
        if measured is not None:
            info["prices"]["timed_updates"] = measured
    except (OSError, ValueError, KeyError) as exc:
        info["prices"]["timing_reason"] = str(exc)
    from .progressive_velocity_broad_host import inventory as broad_inventory

    try:
        broader = broad_inventory(root, source)
        if broader is not None and measured is not None:
            if broader["ordinary_measurement"] != measured["measurement"]:
                raise ValueError("broader timing belongs to a different ordinary calibration")
            info["prices"]["broader_updates"] = broader
            if broader["disposition"] == "requires_fresh_disjoint_start":
                fresh = broad_inventory(root, source, fresh=True)
                if fresh is not None:
                    if (
                        fresh["overlap"] != broader["identity"]
                        or fresh["ordinary_measurement"] != measured["measurement"]
                    ):
                        raise ValueError(
                            "fresh timing belongs to a different disjointness decision"
                        )
                    info["prices"]["fresh_start"] = fresh
    except (OSError, ValueError, KeyError) as exc:
        info["prices"]["broader_reason"] = str(exc)
    info["prices"] = velocity_forecast(info, policy)
    if info["prices"].get("fresh_start"):
        from oeis_learn.generalization.progressive.program_search_costing import supported_velocity_wave
        from .progressive_velocity_checkpoint_host import inventory as checkpoint_inventory

        try:
            began = time.monotonic_ns()
            cycles = checkpoint_inventory(root, source)
            read_us = (time.monotonic_ns() - began) // 1000
            if cycles is not None:
                if cycles["origin"] != info["prices"]["fresh_start"]["identity"]:
                    raise ValueError("checkpoint cycles belong to a different fresh stream")
                info["prices"]["checkpoint_cycles"] = cycles
                supported = supported_velocity_wave(policy, info["prices"], info["prices"]["fresh_start"], cycles)
                if supported is not None:
                    info["prices"]["supported_wave"] = supported
                    from .progressive_velocity_launch import launch_preflight

                    info["prices"]["launch_preparation"] = launch_preflight(
                        root, policy, info, floor=free_disk_gib << 30, checkpoint_read_us=read_us
                    )
        except (OSError, ValueError, KeyError) as exc:
            info["prices"]["checkpoint_reason"] = str(exc)
    info["assessment"] = policy["assessment"]
    ordinary = next(r for r in info["cuts"] if (r["method"], r["look"]) == ("00", 4_000_000))
    if any(m["deficit_bytes"] or m["deficit_inodes"] for m in info["capacity"]["mounts"] or ()):
        reason, next_action = (
            "Registered simultaneous writes exceed a physical mount reserve",
            "inspect_capacity",
        )
    elif not ordinary["weights"]:
        reason, next_action = (
            "Open the original ordinary 4M weight slice and stream bound to the verified round028 pilot",
            "inspect_inputs",
        )
    elif info["families"]["count"] is None:
        reason, next_action = (
            "Open the registered development family census to determine the matched probe size",
            "inspect_inputs",
        )
    elif info["prices"]["followup_family_available"] is not True:
        reason, next_action = (
            "Register one eligible OEIS development family before reserving the matched follow-up",
            "inspect_inputs",
        )
    elif info["prices"]["followup_reserve_us"] is None:
        reason, next_action = (
            "Price one ordinary and one challenger OEIS portfolio with selected scoring on exposed inputs",
            "qualify",
        )
    elif info["matching_pairs"]["count"] is None:
        reason, next_action = (
            "Open the training-pair measurement to determine whether the matching arms can enter the wave",
            "inspect_inputs",
        )
    elif info["matching_pairs"]["count"] == 0 and info["matching_pairs"]["dispositions"].get(
        "needs_native_replay", 0
    ):
        reason, next_action = (
            "Qualify native replay on the unresolved matching-pair rows before funding the matching arms",
            "qualify",
        )
    elif not info["prices"].get("timed_updates"):
        reason, next_action = (
            "Measure the exposed ordinary-stream matching calibration and optimizer batches together",
            "qualify",
        )
    elif "optimizer_to_4M_recipe_01" in info["prices"]["wave"]["missing"] and not info[
        "prices"
    ].get("broader_updates"):
        reason, next_action = (
            "Audit the broader stream and measure eligible recipes 01, 02 and 03 with their registered matching inputs",
            "qualify",
        )
    elif (info["prices"].get("fresh_start") or {}).get(
        "disposition"
    ) == "no_disjoint_ordinary_support":
        reason, next_action = (
            "The new draw frame supplied no eligible ordinary positives. Inspect its exposure counts before extending the frame",
            "inspect_inputs",
        )
    elif (info["prices"].get("fresh_start") or {}).get("disposition") == "fresh_ordinary_only":
        if info["prices"].get("supported_wave"):
            reason, next_action = (
                "Price the outstanding work for the four supported treatments using the exposed timing panel",
                "qualify",
            )
        else:
            reason, next_action = (
                "The fresh ordinary stream is priced. Inspect the native breadth shortage before extending the broader draw frame",
                "inspect_inputs",
            )
    elif not info["prices"].get("fresh_start") and (info["prices"].get("broader_updates") or {}).get(
        "disposition"
    ) == "requires_fresh_disjoint_start":
        reason, next_action = (
            "The original positives overlap protected families. Qualify the fresh ordinary control and eligible matched recipes",
            "qualify",
        )
    elif (info["prices"].get("fresh_start") or info["prices"].get("broader_updates") or {}).get(
        "disposition"
    ) == "requires_frozen_difficulty_map":
        reason, next_action = (
            "Restore the frozen difficulty calibration before pricing broader composition",
            "inspect_inputs",
        )
    elif info["prices"].get("broader_updates") and any(
        "optimizer_to_4M_recipe_" + method in info["prices"]["wave"]["missing"]
        for method in ("02", "03")
    ):
        reason, next_action = (
            "Inspect each stream’s measured matching coverage and common calibration before pricing the missing matching recipes",
            "qualify",
        )
    else:
        reason, next_action = (
            "Measure the outstanding treatment and CPU costs under a preparation grant before funding the screen",
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
            return preflight(root, source, free_disk_gib=getattr(args, "free_disk_gib", 5)), 0
        if action == "status":
            return inspect_status(folder), 0
        if not funded:
            from .progressive_velocity_verify import private_result

            return private_result(root, folder, source=source)
        grant = checked("grant", load(args.authority))
        authorizer = confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
        policy = velocity_policy()
        revision = candidate_revision(Path(source), root)
        from .progressive_velocity_qualification import qualification_parent

        from .progressive_velocity_donation import donation_parent
        from .progressive_velocity_preparation import preparation_parent
        from .progressive_velocity_timing_host import pair_pointer, timing_parent

        claimed = grant["registration_or_parent_ref"]
        donation = action == "qualify" and claimed == donation_parent(revision)
        preparation = action == "qualify" and claimed == preparation_parent(revision)
        from . import progressive_velocity_broad_host as broad
        from . import progressive_velocity_checkpoint_host as checkpoints

        timing = broader = fresh = checkpoint = False
        if (
            action == "qualify"
            and not donation
            and not preparation
            and claimed
            not in {
                qualification_parent(revision),
                qualification_parent(revision, children=True),
            }
        ):
            if (folder / broad.FRESH_MARKER).is_file():
                try:
                    checkpoint = claimed == checkpoints.parent(revision, checkpoints.origin(folder))
                except (OSError, ValueError, KeyError):
                    pass
            try:
                timing = not checkpoint and claimed == timing_parent(revision, pair_pointer(root)["receipt"])
                if not timing and not checkpoint:
                    broader = claimed == broad.parent(
                        revision, broad.original_pointers(root, folder)
                    )
                    if not broader and (folder / broad.MARKER).is_file():
                        fresh = claimed == broad.parent(revision, broad.original_pointers(root, folder, fresh=True))
            except FileNotFoundError:
                pass
        parent = None
        if action == "qualify":
            parent = (
                claimed
                if donation
                or preparation
                or timing
                or broader
                or fresh
                or checkpoint
                or claimed == qualification_parent(revision, children=True)
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
                if checkpoint:
                    return checkpoints.qualify(args, root, folder, source, revision, grant, host_factory)
                if broader or fresh:
                    return broad.qualify(args, root, folder, source, revision, grant, host_factory, fresh=fresh)
                if timing:
                    from .progressive_velocity_timing_host import qualify
                elif preparation:
                    from .progressive_velocity_preparation import qualify
                elif donation:
                    from .progressive_velocity_donation import qualify
                else:
                    from .progressive_velocity_qualification import qualify

                return qualify(args, root, folder, source, revision, grant, host_factory)
            return (
                result(
                    action,
                    "BLOCKED",
                    "Accepted pre-score commitment for this scope required",
                    "qualify",
                ),
                2,
            )
    except Exception as exc:
        return (
            result(
                action,
                "FAILED",
                str(exc) or type(exc).__name__,
                "inspect_inputs",
                first_error=dict(
                    error=type(exc).__name__, stage=action, reason=str(exc) or type(exc).__name__
                ),
                **({"verification_scope": "unverified_inputs"} if action == "verify" else {}),
            ),
            1,
        )
