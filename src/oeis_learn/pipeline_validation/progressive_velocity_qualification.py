"""Workstation-only round029 qualification; durable grants and charges belong to the donor."""

from pathlib import Path
import time

from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.contracts import digest, load, reference, seal
from oeis_learn.generalization.progressive.velocity_records import SESSION, operation_key
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_round029 import result


def _original_profile(root):
    from oeis_learn.generalization.progressive.campaign import learned_search_policy
    from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.velocity_records import open_stored
    from .progressive_round028_audit import ORIGINAL

    campaign = load(
        confined(root, ORIGINAL + "/prepare-research/frozen/campaign.json"),
        maximum=CAMPAIGN_DOCUMENT_BYTES,
    )
    if campaign != seal(campaign) or campaign["identity"] != learned_search_policy()["campaign"]:
        raise ValueError("original frozen training recipe differs")
    entry = next((r for r in campaign["payload"]["references"] if r["name"] == "recipe"), None)
    if not entry:
        raise ValueError("authenticated original recipe missing")
    ref = dict(owner=entry["role"], format=entry["kind"], artifact=entry["artifact"])
    recipe = load(open_stored(root, ref, owner="custodian", format="recipe"))
    return resolve_profile(recipe, seed=29, dropout=0)


def learner_mount(space, *, floor, children=False):
    from .progressive_research_resources import velocity_guard
    from .progressive_pilot_capacity import mount_demand

    generations = 5 if children else 3
    basis = dict(
        owner="qualification learner",
        scope=(
            "five coexisting parent and child full-state generations"
            if children
            else "three coexisting disposable full-state generations"
        ),
        revision="3ecac2fdca8613dc1601864d51f610254308a58a",
        harm="checkpoint publication could consume the physical reserve",
        rehearsal="batch-16/32 update, backward interruption and restore",
        exhaustion="retain earlier cut and price the next batch",
    )
    guards = [
        velocity_guard(
            units="bytes", observed=1_257_820_683, projected=generations * 1_257_820_683, **basis
        ),
        velocity_guard(units="inodes", observed=13, projected=generations * (13 + 6), **basis),
    ]
    available = {u: space["free_" + u] for u in ("bytes", "inodes")}
    return dict(
        guards=guards,
        **mount_demand(
            available=available,
            reserve=dict(bytes=floor, inodes=0),
            pending=[{g["units"]: g["value"] for g in guards}],
        ),
    )


def _supervisor_fact(out, filename, field):
    try:
        return load(confined(out, filename)).get(field)
    except (OSError, ValueError):
        return None


def _receipt(area, out, binding):
    from .progressive_round028_receipts import verify_role_mounts
    from .progressive_velocity_worker import verify_rehearsal

    request = load(confined(area, "request.json"))
    issued = load(confined(out, "issued.json"))
    if (
        request != seal(request)
        or request["version"]
        != (
            "round029-device-request/v2"
            if binding.get("child_restore")
            else "round029-device-request/v1"
        )
        or request["binding"] != binding["identity"]
        or issued != seal(issued)
        or issued["request"] != request["identity"]
    ):
        raise ValueError("issued learner request differs from the bound qualification")
    produced = load(confined(out, "producer.json"))
    verify_rehearsal(produced, request)
    verify_role_mounts(
        load(confined(out, "container-inspect.json")),
        request["image"],
        {"/work/inputs/request.json": (area / "request.json", False), "/work/output": (out, True)},
    )
    process = load(confined(out, "process.json"))
    cleanup = load(confined(out, "cleanup.json"))
    lease = load(confined(out, "gpu-lease.json"))["elapsed_us"]
    if (
        process["outcome"] != "COMPLETE"
        or process["returncode"] != 0
        or process["cleanup_failure"]
        or cleanup["acknowledged"] is not True
        or cleanup["within_bound"] is not True
        or type(lease) is not int
        or not 0 < produced["elapsed_us"] <= lease
    ):
        raise ValueError("worker termination, cleanup or measured lease differs")
    documents = [
        area / "request.json",
        *(
            out / name
            for name in (
                "issued.json",
                "producer.json",
                "container-inspect.json",
                "process.json",
                "cleanup.json",
                "gpu-lease.json",
            )
        ),
    ]
    return seal(
        dict(
            binding=binding["identity"],
            attempt=out.name,
            request=request["identity"],
            debits=issued["debits"],
            producer=produced["identity"],
            artifacts=[reference(area, p) for p in documents],
            batches=produced["batches"],
            peak_device_bytes=produced["peak_device_bytes"],
            historical_peak_device_bytes=4_852_388_352,
            literal_dispositions={
                name: proof["status"] for name, proof in produced["literal_probes"].items()
            },
            lease_us=lease,
            telemetry=produced["telemetry"],
        )
    )


def qualify(args, root, folder, source, revision, grant, host_factory):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from .progressive_handoff import stage
    from .progressive_round028_rehearsal import physical_scope, space
    from .progressive_roles import ROUND029_ROLE_POLICY
    from .progressive_velocity_worker import rehearsal_work

    children = grant["registration_or_parent_ref"] == qualification_parent(revision, children=True)
    area = confined(folder, "qualification/" + grant["identity"][7:], exists=False)
    area.mkdir(parents=True, exist_ok=True)
    binding = seal(
        dict(
            version="round029-learner-rehearsal/v1",
            **({"child_restore": True} if children else {}),
            grant=grant["identity"],
            candidate=revision,
            role_policy=ROUND029_ROLE_POLICY,
        )
    )
    retain(area / "binding.json", binding)
    retain(area / "grant.json", grant)
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    receipt = area / "receipt.json"
    first_error = area / "first-error.json"

    def status(state, reason, next_action, **data):
        snapshot = journal.snapshot()
        if first_error.exists():
            data["first_error"] = load(confined(area, "first-error.json"))
        diagnostic = data.get("diagnostics", {})
        data.update({k: diagnostic[k] for k in ("deficit_bytes", "deficit_inodes") if k in diagnostic})
        answer = result(
            "qualify",
            state,
            reason,
            next_action,
            remaining=snapshot["remaining"]["elapsed_us"],
            stage="qualification",
            scope="learner_child_restore" if children else "learner_update_restore_only",
            charged_tokens=snapshot["consumed"]["tokens"],
            elapsed_us=snapshot["consumed"]["elapsed_us"],
            remaining_tokens=snapshot["remaining"]["tokens"],
            research_dispatches=0,
            **data,
        )
        summary = folder / "status.json"
        sealed = seal(answer)
        if not summary.exists() or load(confined(folder, "status.json")) != sealed:
            revise(summary, sealed)
        return answer

    def complete(out):
        record = _receipt(area, out, binding)
        for key in record["debits"]:
            journal.commit(key)
        retain(receipt, record)
        return status(
            "COMPLETE",
            "Disposable learner hardware rehearsal verified",
            "review_qualification",
            last=record["identity"],
            receipt=record,
            receipt_path=str(receipt.relative_to(root)),
        ), 0

    if receipt.exists():
        prior = load(confined(area, "receipt.json"))
        return complete(confined(area, prior["attempt"]))
    # A supervisor loss cannot refund unobserved elapsed time or issued tokens.
    previous = journal.reconcile()
    for key in previous["pending"]:
        journal.commit(key)
    for old in sorted(area.glob("attempt-*")):
        if all(
            (old / name).is_file()
            for name in (
                "issued.json",
                "producer.json",
                "container-inspect.json",
                "process.json",
                "cleanup.json",
                "gpu-lease.json",
            )
        ):
            if load(confined(old, "process.json"))["outcome"] == "COMPLETE":
                return complete(confined(area, old.name))

    state = journal.snapshot()
    left = state["remaining"]["elapsed_us"] - grant["holdback_us"]
    # The shared stage supervisor uses a 120 s cleanup/reclamation deadline.
    cleanup_seconds = 120
    if left // 1_000_000 <= cleanup_seconds:
        return status(
            "BLOCKED",
            "Remaining workstation time cannot fund launch and reclamation",
            "inspect_grant",
            cleanup_seconds=cleanup_seconds,
        ), 2
    costs = rehearsal_work(children=children)
    costs["disk_bytes"] = learner_mount(
        {"free_bytes": 0, "free_inodes": 0}, floor=0, children=children
    )["required_bytes"]
    if any(state["remaining"][k] < v for k, v in costs.items()):
        return status(
            "BLOCKED", "Original rehearsal resources cannot fund this dispatch", "inspect_attempt"
        ), 2
    coord = dict(
        study=grant["identity"],
        stage="qualification",
        method="learner",
        seed=29,
        look=0,
        panel="exposed",
        profile="disposable",
        input_mode="true",
        role="train",
        ordinal=0,
    )
    attempt = sum("/qualify/" in key for key in state["committed"])
    key = operation_key(coord, attempt)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def now():
        return time.monotonic_ns() // 1000

    start = now()
    clock = journal.reserve(
        key["logical"],
        "qualify",
        attempt,
        dict(dict.fromkeys(KEYS, 0), elapsed_us=left),
        boot=boot,
        monotonic_us=start,
    )
    out = area / f"attempt-{attempt:04d}"
    work = None
    stage_started = False

    def seconds_left():
        remaining = (start + left - now()) // 1_000_000 - cleanup_seconds
        if remaining < 1:
            raise ValueError("qualification time cannot fund the remaining work and reclamation")
        return remaining

    try:
        out.mkdir()
        status(
            "PARTIAL",
            "Preparing installed batch-16/32 learner and complete-state restore",
            "inspect_progress",
            operation=key,
            attempt=attempt,
            reserved_us=left,
        )
        profile = _original_profile(root)
        host = host_factory(args, source_root=source, workspace=root, revision=revision)
        capacity = learner_mount(
            space(root), floor=host.policy["free_disk_floor_bytes"], children=children
        )
        if capacity["deficit_bytes"] or capacity["deficit_inodes"]:
            raise ArithmeticError("Physical checkpoint reserve would be exceeded")
        request_path = area / "request.json"
        if request_path.exists():
            saved = load(confined(area, "request.json"))
            image = host.engine.image(saved["image"])
            if saved != seal(saved) or image != saved["image"]:
                raise ValueError("original qualified worker image differs")
        elif args.image:
            image = host.engine.image(args.image)
        else:
            image = host.engine.build(
                source,
                target="gpu",
                tag="oeis-learn:round029-" + revision,
                base_image=args.gpu_base_image,
                seconds=min(86400, seconds_left()),
            )
        capacity = learner_mount(
            space(root), floor=host.policy["free_disk_floor_bytes"], children=children
        )
        if capacity["deficit_bytes"] or capacity["deficit_inodes"]:
            raise ArithmeticError("Insufficient post-build checkpoint capacity")
        physical = physical_scope(host, "hip", machine=True)
        request = seal(
            dict(
                version="round029-device-request/v2" if children else "round029-device-request/v1",
                binding=binding["identity"],
                source=digest(collect_snapshot(source)),
                image=image,
                physical=physical,
                profile=profile,
                batch_sizes=[16, 32],
                role_policy=ROUND029_ROLE_POLICY,
            )
        )
        retain(request_path, request)
        seconds = seconds_left()
        work = journal.reserve(
            key["logical"],
            "learner",
            attempt,
            dict(dict.fromkeys(KEYS, 0), **costs),
            boot=boot,
            monotonic_us=now(),
        )
        retain(
            out / "issued.json",
            seal(dict(debits=[clock["key"], work["key"]], request=request["identity"])),
        )
        stage_started = True
        stage(
            host,
            source,
            image,
            out,
            ["python", "-I", "-m", "oeis_learn.pipeline_validation.progressive_velocity_worker"],
            gpu=True,
            record_lease=True,
            seconds=seconds,
            readonly_inputs=[(request_path, "/work/inputs/request.json")],
        )
        # Verify the whole installed boundary before claiming completion.
        record = _receipt(area, out, binding)
        retain(receipt, record)
    except (Exception, KeyboardInterrupt) as exc:
        capacity_fault = isinstance(exc, ArithmeticError) and "capacity" in locals()
        diagnostics = {
            "reason": str(exc) or type(exc).__name__,
            "error": type(exc).__name__,
            "attempt": attempt,
            "stage": "qualification",
            "attempt_path": str(out.relative_to(root)),
            "exit": 130 if isinstance(exc, KeyboardInterrupt) else getattr(exc, "returncode", None),
        }
        process_exit = _supervisor_fact(out, "process.json", "returncode")
        if process_exit is not None and diagnostics["exit"] is None:
            diagnostics["exit"] = process_exit
        if capacity_fault:
            diagnostics.update(
                deficit_bytes=capacity["deficit_bytes"], deficit_inodes=capacity["deficit_inodes"]
            )
            retain(out / "capacity.json", seal(capacity))
        if not first_error.exists():
            retain(first_error, seal(diagnostics))
        failure = (
            "BLOCKED" if capacity_fault else "FAILED",
            diagnostics["reason"],
            "inspect_capacity" if capacity_fault else "inspect_attempt",
            diagnostics,
        )
    else:
        failure = None
    finally:
        if work is not None:
            journal.commit(work["key"])
        acknowledged = (
            not stage_started or _supervisor_fact(out, "cleanup.json", "acknowledged") is True
        )
        end = now()
        journal.commit(
            clock["key"],
            boot=boot,
            monotonic_us=end,
            termination_proven=acknowledged and end - start <= left,
        )
    if failure:
        state_name, reason, action, diagnostics = failure
        code = (
            130
            if diagnostics["error"] == "KeyboardInterrupt"
            else 2
            if state_name == "BLOCKED"
            else 1
        )
        return status(state_name, reason, action, diagnostics=diagnostics), code
    return complete(out)


def qualification_parent(revision, *, children=False):
    from oeis_learn.generalization.progressive.campaign import velocity_policy

    return digest(
        [
            velocity_policy(),
            revision,
            "disposable_child_restore" if children else "disposable_learner_restore",
        ]
    )


def authorize_learner_check(root, minutes, revision, *, children=False):
    """Create operator-side authorization only on an explicit, priced-duration invocation."""
    from .progressive_velocity_worker import rehearsal_work

    work = dict(
        rehearsal_work(children=children),
        disk_bytes=learner_mount({"free_bytes": 0, "free_inodes": 0}, floor=0, children=children)[
            "required_bytes"
        ],
    )
    return authorize_qualification(
        root, minutes, qualification_parent(revision, children=children), work
    )


def authorize_qualification(root, minutes, parent, work):
    from oeis_learn.generalization.budgets import integer
    from oeis_learn.generalization.progressive.velocity_records import physical_root_id

    total = integer(minutes, 1) * 60_000_000
    terms = dict(
        version="round029-grant/v1",
        study_id=SESSION,
        scope="qualification",
        registration_or_parent_ref=parent,
        state_root_id=physical_root_id(root),
        total_us=total,
        holdback_us=0,
        allowed_operations=["qualify"],
        resources=dict(dict.fromkeys(KEYS, 0), elapsed_us=total, **work),
    )
    approval = seal(dict(version="round029-authorization/v1", terms=digest(terms)))
    authority = confined(root, "run-state/discovery-authorizations", exists=False)
    retain(authority / (approval["identity"][7:] + ".json"), approval)
    grant = seal(dict(terms, authorizing_record_ref=approval["identity"]))
    target = authority / ("grant-" + grant["identity"][7:] + ".json")
    retain(target, grant)
    return target
