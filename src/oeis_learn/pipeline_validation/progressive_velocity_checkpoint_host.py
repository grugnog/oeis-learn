"""One separately funded, resumable checkpoint qualification on the retained fresh projection."""

import time

from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import digest, load, reference, seal
from oeis_learn.generalization.progressive.trials import integer
from oeis_learn.generalization.progressive.velocity_records import SESSION, operation_key
from . import progressive_velocity_broad_host as broad
from . import progressive_velocity_checkpoint as matrix
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_handoff import stage
from .progressive_research_resources import velocity_capacity
from .progressive_roles import ROUND029_ROLE_POLICY
from .progressive_round028_rehearsal import physical_scope
from .progressive_velocity_image import preparation_boundary
from .progressive_velocity_operations import InsufficientAuthority, replay_operation, run_operation
from .progressive_velocity_qualification import authorize_qualification, learner_mount
from .progressive_velocity_runtime import (
    cached_runtime,
    package_evidence,
    package_worker,
    wheel_mounts,
)

MARKER = "preparation/checkpoint-matrix.json"


def origin(folder):
    pointer = load(confined(folder, broad.FRESH_MARKER))
    if pointer != seal(pointer):
        raise ValueError("fresh qualification pointer differs")
    return pointer


def parent(revision, pointer):
    return digest([matrix.VERSION, revision, pointer, matrix.METHODS, ROUND029_ROLE_POLICY])


def inputs(root, folder, source):
    fresh = broad.inventory(root, source, fresh=True)
    pointer = origin(folder)
    if (
        not fresh
        or fresh["identity"] != pointer["receipt"]
        or not set(matrix.METHODS) <= set((fresh["device"] or {}).get("methods", {}))
    ):
        raise ValueError("the four ordinary-stream optimizer prices must be privately verified")
    previous = confined(folder, "qualification/" + pointer["grant"][7:])
    grant = load(confined(previous, "grant.json"))
    cpu = broad.attempt(previous, broad.coordinates(grant, "prepare"), grant)
    path = confined(cpu, "native/projection.json")
    view = load(path, maximum=broad.MAX_INPUT)
    timed = load(confined(previous, "device-request.json"))
    if (
        timed != seal(timed)
        or timed["projection"] != reference(path.parent, path)
        or view["fresh_start"] != fresh["start"]["identity"]
    ):
        raise ValueError("exposed training projection differs from the paid fresh timing")
    return (
        pointer,
        path,
        timed["physical"],
        matrix.design(view, origin=fresh["identity"], microbatch=timed["design"]["microbatch"]),
    )


def authorize(root, source, revision, minutes):
    folder = confined(root, "run-state/discovery/" + SESSION)
    pointer, _, _, plan = inputs(root, folder, source)
    return authorize_qualification(
        root, int(minutes), parent(revision, pointer), {k: 2 * v for k, v in plan["work"].items()}
    )


def coordinates(grant, phase):
    return dict(broad.coordinates(grant, phase), method="checkpoint_matrix")


def mount_totals(mounts):
    return [
        {
            k: v
            for k, v in row.items()
            if k.startswith(("available_", "required_", "reserved_", "deficit_"))
        }
        for row in mounts
    ]


def capacity(root, area, revision, phase, floor):
    # Recipes run serially and delete their disposable directory between methods.
    if phase == "package":
        peak = dict(bytes=2 * broad.archive_size(root, revision), inodes=16)
        observed = dict(bytes=0, inodes=0)
    else:
        checkpoint = learner_mount({"free_bytes": 0, "free_inodes": 0}, floor=0)
        peak = {k: checkpoint["required_" + k] // 2 for k in ("bytes", "inodes")}
        observed = {g["units"]: g["observed"] for g in checkpoint["guards"]}
    path = str(area.relative_to(root))
    owner = dict(
        owner="checkpoint_matrix_" + phase,
        path=path,
        scope=phase,
        retained=[],
        observed=observed,
        projected=peak,
        revision=revision,
        harm="incomplete checkpoint publication",
        exhaustion="retain completed receipts and inspect capacity",
        rehearsal="T060",
    )
    mounts = velocity_capacity(
        root, owners=[owner], floors=[dict(path=path, bytes=floor, inodes=0)]
    )["mounts"]
    if any(row["deficit_bytes"] or row["deficit_inodes"] for row in mounts):
        raise broad.CapacityShortfall(phase, mounts, None)
    return mount_totals(mounts)


def dispatch_capacity(out, request):
    data = load(confined(out.parent, out.name + "-capacity.json"))
    if (
        data != seal(data)
        or data["request"] != request["identity"]
        or not data["mounts"]
        or any(
            row["deficit_" + unit] != 0
            or row["available_" + unit] < row["required_" + unit] + row["reserved_" + unit]
            for row in data["mounts"]
            for unit in ("bytes", "inodes")
        )
    ):
        raise ValueError("dispatch capacity differs from the recorded requirement or floor")
    return data["mounts"]


def partial_treatments(area, grant, plan):
    logical = operation_key(coordinates(grant, "measure"), 0)["logical"][7:]
    found = {}
    for attempt in range(2):
        for method in matrix.METHODS:
            filename = f"operations/{logical}/attempt-{attempt:04d}/recipe-{method}.json"
            try:
                item = load(confined(area, filename))
                row = matrix.checked_row(plan, method, item["measurement"])
                if item == matrix.treatment(plan, method, row):
                    found[f"attempt-{attempt}-recipe-{method}"] = item
            except (OSError, TypeError, KeyError, ValueError):
                continue
    return found


def incomplete(root, source, authority):
    folder = confined(root, "run-state/discovery/" + SESSION)
    grant = load(confined(root, authority))
    _, _, _, plan = inputs(root, folder, source)
    area = confined(folder, "qualification/" + grant["identity"][7:])
    return partial_treatments(area, grant, plan)


def stages(root, area, grant, binding, projection, plan, run):
    def verify_package(out):
        runtime = load(confined(out, "runtime.json"))
        package_evidence(root, area, runtime, binding)
        return seal(dict(runtime=runtime["identity"], capacity=dispatch_capacity(out, binding)))

    installed = run(
        "package", binding, {}, verify_package, lambda out: (out / "runtime.json").is_file()
    )
    package_out = broad.attempt(area, coordinates(grant, "package"), grant)
    runtime = load(confined(package_out, "runtime.json"))
    request = seal(
        dict(
            version=matrix.VERSION,
            physical=binding["physical"],
            image=runtime["parent"]["image"],
            source=binding["source"],
            role_policy=ROUND029_ROLE_POLICY,
            design=plan,
            projection=reference(projection.parent, projection),
        )
    )

    def verify_device(out):
        from .progressive_generator_probe import usage_summary

        if load(confined(area, "request.json")) != request:
            raise ValueError("checkpoint matrix request differs")
        broad.check_stage(area, out, runtime, area / "request.json", projection, gpu=True)
        data = load(confined(out, "producer.json"))
        lease = integer(load(confined(out, "gpu-lease.json"))["elapsed_us"], 1)
        elapsed = integer(data["elapsed_us"], 1)
        telemetry = usage_summary(confined(out, "telemetry/samples.jsonl").parent, elapsed)
        if (
            data != seal(data)
            or data["request"] != request["identity"]
            or data["device"] != "hip"
            or elapsed > lease
            or telemetry != data["telemetry"]
            or telemetry["covered_us"] > elapsed
        ):
            raise ValueError("invalid checkpoint matrix producer or physical clocks")
        rows = matrix.checked_methods(plan, data["matrix"])
        for method, row in rows.items():
            if load(confined(out, f"recipe-{method}.json")) != matrix.treatment(plan, method, row):
                raise ValueError(
                    "individually published treatment differs from its completed matrix"
                )
        if (
            sum(
                r[k]
                for r in rows.values()
                for k in ("first_cut_us", "resumed_cut_us", "restore_us")
            )
            > data["elapsed_us"]
        ):
            raise ValueError("treatment clocks exceed the measured device stage")
        return seal(
            dict(
                producer=data["identity"],
                methods=rows,
                capacity=dispatch_capacity(out, request),
                elapsed_us=elapsed,
                lease_us=lease,
                telemetry=telemetry,
                worker_cpu_us=integer(data["worker_cpu_us"]),
                peak_device_bytes=integer(data["peak_device_bytes"], 1),
            )
        )

    paid = run(
        "measure",
        request,
        plan["work"],
        verify_device,
        lambda out: (out / "producer.json").is_file(),
    )
    return seal(
        dict(
            version=matrix.VERSION,
            binding=binding["identity"],
            origin=plan["origin"],
            package_capacity=installed["evidence"]["capacity"],
            acquisition=installed["identity"],
            measurement=paid["identity"],
            **paid["evidence"],
        )
    )


def replay(root, folder, area, source, grant, binding, snapshot):
    pointer, projection, physical, plan = inputs(root, folder, source)
    if (
        binding != seal(binding)
        or binding["origin"] != pointer
        or binding["physical"] != physical
        or binding["plan"] != plan["identity"]
        or binding["source"] != digest(collect_snapshot(source))
    ):
        raise ValueError("checkpoint origins or recorded candidate source differ")

    def reopen(phase, request, work, verify, _installed):
        return replay_operation(
            area, coordinates(grant, phase), request, grant, work, snapshot, verify
        )

    return stages(root, area, grant, binding, projection, plan, reopen)


def inventory(root, source):
    """Reopen the measured cycles independently of the last command on the workstation."""
    from oeis_learn.generalization.progressive.velocity_records import (
        physical_root_id,
        validate_grant,
    )
    from .progressive_velocity_timing_host import original_price_source

    folder = confined(root, "run-state/discovery/" + SESSION, exists=False)
    if not (folder / MARKER).is_file():
        return None
    marker = load(confined(folder, MARKER))
    area = confined(folder, "qualification/" + marker["grant"][7:])
    grant = load(confined(area, "grant.json"))
    binding = load(confined(area, "binding.json"))
    if (
        marker != seal(marker)
        or binding != seal(binding)
        or binding["version"] != matrix.VERSION
        or marker["grant"] != grant["identity"]
        or binding["grant"] != grant["identity"]
    ):
        raise ValueError("checkpoint marker differs from its original grant")
    authority = (
        "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json"
    )
    validate_grant(
        grant,
        load(confined(root, authority)),
        root=physical_root_id(root),
        parent=parent(binding["candidate"], binding["origin"]),
        action="qualify",
    )
    original, _, _ = original_price_source(root, source, binding)
    balance = BudgetJournal.inspect_existing(
        confined(area, "accounting"), grant["identity"], grant["resources"], include_work=True
    )
    record = replay(root, folder, area, original, grant, binding, balance)
    if (
        balance["pending"]
        or record != load(confined(area, "receipt.json"))
        or record["identity"] != marker["receipt"]
    ):
        raise ValueError("original checkpoint receipt or paid boundary differs")
    return dict(
        candidate=binding["candidate"],
        receipt=record["identity"],
        origin=record["origin"],
        methods=record["methods"],
        lease_us=record["lease_us"],
        charged_tokens=balance["consumed"]["tokens"],
    )


def qualify(args, root, folder, source, revision, grant, host_factory):
    from .progressive_round029 import result

    pointer, projection, physical, plan = inputs(root, folder, source)
    if parent(revision, pointer) != grant["registration_or_parent_ref"]:
        raise ValueError("checkpoint qualification authority differs")
    area = confined(folder, "qualification/" + grant["identity"][7:], exists=False)
    area.mkdir(parents=True, exist_ok=True)
    binding = seal(
        dict(
            version=matrix.VERSION,
            candidate=revision,
            source=digest(collect_snapshot(source)),
            grant=grant["identity"],
            origin=pointer,
            physical=physical,
            plan=plan["identity"],
        )
    )
    retain(area / "binding.json", binding)
    retain(area / "grant.json", grant)
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    host = None
    phase = "package"

    def machine():
        nonlocal host
        if host is None:
            host = host_factory(args, source_root=source, workspace=root, revision=revision)
        return host

    def run(name, request, work, verify, installed):
        nonlocal phase
        phase = name
        space = []

        def guard():
            floor = max(
                machine().policy["free_disk_floor_bytes"], getattr(args, "free_disk_gib", 0) << 30
            )
            space.extend(capacity(root, area, revision, phase, floor))

        def execute(out, allowance):
            deadline = time.monotonic_ns() + allowance * 1000
            retain(
                out.parent / (out.name + "-capacity.json"),
                seal(dict(request=request["identity"], mounts=space)),
            )

            def seconds():
                left = (deadline - time.monotonic_ns()) // 1_000_000_000 - 120
                if left < 1:
                    raise InsufficientAuthority("insufficient funded time for launch and cleanup")
                return left

            if name == "package":
                with preparation_boundary(out, deadline):
                    prior = cached_runtime(root, folder, revision, machine().engine)
                    if prior is None:
                        raise ValueError("settled compatible runtime required")
                    runtime = package_worker(
                        root,
                        source,
                        area,
                        out,
                        revision,
                        prior,
                        binding["source"],
                        machine(),
                        stage,
                        seconds,
                    )
                    retain(out / "runtime.json", runtime)
                return
            try:
                package_out = broad.attempt(area, coordinates(grant, "package"), grant)
                runtime = load(confined(package_out, "runtime.json"))
                if (
                    machine().engine.image(request["image"]) != request["image"]
                    or physical_scope(machine(), "hip", machine=True) != physical
                ):
                    raise ValueError("checkpoint matrix physical device or image differs")
                extras = wheel_mounts(area, runtime)
                retain(area / "request.json", request)
                mounts = [
                    *extras,
                    (area / "request.json", "/work/inputs/request.json"),
                    (projection, "/work/inputs/projection.json"),
                ]
                timeout = seconds()
            except (Exception, KeyboardInterrupt):
                retain(
                    out / "cleanup.json",
                    dict(acknowledged=True, within_bound=time.monotonic_ns() <= deadline),
                )
                raise
            stage(
                machine(),
                source,
                request["image"],
                out,
                [
                    "python",
                    "-I",
                    "-B",
                    extras[0][1],
                    "checkpoint-matrix",
                    extras[1][1],
                    runtime["wheel"]["sha256"],
                    binding["source"],
                ],
                gpu=True,
                record_lease=True,
                seconds=timeout,
                readonly_inputs=mounts,
            )

        return run_operation(
            area=area,
            grant=grant,
            journal=journal,
            minimum_us=121_000_000,
            coordinates=coordinates(grant, name),
            request=request,
            cost=work,
            execute=execute,
            verify=verify,
            installed=installed,
            before_dispatch=guard,
            max_attempts=2,
        )

    extra = {}
    try:
        rec = stages(root, area, grant, binding, projection, plan, run)
        retain(area / "receipt.json", rec)
        marker = seal(dict(grant=grant["identity"], receipt=rec["identity"]))
        if not (folder / MARKER).is_file() or load(confined(folder, MARKER)) != marker:
            revise(folder / MARKER, marker)
        status, code, reason, nxt = (
            "COMPLETE",
            0,
            "Four disposable checkpoint cycles verified",
            "verify",
        )
        extra = dict(
            last=rec["identity"],
            receipt=rec,
            receipt_path=str((area / "receipt.json").relative_to(root)),
        )
    except (Exception, KeyboardInterrupt) as exc:
        problem = dict(stage=phase, error=type(exc).__name__, reason=str(exc) or type(exc).__name__)
        if isinstance(exc, broad.CapacityShortfall):
            extra.update(exc.deficits)
            problem.update(exc.deficits)
            problem["capacity"] = dict(mounts=mount_totals(exc.capacity["mounts"]))
        if not (area / "first-error.json").exists():
            retain(area / "first-error.json", seal(problem))
        status, code = ("BLOCKED", 2) if isinstance(exc, InsufficientAuthority) else ("FAILED", 1)
        reason, nxt = (
            problem["reason"],
            "inspect_capacity" if isinstance(exc, broad.CapacityShortfall) else "inspect_attempt",
        )
        extra["diagnostics"] = problem
    if (area / "first-error.json").is_file():
        extra["first_error"] = load(confined(area, "first-error.json"))
    balance = journal.snapshot()
    value = result(
        "qualify",
        status,
        reason,
        nxt,
        remaining=balance["remaining"]["elapsed_us"],
        elapsed_us=balance["consumed"]["elapsed_us"],
        charged_tokens=balance["consumed"]["tokens"],
        research_dispatches=0,
        scope="checkpoint_matrix",
        stage=phase,
        **extra,
    )
    if not (folder / "status.json").is_file() or load(confined(folder, "status.json")) != seal(
        value
    ):
        revise(folder / "status.json", seal(value))
    return value, code
