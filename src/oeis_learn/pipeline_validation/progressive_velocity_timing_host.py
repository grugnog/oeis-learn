"""Fund, run and reopen one exposed-input timing matrix on the physical HIP worker."""

import json
from pathlib import Path
import re
import runpy
import time
import zipfile

from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest, load, reference, seal
from oeis_learn.generalization.progressive.velocity_records import (
    SESSION,
    operation_key,
    physical_root_id,
    validate_grant,
)
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_handoff import stage
from .progressive_roles import ROUND029_ROLE_POLICY
from .progressive_round028_receipts import verify_role_mounts
from .progressive_round028_rehearsal import physical_scope
from .progressive_velocity_census import training_source
from .progressive_velocity_donor import PILOT_REPORT
from .progressive_velocity_image import preparation_boundary, storage
from .progressive_velocity_operations import InsufficientAuthority, replay_operation, run_operation
from .progressive_velocity_preparation import funded_inventory
from .progressive_velocity_supply import _area
from .progressive_velocity_timing import matrix_prices, timing_design, timing_projection

VERSION = "round029-ordinary-timing/v1"


def pair_pointer(root):
    try:
        area = _area(root, "matching_pairs")
        pointer = load(confined(area, "funded.json"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Run the donor supply qualification on this detached workstation before timing updates"
        ) from exc
    if pointer != seal(pointer) or pointer["kind"] != "matching_pairs":
        raise ValueError("original matching-pair payment required")
    return pointer


def timing_parent(revision, pair_receipt):
    return digest([VERSION, velocity_policy(), revision, PILOT_REPORT, pair_receipt])


def donor_inputs(root, pointer):
    public, pairs = funded_inventory(root, "matching_pairs", pointer, with_original=True)
    _, package, sources = training_source(root)
    # training_source checks both pinned archives and their pilot-reader result.
    from .progressive_velocity_donor import ARCHIVES
    from .progressive_research_inputs import checked_file

    pilot = next(r for r in sources if r["sha256"] == "sha256:" + ARCHIVES["pilot"][1])
    with zipfile.ZipFile(checked_file(root, pilot)) as archive:
        report = json.loads(archive.read("pilot.json"))
    cut = next(r for r in report["looks"] if r["target"] == 0)["cuts"]["ordinary"]
    if (
        cut["stream"] != package["stream"]["identity"]
        or cut["updates"]
        or cut["useful_tokens"]
        or cut["cursor"]
        or cut["seeds"] != package["stream"]["seeds"]
    ):
        raise ValueError("ordinary zero cut differs from the exposed training stream")
    projection = timing_projection(package, pairs, cut["initial"], public["measurement"])
    return projection, timing_design(
        projection, len(package["stream"]["order"]), package["microbatch"]
    )


def authorize_timing(root, minutes, revision):
    from .progressive_velocity_qualification import authorize_qualification
    from .progressive_velocity_timing import timing_limits

    pointer = pair_pointer(root)

    # The paid preparation opens and verifies the training archive and selects the
    # actual samples. Funding needs only the prior payment and public recipe limits.
    work = timing_limits()
    return authorize_qualification(root, minutes, timing_parent(revision, pointer["receipt"]), work)


def coordinates(grant, phase):
    return dict(
        study=grant["identity"],
        stage="qualification",
        method="ordinary_timing",
        seed=0,
        look=0,
        panel="exposed_training",
        profile="timed_updates",
        input_mode="true",
        role=phase,
        ordinal=0,
    )


def bind(revision, source_hash, grant, pointer):
    return seal(
        dict(
            version=VERSION,
            candidate=revision,
            source=source_hash,
            grant=grant["identity"],
            pair_pointer=pointer,
        )
    )


def request_at(root, area, binding, inputs):
    projection, design = inputs
    path = confined(area, "projection.json")
    req = load(confined(area, "request.json"))
    if (
        req != seal(req)
        or req["version"]
        != ("round029-timing-request/v2" if "runtime" in req else "round029-timing-request/v1")
        or req["binding"] != binding["identity"]
        or req["source"] != binding["source"]
        or req["role_policy"] != ROUND029_ROLE_POLICY
        or req["projection"] != reference(area, path)
        or load(path, maximum=req["projection"]["bytes"]) != projection
        or req["design"] != design
    ):
        raise ValueError(
            "timing request, positive streams or source differ from original preparation"
        )
    if "runtime" in req:
        from .progressive_velocity_runtime import package_evidence

        if req["runtime"]["parent"]["image"] != req["image"]:
            raise ValueError("wheel belongs to a different runtime image")
        package_evidence(root, area, req["runtime"], binding)
    return req


def prepared(out, req):
    producer = load(confined(out, "producer.json"))
    fields = dict(request=req["identity"])
    if "storage" in producer:
        mounts = {k: load(confined(out, "storage-" + k + ".json")) for k in ("before", "after")}
        fields["storage"] = dict(mounts, transport=producer["storage"]["transport"])
        transport = fields["storage"]["transport"]
        if "runtime" in req:
            if (
                not isinstance(transport, dict)
                or set(transport) != {"bytes", "inodes"}
                or any(type(v) is not int or v <= 0 for v in transport.values())
            ):
                raise ValueError("measured candidate transport allocations required")
        elif transport is not None:
            raise ValueError("standalone image has no candidate wheel transport")
    if producer != seal(fields):
        raise ValueError("original timing preparation or storage observations differ")
    return producer


def device_evidence(area, out, req, projection):
    produced = load(confined(out, "producer.json"))
    if (
        produced != seal(produced)
        or produced["request"] != req["identity"]
        or produced["device"] != "hip"
    ):
        raise ValueError("timing producer differs from the issued request")
    mounts = {
        "/work/inputs/request.json": (area / "request.json", False),
        "/work/inputs/projection.json": (area / "projection.json", False),
        "/work/output": (out, True),
    }
    if "runtime" in req:
        from .progressive_velocity_runtime import wheel_mounts

        mounts.update({dest: (p, False) for p, dest in wheel_mounts(area, req["runtime"])})
    verify_role_mounts(load(confined(out, "container-inspect.json")), req["image"], mounts)
    proc, cleanup, lease = [
        load(confined(out, p)) for p in ("process.json", "cleanup.json", "gpu-lease.json")
    ]
    peak, elapsed = produced["peak_device_bytes"], produced["elapsed_us"]
    if (
        proc["outcome"] != "COMPLETE"
        or proc["returncode"] != 0
        or proc["cleanup_failure"]
        or cleanup["acknowledged"] is not True
        or cleanup["within_bound"] is not True
        or type(lease["elapsed_us"]) is not int
        or type(elapsed) is not int
        or not 0 < elapsed <= lease["elapsed_us"]
        or type(peak) is not int
        or peak <= 0
    ):
        raise ValueError("original HIP timing, termination, or memory evidence differs")
    from .progressive_generator_probe import usage_summary
    from oeis_learn.generalization.progressive.trials import integer

    sample_path = confined(out, "telemetry/samples.jsonl")
    telemetry = usage_summary(sample_path.parent, elapsed)
    if telemetry != produced["telemetry"] or telemetry["covered_us"] > elapsed:
        raise ValueError(
            "sampled activity or coverage differs from the original device observations"
        )
    cpu_us = integer(produced["worker_cpu_us"])
    units = matrix_prices(produced["matrix"], projection, req["design"], device="cuda")
    methods = produced["matrix"]["methods"]
    accounted = units["calibration_us"] + sum(
        m["startup_us"] + sum(x["elapsed_us"] for x in m["samples"].values())
        for m in methods.values()
        if "samples" in m
    )
    if accounted > elapsed:
        raise ValueError("serial calibration and optimizer clocks exceed worker elapsed time")
    return seal(
        dict(
            producer=produced["identity"],
            **({"runtime": req["runtime"]["identity"]} if "runtime" in req else {}),
            units=units,
            worker_cpu_us=cpu_us,
            telemetry=telemetry,
            lease_us=lease["elapsed_us"],
            peak_device_bytes=peak,
            artifacts=[
                reference(area, out / p)
                for p in (
                    "producer.json",
                    "container-inspect.json",
                    "process.json",
                    "cleanup.json",
                    "gpu-lease.json",
                    "telemetry/samples.jsonl",
                )
            ],
        )
    )


def combined(binding, prepared_receipt, measurement):
    evidence = measurement["evidence"]
    return seal(
        dict(
            version=VERSION,
            binding=binding["identity"],
            preparation=prepared_receipt["identity"],
            **(
                {"storage": prepared_receipt["evidence"]["storage"]}
                if "storage" in prepared_receipt["evidence"]
                else {}
            ),
            **({"runtime": evidence["runtime"]} if "runtime" in evidence else {}),
            measurement=measurement["identity"],
            units=evidence["units"],
            worker_cpu_us=evidence["worker_cpu_us"],
            telemetry=evidence["telemetry"],
            lease_us=evidence["lease_us"],
            peak_device_bytes=evidence["peak_device_bytes"],
        )
    )


def replay(root, area, binding, grant, snapshot):
    pointer = binding["pair_pointer"]
    if binding != bind(binding["candidate"], binding["source"], grant, pointer):
        raise ValueError("timing binding changed")
    data = donor_inputs(root, pointer)
    req = request_at(root, area, binding, data)
    pre = replay_operation(
        area,
        coordinates(grant, "prepare"),
        binding,
        grant,
        {},
        snapshot,
        lambda out: prepared(out, req),
    )
    paid = replay_operation(
        area,
        coordinates(grant, "measure"),
        req,
        grant,
        req["design"]["cost"],
        snapshot,
        lambda out: device_evidence(area, out, req, data[0]),
    )
    return combined(binding, pre, paid)


def preparation_cost(area, grant, snapshot, record):
    from oeis_learn.generalization.progressive.program_search_costing import workstation_accounting

    methods = sorted(record["units"]["methods"].keys() | record["units"]["unavailable"].keys())
    host = grant["state_root_id"]
    operations = {}
    for phase, expected in (("prepare", record["preparation"]), ("measure", record["measurement"])):
        logical = operation_key(coordinates(grant, phase), 0)["logical"]
        receipt = load(confined(area, "operations/" + logical[7:] + "/receipt.json"))
        if receipt["identity"] != expected or receipt != seal(receipt):
            raise ValueError("workstation cost differs from the verified role receipt")
        operations[logical] = (phase, receipt["attempt"])
    records = []
    boundaries = {}
    for key, window in snapshot["windows"].items():
        logical, number = key.split("/dispatch/")
        phase, completed = operations[logical]
        success = int(number) == completed
        state = "complete" if success else "open" if window["event"] == "RESERVE" else "failed"
        resources = {}
        if phase == "prepare":
            resources = dict(lease_us=0, covered_us=0, busy_us=0)
        elif success:
            resources = dict(
                lease_us=record["lease_us"],
                worker_cpu_us=record["worker_cpu_us"],
                covered_us=record["telemetry"]["covered_us"],
                busy_us=record["telemetry"]["busy_us"],
            )
        records.append(
            dict(
                window,
                origin="incremental",
                state=state,
                role=phase,
                receipt=key,
                host=host,
                beneficiaries=methods,
                **resources,
            )
        )
        if phase == "prepare" and int(number) == 0:
            boundaries["launch"] = dict(host=host, boot=window["boot"], at_us=window["start_us"])
        if phase == "measure" and success:
            boundaries["decision"] = dict(host=host, boot=window["boot"], at_us=window["end_us"])
    return workstation_accounting(records, **boundaries)


def checked_price_source(root, source, binding):
    """Retain historical prices across reader changes while pinning the measured workload."""
    if source is None:
        raise ValueError("candidate source required for timing reuse")
    current = collect_snapshot(source)
    if digest(current) == binding["source"]:
        return
    revision = binding["candidate"]
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("original timed candidate revision required")
    original = confined(root, "run-state/discovery-tools/round028-source/" + revision)
    # Execute the reader's verified helper, never a verifier taken from the cache being checked.
    helper = confined(source, "specs/015-learned-program-search/validation/candidate_source.py")
    runpy.run_path(str(helper))["verify_tree"](root, original, revision)
    previous = collect_snapshot(original)
    if digest(previous) != binding["source"]:
        raise ValueError("original timing source differs from the funded binding")
    # Only these out-of-worker readers, orchestration and tests may change.
    # Model, optimizer, calibration, input encoding, runtime, policy, and contracts
    # all retain their original identities. Acquisition is priced separately.
    host_only = {
        "progressive_round029.py", "progressive_velocity_timing_host.py",
        "progressive_velocity_verify.py", "progressive_velocity_wheel.py",
        "progressive_velocity_broad.py", "progressive_velocity_broad_host.py",
    }
    for name in previous.keys() | current.keys():
        if previous.get(name) == current.get(name) or name.startswith("tests/"):
            continue
        path = Path(name)
        if path.parent.as_posix() == "src/oeis_learn/pipeline_validation" and path.name in host_only:
            continue
        if name == "src/oeis_learn/generalization/progressive/program_search_costing.py":
            continue
        raise ValueError("workload changed since the original optimizer timing: " + name)


def timing_inventory(root, source):
    folder = confined(root, "run-state/discovery/" + SESSION, exists=False)
    if not folder.is_dir() or not (folder / "preparation/update-prices.json").exists():
        return None
    marker = load(confined(folder, "preparation/update-prices.json"))
    if marker != seal(marker):
        raise ValueError("original timing pointer differs")
    area = confined(folder, "qualification/" + marker["grant"][7:])
    grant, binding = [load(confined(area, p)) for p in ("grant.json", "binding.json")]
    if (
        marker["grant"] != grant["identity"]
        or pair_pointer(root) != binding["pair_pointer"]
    ):
        raise ValueError("price inputs differ from the funded donor census")
    checked_price_source(root, source, binding)
    approval = load(
        confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
    )
    validate_grant(
        grant,
        approval,
        root=physical_root_id(root),
        action="qualify",
        parent=timing_parent(binding["candidate"], binding["pair_pointer"]["receipt"]),
    )
    snapshot = BudgetJournal.inspect_existing(
        confined(area, "accounting"), grant["identity"], grant["resources"], include_windows=True
    )
    record = replay(root, area, binding, grant, snapshot)
    if (
        snapshot["pending"]
        or record != load(confined(area, "receipt.json"))
        or marker["receipt"] != record["identity"]
    ):
        raise ValueError("original timing receipt differs from the physical journal")
    return dict(
        candidate=binding["candidate"],
        measurement=record["measurement"],
        preparation_cost=preparation_cost(area, grant, snapshot, record),
        **record["units"],
    )


def qualify(args, root, folder, source, revision, grant, host_factory):
    from .progressive_round029 import result

    area = confined(folder, "qualification/" + grant["identity"][7:], exists=False)
    area.mkdir(parents=True, exist_ok=True)
    pointer = pair_pointer(root)
    if timing_parent(revision, pointer["receipt"]) != grant["registration_or_parent_ref"]:
        raise ValueError("timing authority belongs to another donor census")
    binding = bind(revision, digest(collect_snapshot(source)), grant, pointer)
    retain(area / "binding.json", binding)
    retain(area / "grant.json", grant)
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    base = dict(area=area, journal=journal, grant=grant, minimum_us=121_000_000, max_attempts=2)
    host = None
    data = None

    def inputs():
        nonlocal data
        if data is None:
            data = donor_inputs(root, pointer)
        return data

    def worker():
        nonlocal host
        if host is None:
            host = host_factory(args, source_root=source, workspace=root, revision=revision)
        return host

    def seconds(deadline):
        count = (deadline - time.monotonic_ns()) // 1_000_000_000 - 120
        if count < 1:
            raise InsufficientAuthority(
                "remaining timing budget cannot fund launch and the 120-second cleanup allowance"
            )
        return min(count, 86400)

    def acquire(out, deadline):
        projection, design = inputs()
        retain(area / "projection.json", projection)
        physical = worker()
        retain(out / "storage-before.json", storage(root, physical.engine))
        path = area / "request.json"
        runtime = None
        if path.exists():
            saved = load(confined(area, path))
            image = physical.engine.image(saved["image"])
            runtime = saved.get("runtime")
        elif args.image:
            image = physical.engine.image(args.image)
        else:
            from .progressive_velocity_runtime import cached_runtime, package_worker

            parent = (
                None
                if args.gpu_base_image
                else cached_runtime(root, folder, revision, physical.engine)
            )
            if parent is not None:
                image = parent["image"]
                runtime = package_worker(
                    root,
                    source,
                    area,
                    out,
                    revision,
                    parent,
                    binding["source"],
                    physical,
                    stage,
                    lambda: seconds(deadline),
                )
            else:
                allowance = seconds(deadline)
                logs = out / "image-build"
                logs.mkdir()
                image = physical.engine.build(
                    source,
                    target="gpu",
                    tag="oeis-learn:round029-" + revision,
                    base_image=args.gpu_base_image,
                    seconds=allowance,
                    evidence=logs,
                )
        req = seal(
            dict(
                version="round029-timing-request/v2" if runtime else "round029-timing-request/v1",
                **({"runtime": runtime} if runtime else {}),
                binding=binding["identity"],
                source=binding["source"],
                image=image,
                physical=physical_scope(physical, "hip", machine=True),
                role_policy=ROUND029_ROLE_POLICY,
                projection=reference(area, area / "projection.json"),
                design=design,
            )
        )
        retain(path, req)
        return req

    attempt_out = None

    def prepare(out, reserve):
        nonlocal attempt_out
        attempt_out = out
        deadline = time.monotonic_ns() + reserve * 1000
        with preparation_boundary(out, deadline):
            try:
                req = acquire(out, deadline)
            finally:
                if host is not None:
                    retain(out / "storage-after.json", storage(root, host.engine))
            from .progressive_pilot_capacity import owned_allocations
            from .progressive_research_inputs import checked_file

            transport = None
            if "runtime" in req:
                transport = owned_allocations(
                    [
                        checked_file(area, req["runtime"][k])
                        for k in ("archive", "bootstrap", "wheel")
                    ]
                )
            mounts = {k: load(confined(out, "storage-" + k + ".json")) for k in ("before", "after")}
            retain(
                out / "producer.json",
                seal(dict(request=req["identity"], storage=dict(mounts, transport=transport))),
            )

    def measure(out, reserve):
        nonlocal attempt_out
        attempt_out = out
        deadline = time.monotonic_ns() + reserve * 1000
        try:
            physical = worker()
            if (
                physical.engine.image(req["image"]) != req["image"]
                or physical_scope(physical, "hip", machine=True) != req["physical"]
            ):
                raise ValueError("registered HIP worker changed before timing")
            worker_args = [
                "python",
                "-I",
                "-m",
                "oeis_learn.pipeline_validation.progressive_velocity_timing",
            ]
            mounts = [
                (area / name, "/work/inputs/" + name)
                for name in ("request.json", "projection.json")
            ]
            if "runtime" in req:
                from .progressive_velocity_runtime import wheel_mounts

                extras = wheel_mounts(area, req["runtime"])
                mounts.extend(extras)
                worker_args = [
                    "python",
                    "-I",
                    "-B",
                    extras[0][1],
                    "run",
                    extras[1][1],
                    req["runtime"]["wheel"]["sha256"],
                    req["source"],
                ]
            allowance = seconds(deadline)
        except (Exception, KeyboardInterrupt):
            # Only synchronous metadata and file checks ran; stage owns cleanup
            # as soon as the physical command is handed to it.
            retain(
                out / "cleanup.json",
                dict(acknowledged=True, within_bound=time.monotonic_ns() <= deadline),
            )
            raise
        stage(
            physical,
            source,
            req["image"],
            out,
            worker_args,
            gpu=True,
            record_lease=True,
            seconds=allowance,
            readonly_inputs=mounts,
        )

    phase = "prepare"
    extra = {}
    try:
        pre = run_operation(
            **base,
            coordinates=coordinates(grant, phase),
            request=binding,
            cost={},
            execute=prepare,
            verify=lambda out: prepared(out, request_at(root, area, binding, inputs())),
        )
        req = request_at(root, area, binding, inputs())
        phase = "measure"
        paid = run_operation(
            **base,
            coordinates=coordinates(grant, phase),
            request=req,
            cost=req["design"]["cost"],
            execute=measure,
            verify=lambda out: device_evidence(area, out, req, inputs()[0]),
        )
        record = combined(binding, pre, paid)
        path = area / "receipt.json"
        retain(path, record)
        marker = folder / "preparation/update-prices.json"
        pointer = seal(dict(grant=grant["identity"], receipt=record["identity"]))
        if not marker.exists() or load(confined(folder, marker)) != pointer:
            revise(marker, pointer)
        state, code, reason, next_action = (
            "COMPLETE",
            0,
            "Replayed the ordinary-stream timing matrix and debits",
            "preflight",
        )
        extra = dict(
            last=record["identity"], receipt=record, receipt_path=str(path.relative_to(root))
        )
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) or type(exc).__name__
        path = area / "first-error.json"
        problem = dict(stage=phase, error=type(exc).__name__, reason=reason)
        if attempt_out is not None:
            problem["attempt_path"] = str(attempt_out.relative_to(root))
            extra["diagnostics"] = problem
        if not path.exists():
            retain(path, seal(problem))
        blocked = isinstance(exc, InsufficientAuthority)
        state, code, next_action = (
            ("BLOCKED", 2, "inspect_attempt")
            if blocked
            else ("FAILED", 130 if isinstance(exc, KeyboardInterrupt) else 1, "inspect_attempt")
        )
    snapshot = journal.snapshot()
    if (area / "first-error.json").is_file():
        extra["first_error"] = load(confined(area, "first-error.json"))
    value = result(
        "qualify",
        state,
        reason,
        next_action,
        scope="ordinary_update_timings",
        stage=phase,
        remaining=snapshot["remaining"]["elapsed_us"],
        elapsed_us=snapshot["consumed"]["elapsed_us"],
        charged_tokens=snapshot["consumed"]["tokens"],
        remaining_tokens=snapshot["remaining"]["tokens"],
        research_dispatches=0,
        **extra,
    )
    path = folder / "status.json"
    if not path.exists() or load(confined(folder, path)) != seal(value):
        revise(path, seal(value))
    return value, code
