"""Fund the origin audit, native stream preparation, and both broader HIP timings together."""

from pathlib import Path
import sys
import time

from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.campaign import learned_search_policy, velocity_policy
from oeis_learn.generalization.progressive.contracts import (
    digest,
    load,
    read_bound_record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.velocity_records import (
    SESSION,
    operation_key,
    physical_root_id,
    validate_grant,
)
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_handoff import stage
from .progressive_research_inputs import checked_file
from .progressive_roles import ROUND029_ROLE_POLICY
from .progressive_round028_receipts import verify_role_mounts
from .progressive_round028_rehearsal import physical_scope
from .progressive_velocity_broad import MAX_INPUT, checked_request, design, prices, supply_counts
from .progressive_velocity_census import SEED, prior_exposures, practice_prompts, training_source
from .progressive_velocity_image import preparation_boundary
from .progressive_velocity_operations import InsufficientAuthority, replay_operation, run_operation
from .progressive_velocity_preparation import funded_inventory
from .progressive_velocity_runtime import (
    cached_runtime,
    package_evidence,
    package_worker,
    wheel_mounts,
)
from .progressive_velocity_supply import _area
from . import progressive_velocity_timing_host as ordinary

VERSION = "round029-broader-timing/v1"
DRAW_BLOCK = 512
MARKER = "preparation/broader-prices.json"


def original_pointers(root, folder):
    ordinary_price = load(confined(folder, "preparation/update-prices.json"))
    families = load(confined(_area(root, "families"), "funded.json"))
    if ordinary_price != seal(ordinary_price) or families != seal(families):
        raise ValueError("original timing and family payments required")
    return dict(ordinary=ordinary_price, families=families)


def parent(revision, origins):
    return digest([VERSION, revision, velocity_policy(), DRAW_BLOCK, origins])


def authorize(root, minutes, revision):
    from oeis_learn.decoder.bigint_codec import MAX_BODY_TOKENS
    from .progressive_velocity_qualification import authorize_qualification

    folder = confined(root, "run-state/discovery/" + SESSION)
    # Two recipes, two support classes, three load bands and two samples, with one retry.
    attempts = 2 * 2 * 3 * 2 * 2
    return authorize_qualification(
        root,
        minutes,
        parent(revision, original_pointers(root, folder)),
        dict(attempts=attempts, tokens=attempts * 32 * MAX_BODY_TOKENS),
    )


def coordinates(grant, phase):
    return dict(ordinary.coordinates(grant, phase), method="broader_timing", stage="qualification")


def attempt(area, coords, grant):
    block = confined(area, "operations/" + operation_key(coords, 0)["logical"][7:])
    op = load(confined(block, "receipt.json"))
    if op != seal(op) or op["grant"] != grant["identity"]:
        raise ValueError("original paid dispatch required")
    return confined(block, f"attempt-{op['attempt']:04d}")


def paid_calibration(root, source, folder, origins):
    timed = ordinary.timing_inventory(root, source)
    marker = origins["ordinary"]
    area = confined(folder, "qualification/" + marker["grant"][7:])
    grant, rec = (load(confined(area, k + ".json")) for k in ("grant", "receipt"))
    if (
        not timed
        or rec["identity"] != marker["receipt"]
        or rec["measurement"] != timed["measurement"]
    ):
        raise ValueError("ordinary calibration must reopen with its original debit")
    out = attempt(area, ordinary.coordinates(grant, "measure"), grant)
    matrix = load(confined(out, "producer.json"))["matrix"]
    cal = matrix["calibration"]
    view = load(confined(area, "projection.json"), maximum=MAX_INPUT)
    return (
        dict(weight=cal["weight"], observation=cal["identity"], measurement=timed["measurement"]),
        view["initial_weights"],
        load(confined(area, "request.json"))["physical"],
    )


def custodian(root, source, out, origins):
    from oeis_learn.generalization.progressive.generator import draw
    from oeis_learn.generalization.progressive.program_search_panels import fresh_groups, reserved_family_rows
    from .progressive_panel_sources import historical_row, read_inputs
    from .progressive_preparation import verify_draw
    from .progressive_round028_audit import ORIGINAL

    folder = confined(root, "run-state/discovery/" + SESSION)
    cal, weights, machine = paid_calibration(root, source, folder, origins)
    freeze, pkg, originals = training_source(root)
    _, family = funded_inventory(root, "families", origins["families"], with_original=True)
    stock = read_bound_record(
        root, freeze["roster_source"], version="round028-private-panel-supply/v1"
    )
    panel = checked_file(root, freeze["roster_source"]).parent
    held = reference(panel, confined(panel, "calibration-reservation.json"))
    campaign = confined(root, ORIGINAL + "/prepare-research/frozen/campaign.json")
    registered = read_bound_record(
        root, reference(root, campaign), identity=learned_search_policy()["campaign"]
    )
    inputs = read_inputs(
        root, {r["name"]: r for r in registered["payload"]["references"]}, velocity=True
    )
    if pkg["exposure_cut"] != digest(
        [inputs["census"], inputs["data"]["identity"], stock["identity"], held]
    ):
        raise ValueError("original pilot exposure boundary differs")
    stream = pkg["stream"]
    rows = {r["draw_identity"]: r for r in pkg["rows"]}
    keys = stream["order"]
    if stream["row_identities"] != digest([rows[k]["identity"] for k in keys]):
        raise ValueError("original ordinary positive slots differ")
    original_refs = [
        *originals,
        freeze["roster_source"],
        reference(root, campaign),
        reference(root, panel / held["path"]),
        *inputs["originals_used"],
    ]
    refs = {r["path"]: r for r in original_refs}
    common = dict(
        origins=origins,
        stream=stream["identity"],
        ordinary_slots=len(keys),
        calibration=cal,
        initial_weights=weights,
        physical=machine,
        drawn=0,
    )

    def finish(disposition, counts, data=None):
        if data is not None:
            retain(out / "training.json", seal(data))
        retain(
            out / "custody.json",
            seal(
                dict(
                    common,
                    disposition=disposition,
                    counts=counts,
                    sources=[refs[p] for p in sorted(refs)],
                    training=reference(out, out / "training.json") if data else None,
                )
            ),
        )

    if not stock["calibrated"] or not stock["allocations"].get("calibration"):
        finish("requires_frozen_difficulty_map", {})
        return
    diff_path = confined(panel, "difficulty-map.json")
    difficulty = read_bound_record(
        panel,
        reference(panel, diff_path),
        identity=stock["allocations"]["calibration"],
        version="round028-difficulty-map/v1",
    )
    ref = reference(root, diff_path)
    refs[ref["path"]] = ref
    protected = prior_exposures(inputs, stock, read_bound_record(panel, held), [])
    wanted = {r["task"] for roster in family["panels"].values() for r in roster}
    reserved = {r["id"]: r for r in inputs["candidates"] if r["id"] in wanted}
    if set(reserved) != wanted:
        raise ValueError("original reserved family witnesses missing")
    protected.extend(
        historical_row(r, inputs["support"], "reservation", velocity=True)
        for r in reserved_family_rows(inputs["candidates"], list(reserved.values()))
    )
    familiar = set(inputs["support"]["structures"])

    def candidate(row):
        return dict(
            row,
            id=row["draw_identity"],
            stratum="parameter" if row["structure"] in familiar else "training",
        )

    protected.extend(candidate(rows[k]) for k in rows.keys() - set(keys))
    positives = [candidate(rows[k]) for k in keys]

    def eligible(cases):
        requested = {r["id"] for r in cases}
        bridges = [r for r in inputs["candidates"] if r["id"] not in requested]
        groups, counts = fresh_groups(
            [*cases, *bridges], protected, indexes=(), velocity=True, exposed_prompts=practice_prompts()
        )
        eligible_ids = {r["id"] for group in groups.values() for r in group} & requested
        counts.update(candidates=len(requested), eligible=len(eligible_ids),
                      excluded=len(requested - eligible_ids),
                      components=sum(any(r["id"] in requested for r in group) for group in groups.values()))
        return eligible_ids, counts

    original_keys, counts = eligible(positives)
    if original_keys != set(keys):
        finish("requires_fresh_disjoint_start", counts)
        return
    law = stream["law"]
    start = pkg["native_draws"]
    end = min(start + DRAW_BLOCK, law["max_draws"])
    additions = []
    for ordinal in range(start, end):
        row, _ = verify_draw(draw(law, ordinal))
        if row is not None:
            additions.append(candidate(row))
    clean, combined_counts = eligible([*positives, *additions])
    ordinals = [r["ordinal"] for r in additions if r["id"] in clean]
    common["drawn"] = end - start
    finish(
        "eligible",
        dict(ordinary=counts, combined=combined_counts, prospective_eligible=len(ordinals)),
        dict(
            stream=stream,
            ordinary_rows=[rows[k] for k in keys],
            familiar=sorted(familiar & {r["structure"] for r in positives}),
            difficulty=difficulty,
            eligible_ordinals=ordinals,
            pair_seed=SEED,
            profile=pkg["profile"],
            microbatch=pkg["microbatch"],
            initial_weights=weights,
            calibration=cal,
        ),
    )


def status_ok(out):
    process, cleanup = (load(confined(out, k + ".json")) for k in ("process", "cleanup"))
    if (
        process["outcome"] != "COMPLETE"
        or process["returncode"] != 0
        or process["cleanup_failure"]
        or cleanup["acknowledged"] is not True
        or cleanup["within_bound"] is not True
    ):
        raise ValueError("broader qualification worker did not terminate cleanly")


def cpu_evidence(root, area, binding, out):
    status_ok(out / "custodian")
    custody = load(confined(out, "custodian/custody.json"))
    if custody != seal(custody) or custody["origins"] != binding["origins"]:
        raise ValueError("broader preparation differs from its private donor parents")
    for ref in custody["sources"]:
        checked_file(root, ref)
    base = dict(
        custody=custody["identity"],
        disposition=custody["disposition"],
        counts=custody["counts"],
        ordinary_slots=custody["ordinary_slots"],
        drawn=custody["drawn"],
        calibration=custody["calibration"],
        physical=custody["physical"],
    )
    if custody["training"] is None:
        if custody["disposition"] == "requires_fresh_disjoint_start" and (
            custody["counts"]["candidates"] != custody["ordinary_slots"]
            or not custody["counts"]["excluded"]
        ):
            raise ValueError("fresh start requires a measured exclusion in the original stream")
        if custody["disposition"] not in (
            "requires_frozen_difficulty_map",
            "requires_fresh_disjoint_start",
        ):
            raise ValueError("unrecognized broader preparation disposition")
        return seal(dict(base, design=None))
    original = load(checked_file(out / "custodian", custody["training"]), maximum=MAX_INPUT)
    runtime = load(confined(out, "runtime.json"))
    package_evidence(root, area, runtime, binding)
    requested = load(confined(out, "native-request.json"), maximum=MAX_INPUT)
    if original != seal(original) or requested != role_request(
        binding, runtime, "prepare-broad", original
    ):
        raise ValueError("native preparation received different positive inputs")
    stage_out = confined(out, "native")
    check_stage(area, stage_out, runtime, out / "native-request.json", None, gpu=False)
    produced = load(confined(stage_out, "producer.json"))
    if produced != seal(produced) or produced["request"] != requested["identity"]:
        raise ValueError("native composition receipt differs")
    # Reopen all selected original rows, the complete audit and the signed stream,
    # even when the prescribed draws did not increase diversity.
    from .progressive_velocity_broad import projection
    from oeis_learn.generalization.progressive.generator import broader_stream

    (receipt_path,) = stage_out.glob(
        "run-state/discovery/" + SESSION + "/preparation/composition/*/receipt.json"
    )
    receipt = load(confined(stage_out, receipt_path), maximum=MAX_INPUT)
    base_dir = receipt_path.parent

    def reopen(ref):
        return load(checked_file(base_dir, ref), maximum=ref["bytes"])

    plan = load(confined(base_dir, "plan.json"))
    if (
        plan != seal(plan)
        or receipt != seal(receipt)
        or receipt["plan"] != plan["identity"]
        or plan["ordinary"] != requested["stream"]["identity"]
        or plan["ordinals"] != requested["eligible_ordinals"]
        or plan["calibration"] != requested["difficulty"]["identity"]
        or plan["law"] != digest(requested["stream"]["law"])
        or plan["familiar"] != requested["familiar"]
        or plan["source"] != binding["source"]
    ):
        raise ValueError("native composition draw plan differs from the funded request")
    stream, audit = reopen(receipt["stream"]), reopen(receipt["audit"])
    rows = {k: reopen(ref) for k, ref in receipt["rows"].items()}
    stats = produced["supply"]
    if (
        audit != seal(audit)
        or audit["plan"] != plan["identity"]
        or stream != seal(stream)
        or audit["identity"] != stream["audit"]
        or stats["audit"] != audit["identity"]
        or stats["stream"] != stream["identity"]
        or set(stream["order"]) != rows.keys()
        or any(
            r != seal(r) or r["draw_identity"] != k or r["identity"] != audit["witnesses"][k]
            for k, r in rows.items()
        )
    ):
        raise ValueError("native composition audit or selected rows differ")
    slots = audit["slots"]
    excluded = {r["draw"] for r in audit["rejections"]}
    reselected = broader_stream(
        [slots[k] for k in requested["stream"]["order"]],
        [slots[k] for k in sorted(slots.keys() - excluded) if slots[k]["band"] != "unknown"],
        seed=requested["stream"]["seeds"]["training_families"],
        ordinary_ref=plan["ordinary"],
    )
    if reselected["identity"] != audit["selection"] or stream != seal(
        dict(reselected, audit=audit["identity"])
    ):
        raise ValueError("broader slot selection differs from the recorded witnesses")
    expected_stats = supply_counts(stream, audit, requested)
    if produced["disposition"] != (
        "prepared" if stream["available"] else "no_breadth_at_matched_exposure"
    ):
        raise ValueError("broader availability differs from observed diversity")
    if stream["available"]:
        from oeis_learn.generalization.progressive.training_rows import matching_pairs
        from .progressive_velocity_broad import shards

        pairs = load(confined(stage_out, "pairs.json"), maximum=MAX_INPUT)
        if pairs != matching_pairs(
            shards(list(rows.values())), list(dict.fromkeys(stream["order"])), seed=SEED
        ):
            raise ValueError("broader matching support differs from native training rows")
        view = projection(stream, rows, pairs, requested)
        expected_stats.update(
            pair_dispositions=pairs["dispositions"],
            eligible_unique=pairs["eligible_rows"],
            eligible_slots=len(view["donors"]),
            eligible_families=pairs["eligible_families"],
        )
        if view != load(confined(stage_out, "projection.json"), maximum=MAX_INPUT) or produced[
            "design"
        ] != design(view, requested["microbatch"], custody["calibration"]):
            raise ValueError("broader projection or measured work design differs")
    elif produced["design"] is not None:
        raise ValueError("unavailable broader stream cannot authorize optimizer work")
    if stats != expected_stats:
        raise ValueError("reported broader support differs from the private census")
    return seal(
        dict(
            base,
            disposition=produced["disposition"],
            runtime=runtime["identity"],
            producer=produced["identity"],
            supply=stats,
            design=produced["design"],
        )
    )


def role_request(binding, runtime, role, fields):
    return checked_request(
        seal(
            dict(
                fields,
                role=role,
                binding=binding["identity"],
                source=binding["source"],
                role_policy=ROUND029_ROLE_POLICY,
                image=runtime["parent"]["image"],
            )
        ),
        role,
    )


def check_stage(area, out, runtime, request_path, projection_path, *, gpu):
    mounts = {"/work/inputs/request.json": (request_path, False), "/work/output": (out, True)}
    mounts.update({dest: (p, False) for p, dest in wheel_mounts(area, runtime)})
    if projection_path is not None:
        mounts["/work/inputs/projection.json"] = (projection_path, False)
    inspect = load(confined(out, "container-inspect.json"))
    verify_role_mounts(inspect, runtime["parent"]["image"], mounts)
    if not gpu and (
        inspect[0]["HostConfig"].get("Devices") or inspect[0]["HostConfig"].get("DeviceRequests")
    ):
        raise ValueError("native preparation must be CPU-only")
    status_ok(out)


def run_role(host, source, area, out, runtime, request_path, projection_path, *, gpu, seconds):
    extras = wheel_mounts(area, runtime)
    mounts = [*extras, (request_path, "/work/inputs/request.json")]
    if projection_path is not None:
        mounts.append((projection_path, "/work/inputs/projection.json"))
    stage(
        host,
        source,
        runtime["parent"]["image"],
        out,
        [
            "python",
            "-I",
            "-B",
            extras[0][1],
            "time-broad" if gpu else "prepare-broad",
            extras[1][1],
            runtime["wheel"]["sha256"],
            load(confined(area, "binding.json"))["source"],
        ],
        gpu=gpu,
        record_lease=gpu,
        readonly_inputs=mounts,
        seconds=seconds,
    )


def gpu_evidence(area, out, cpu, prepared, binding):
    from .progressive_generator_probe import usage_summary

    req = load(confined(area, "device-request.json"))
    runtime = load(confined(cpu, "runtime.json"))
    view_path = confined(cpu, "native/projection.json")
    expected = gpu_request(binding, runtime, prepared, view_path)
    if req != expected:
        raise ValueError("broader device request differs from verified native preparation")
    check_stage(area, out, runtime, area / "device-request.json", view_path, gpu=True)
    rec = load(confined(out, "producer.json"))
    if rec != seal(rec) or rec["device"] != "hip" or rec["request"] != req["identity"]:
        raise ValueError("broader device output differs")
    lease = load(confined(out, "gpu-lease.json"))["elapsed_us"]
    elapsed = rec["elapsed_us"]
    telemetry = usage_summary(confined(out, "telemetry/samples.jsonl").parent, elapsed)
    accounted = sum(
        m["startup_us"] + sum(s["elapsed_us"] for s in m["samples"].values())
        for m in rec["matrix"]["methods"].values()
    )
    if (
        type(elapsed) is not int
        or type(lease) is not int
        or type(rec["worker_cpu_us"]) is not int
        or rec["worker_cpu_us"] < 0
        or not 0 < accounted <= elapsed <= lease
        or rec["telemetry"] != telemetry
        or telemetry["covered_us"] > elapsed
        or type(rec["peak_device_bytes"]) is not int
        or rec["peak_device_bytes"] <= 0
    ):
        raise ValueError("broader device clocks, sampled activity or memory are inconsistent")
    return seal(
        dict(
            methods=prices(load(view_path, maximum=MAX_INPUT), req["design"], rec["matrix"]),
            producer=rec["identity"],
            lease_us=lease,
            telemetry=telemetry,
            elapsed_us=elapsed,
            peak_device_bytes=rec["peak_device_bytes"],
            worker_cpu_us=rec["worker_cpu_us"],
        )
    )


def gpu_request(binding, runtime, prepared, view_path):
    return role_request(
        binding,
        runtime,
        "time-broad",
        dict(
            preparation=prepared["identity"],
            physical=prepared["evidence"]["physical"],
            projection=reference(view_path.parent, view_path),
            design=prepared["evidence"]["design"],
        ),
    )


def combined(binding, prep, gpu):
    e = prep["evidence"]
    return seal(
        dict(
            version=VERSION,
            binding=binding["identity"],
            preparation=prep["identity"],
            disposition=e["disposition"],
            counts=e["counts"],
            ordinary_slots=e["ordinary_slots"],
            drawn=e["drawn"],
            supply=e.get("supply"),
            calibration=e["calibration"]["observation"],
            ordinary_measurement=e["calibration"]["measurement"],
            sample_work=e["design"]["cost"] if e["design"] else dict(attempts=0, tokens=0),
            measurement=gpu["identity"] if gpu else None,
            device=gpu["evidence"] if gpu else None,
        )
    )


def replay(root, area, grant, binding, snapshot):
    prep = replay_operation(
        area,
        coordinates(grant, "prepare"),
        binding,
        grant,
        {},
        snapshot,
        lambda out: cpu_evidence(root, area, binding, out),
    )
    cost = prep["evidence"]["design"]
    gpu = None
    if cost:
        cpu = attempt(area, coordinates(grant, "prepare"), grant)
        req = load(confined(area, "device-request.json"))
        gpu = replay_operation(
            area,
            coordinates(grant, "measure"),
            req,
            grant,
            cost["cost"],
            snapshot,
            lambda out: gpu_evidence(area, out, cpu, prep, binding),
        )
    return combined(binding, prep, gpu)


def inventory(root, source):
    folder = confined(root, "run-state/discovery/" + SESSION, exists=False)
    marker_path = folder / MARKER
    if not marker_path.is_file():
        return None
    marker = load(confined(root, marker_path))
    area = confined(folder, "qualification/" + marker["grant"][7:])
    grant, binding = (load(confined(area, k + ".json")) for k in ("grant", "binding"))
    if (
        marker != seal(marker)
        or marker["grant"] != grant["identity"]
        or binding != seal(binding)
        or binding["version"] != VERSION
        or binding["grant"] != grant["identity"]
        or original_pointers(root, folder) != binding["origins"]
    ):
        raise ValueError("broader price pointer or paid origins differ")
    ordinary.checked_price_source(root, source, binding)
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
        parent=parent(binding["candidate"], binding["origins"]),
    )
    snapshot = BudgetJournal.inspect_existing(
        confined(area, "accounting"), grant["identity"], grant["resources"]
    )
    record = replay(root, area, grant, binding, snapshot)
    if (
        snapshot["pending"]
        or record != load(confined(area, "receipt.json"))
        or marker["receipt"] != record["identity"]
    ):
        raise ValueError("broader receipt differs from paid operations")
    return record


def qualify(args, root, folder, source, revision, grant, host_factory):
    from .progressive_round029 import result
    from .supervision import launch
    from contextlib import redirect_stdout

    origins = original_pointers(root, folder)
    if grant["registration_or_parent_ref"] != parent(revision, origins):
        raise ValueError("broader timing grant belongs to different paid inputs")
    area = confined(folder, "qualification/" + grant["identity"][7:], exists=False)
    area.mkdir(parents=True, exist_ok=True)
    binding = seal(
        dict(
            version=VERSION,
            candidate=revision,
            source=digest(collect_snapshot(source)),
            grant=grant["identity"],
            origins=origins,
        )
    )
    retain(area / "binding.json", binding)
    retain(area / "grant.json", grant)
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    host = None

    def physical():
        nonlocal host
        if host is None:
            host = host_factory(args, source_root=source, workspace=root, revision=revision)
        return host

    def seconds(deadline):
        left = (deadline - time.monotonic_ns()) // 1_000_000_000 - 120
        if left < 1:
            raise InsufficientAuthority(
                "broader qualification has insufficient time left for dispatch and cleanup"
            )
        return left

    attempt_out = None

    def prepare(out, reserve):
        nonlocal attempt_out
        attempt_out = out
        deadline = time.monotonic_ns() + reserve * 1000
        with preparation_boundary(out, deadline):
            c = out / "custodian"
            c.mkdir()
            argv = [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('oeis_learn.pipeline_validation.progressive_velocity_broad_host',run_name='__main__')",
                str(source / "src"),
                str(root),
                str(source),
                str(c),
                str(area / "binding.json"),
            ]
            try:
                with redirect_stdout(sys.stderr):
                    launch(argv, c, seconds=seconds(deadline), cleanup_seconds=5)
            finally:
                if (c / "process.json").is_file():
                    proc = load(confined(c, "process.json"))
                    retain(
                        c / "cleanup.json",
                        dict(
                            acknowledged=proc["cleanup_failure"] is None,
                            within_bound=time.monotonic_ns() <= deadline,
                        ),
                    )
            custody = load(confined(c, "custody.json"))
            if custody["training"] is None:
                return
            data = load(checked_file(c, custody["training"]), maximum=MAX_INPUT)
            worker = physical()
            old = cached_runtime(root, folder, revision, worker.engine)
            if old is None:
                raise ValueError(
                    "the compatible settled learner runtime is required for broader timing"
                )
            runtime = package_worker(
                root,
                source,
                area,
                out,
                revision,
                old,
                binding["source"],
                worker,
                stage,
                lambda: seconds(deadline),
            )
            retain(out / "runtime.json", runtime)
            req = out / "native-request.json"
            retain(req, role_request(binding, runtime, "prepare-broad", data))
            native = out / "native"
            native.mkdir()
            run_role(
                worker,
                source,
                area,
                native,
                runtime,
                req,
                None,
                gpu=False,
                seconds=seconds(deadline),
            )

    kwargs = dict(area=area, grant=grant, journal=journal, minimum_us=121_000_000, max_attempts=2)
    phase = "prepare"
    extra = {}
    try:
        prep = run_operation(
            **kwargs,
            coordinates=coordinates(grant, phase),
            request=binding,
            cost={},
            execute=prepare,
            verify=lambda out: cpu_evidence(root, area, binding, out),
        )
        gpu = None
        if prep["evidence"]["design"]:
            phase = "measure"
            cpu = attempt(area, coordinates(grant, "prepare"), grant)
            runtime = load(confined(cpu, "runtime.json"))
            path = confined(cpu, "native/projection.json")
            req = gpu_request(binding, runtime, prep, path)
            retain(area / "device-request.json", req)

            def measure(out, reserve):
                nonlocal attempt_out
                attempt_out = out
                deadline = time.monotonic_ns() + reserve * 1000
                worker = physical()
                try:
                    if (
                        worker.engine.image(req["image"]) != req["image"]
                        or physical_scope(worker, "hip", machine=True) != req["physical"]
                    ):
                        raise ValueError(
                            "registered broader runtime image or physical device differs"
                        )
                    allowance = seconds(deadline)
                except (Exception, KeyboardInterrupt):
                    retain(
                        out / "cleanup.json",
                        dict(acknowledged=True, within_bound=time.monotonic_ns() <= deadline),
                    )
                    raise
                run_role(
                    worker,
                    source,
                    area,
                    out,
                    runtime,
                    area / "device-request.json",
                    path,
                    gpu=True,
                    seconds=allowance,
                )

            gpu = run_operation(
                **kwargs,
                coordinates=coordinates(grant, phase),
                request=req,
                cost=req["design"]["cost"],
                execute=measure,
                verify=lambda out: gpu_evidence(area, out, cpu, prep, binding),
            )
        rec = combined(binding, prep, gpu)
        retain(area / "receipt.json", rec)
        marker = seal(dict(grant=grant["identity"], receipt=rec["identity"]))
        if not (folder / MARKER).exists() or load(confined(folder, MARKER)) != marker:
            revise(folder / MARKER, marker)
        state, code, reason, nxt = "COMPLETE", 0, rec["disposition"], "preflight"
        extra.update(
            last=rec["identity"],
            receipt=rec,
            receipt_path=str((area / "receipt.json").relative_to(root)),
        )
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) or type(exc).__name__
        if attempt_out is not None:
            failed = attempt_out / "custodian/failure.json"
            if phase == "prepare" and failed.is_file():
                reason = load(confined(area, failed))["reason"]
        problem = dict(stage=phase, error=type(exc).__name__, reason=reason)
        if attempt_out is not None:
            problem["attempt_path"] = str(attempt_out.relative_to(root))
        if not (area / "first-error.json").exists():
            retain(area / "first-error.json", seal(problem))
        extra["diagnostics"] = problem
        state, code, nxt = (
            ("BLOCKED", 2, "inspect_attempt")
            if isinstance(exc, InsufficientAuthority)
            else ("FAILED", 1, "inspect_attempt")
        )
    if (area / "first-error.json").exists():
        extra["first_error"] = load(confined(area, "first-error.json"))
    snapshot = journal.snapshot()
    value = result(
        "qualify",
        state,
        reason,
        nxt,
        scope="broader_update_timings",
        stage=phase,
        remaining=snapshot["remaining"]["elapsed_us"],
        elapsed_us=snapshot["consumed"]["elapsed_us"],
        charged_tokens=snapshot["consumed"]["tokens"],
        remaining_tokens=snapshot["remaining"]["tokens"],
        research_dispatches=0,
        **extra,
    )
    if not (folder / "status.json").exists() or load(confined(folder, "status.json")) != seal(
        value
    ):
        revise(folder / "status.json", seal(value))
    return value, code


if __name__ == "__main__":
    root, source, out, binding = map(Path, sys.argv[1:])
    try:
        custodian(root, source, out, load(binding)["origins"])
    except Exception as exc:
        retain(out / "failure.json", seal(dict(reason=str(exc), error=type(exc).__name__)))
        raise
