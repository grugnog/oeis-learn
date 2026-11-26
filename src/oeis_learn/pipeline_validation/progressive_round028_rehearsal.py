"""Candidate-bound measurements before spending on the historically selected generator."""
import math
import os
import time
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES, atomic_write, digest, load, reference, seal,
)
from oeis_learn.pipeline_validation.identity import confined
from .preparation_recovery import retain, revise
from .progressive_research_inputs import checked_file
from .progressive_round028_audit import ORIGINAL, Donor
from .research_limits import CPU_ACCEPTANCE_SECONDS, DEVICE_ACCEPTANCE_SECONDS, IMAGE_SECONDS


def sample_stream(base, registration, support_identity):
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.training_rows import matched_batch, matched_stream
    if base["identity"] != support_identity:
        raise ValueError("rehearsal must reopen the training producer from the original diagnostic cut")
    stream = matched_stream(base["shards"], base["layout"], registration, 0)
    if not stream["eligible_rows"]:
        return None
    labels = {p["positive_row"]: p for p in stream["pairs"]}
    rows = {r["draw_identity"]: r for shard in base["shards"] for r in shard}
    # Start at the earliest registered whole batch with an actionable training label.
    offset = next(i for i, key in enumerate(stream["order"]) if key in labels) // 32 * 32
    batches = []
    for cursor in (offset, offset + 32):
        samples = []
        for key in matched_batch(stream, cursor, 32):
            pair = labels.get(key)
            samples.append(dict(positive=rows[key], negative=rows[pair["negative_row"]] if pair else None,
                                label=pair["label"] if pair else None))
        batches.append(samples)
    # Trial 05 is a learning-setting trial in the reference architecture.
    profile = resolve_profile(base["profile"]["base_recipe"], shape="reference",
                              seed=stream["seeds"]["initialization"], dropout=0)
    return seal(dict(version="round028-training-probe/v1", registration=registration, base=base["identity"],
                     stream=stream["identity"], profile=profile, microbatch=base["update_policy"]["microbatch_size"],
                     offset=offset, batches=batches, support={k: stream[k] for k in
                     ("eligible_rows", "eligible_families", "positions")}))


def space(path):
    meta = os.statvfs(path)
    return dict(free_bytes=meta.f_bavail * meta.f_frsize, free_inodes=meta.f_favail)


def original_scale(cut):
    dimensions = dict(campaign_bytes=0, candidate_score_bytes=0, largest_input_bytes=0, input_bytes=0, input_files=0)
    for ref in cut["inputs"]:
        size, path = ref["bytes"], ref["path"]
        dimensions["largest_input_bytes"] = max(size, dimensions["largest_input_bytes"])
        dimensions["input_bytes"] += size
        dimensions["input_files"] += 1
        if path == ORIGINAL + "/prepare-research/frozen/campaign.json":
            dimensions["campaign_bytes"] = size
        if "candidate" in path or "score" in path:
            dimensions["candidate_score_bytes"] += size
    return dimensions


def physical_scope(host, phase):
    # A boot, driver, card, or engine change measures a new physical scope.
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    device = {}
    if phase == "hip":
        host.select_gpu()
        node = Path(host.policy["render_node"])
        card = Path("/sys/class/drm") / node.name / "device"
        device["bus"] = str(card.resolve(strict=True))
        device["rdev"] = node.stat().st_rdev
        for key in ("unique_id", "vendor", "device", "revision"):
            path = card / key
            device[key] = path.read_text().strip() if path.is_file() else None
    return digest([boot, host.policy["runtime"], device])


def failure_locations(source, area, *, role_folder="stages"):
    """Release candidate code coordinates only, including the isolated worker traceback."""
    import re
    import traceback
    parts = [traceback.format_exc()]
    for path in sorted(area.glob(role_folder+"/*/attempt-*/output/stderr.log"),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True)[:1]:
        with confined(area, path.relative_to(area)).open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 32768))
            parts.append(stream.read(32768).decode(errors="replace"))
    locations = []
    for original, number in re.findall(r'File "([^"\n]+)", line ([0-9]+)', "\n".join(parts)):
        parsed = Path(original)
        for marker in ("oeis_learn", "scripts"):
            if marker not in parsed.parts:
                continue
            local = Path(*parsed.parts[parsed.parts.index(marker):])
            if marker == "oeis_learn":
                local = "src" / local
            if (source / local).is_file():
                locations.append([str(local), int(number)])
    return locations[-12:]


def measure_stage(host, source, folder, base_path, stage, image, request, *, invoke):
    base = confined(folder, "stages/" + request["identity"][7:], exists=False)
    base.mkdir(parents=True, exist_ok=True)
    req_path = confined(base, "request.json", exists=False)
    retain(req_path, request)
    seconds = CPU_ACCEPTANCE_SECONDS if stage == "cpu" else DEVICE_ACCEPTANCE_SECONDS
    receipt = confined(base, "receipt.json", exists=False)
    reused = receipt.is_file()
    if not reused:
        # A crash never overwrites the elapsed work, logs or partial checkpoints of its attempt.
        ordinal = len(list(base.glob("attempt-*")))
        attempt = confined(base, f"attempt-{ordinal:04d}", exists=False)
        attempt.mkdir()
        began = time.monotonic_ns()
        atomic_write(attempt / "start.json", seal(dict(request=request["identity"], seconds=seconds,
                                                        wall_ns=time.time_ns(), space=space(folder))))
        if host.engine.image(image) != image:
            raise ValueError("candidate image changed before dispatch")
        output = attempt / "output"
        output.mkdir()
        try:
            invoke(host, source, image, output,
                   ["python", "-I", "-m", "oeis_learn.pipeline_validation.progressive_generator_probe"],
                   gpu=stage == "hip", seconds=seconds, record_lease=stage == "hip", readonly_inputs=[
                       (base_path, "/work/inputs/training.json"), (req_path, "/work/inputs/request.json")])
        finally:
            atomic_write(attempt / "end.json", seal(dict(request=request["identity"],
                         elapsed_us=(time.monotonic_ns() - began) // 1000, space=space(folder))))
        value = seal(dict(version="round028-probe-boundary/v1", request=request["identity"],
                          start=reference(base, attempt / "start.json"), end=reference(base, attempt / "end.json"),
                          producer=reference(base, output / "producer.json"),
                          inspection=reference(base, output / "container-inspect.json"),
                          lease=reference(base, output / "gpu-lease.json") if stage == "hip" else None))
        validate_boundary(base, request, value)
        retain(receipt, value)
    return read_stage(base, request, reused=reused)


def read_stage(base, request, *, reused):
    value = load(confined(base, "receipt.json"))
    producer, elapsed = validate_boundary(base, request, value)
    spent = 0
    unknown = 0
    attempts = sorted(base.glob("attempt-*"))
    for attempt in attempts:
        start = load(confined(base, (attempt / "start.json").relative_to(base)))
        if seal(start) != start or start["request"] != request["identity"]:
            raise ValueError("wrong original attempt charge")
        if (attempt / "end.json").exists():
            end = load(confined(base, (attempt / "end.json").relative_to(base)))
            if seal(end) != end or end["request"] != request["identity"] or end["elapsed_us"] <= 0:
                raise ValueError("changed stage time receipt")
            spent += end["elapsed_us"]
        else:
            spent += start["seconds"] * 1_000_000
            unknown += 1
    return producer, dict(reused=reused, elapsed_us=elapsed, all_attempts_us=spent,
                          unknown_intervals=unknown, attempts=len(attempts),
                          lease_us=load(confined(base, value["lease"]["path"]))["elapsed_us"] if value["lease"] else 0)


def validate_boundary(root, request, value):
    if (seal(value) != value or value["version"] != "round028-probe-boundary/v1"
            or value["request"] != request["identity"]):
        raise ValueError("wrong rehearsal boundary")
    for k in ("producer", "inspection", "start", "end"):
        checked_file(root, value[k])
    producer = load(confined(root, value["producer"]["path"]))
    end = load(confined(root, value["end"]["path"]))
    if (seal(producer) != producer or producer["version"] != "round028-probe-producer/v1"
            or producer["request"] != request["identity"] or producer["stage"] != request["stage"]
            or seal(end) != end or end["request"] != request["identity"] or end["elapsed_us"] <= 0):
        raise ValueError("current rehearsal source and charged stage required")
    if request["stage"] == "hip":
        checked_file(root, value["lease"])
        lease = load(confined(root, value["lease"]["path"]))
        if type(lease.get("elapsed_us")) is not int or not 0 < lease["elapsed_us"] <= end["elapsed_us"]:
            raise ValueError("lease time exceeds the original workstation stage")
    details = producer["details"]
    if request["stage"] == "cpu":
        if (not math.isfinite(details["gradient_l2"]) or details["gradient_l2"] <= 0
                or details["parameter_bytes"] <= 0 or details["native_verified_terms"] != 200):
            raise ValueError("nonzero parameter gradient and native pair required")
    else:
        a, b = (details["arms"][k] for k in ("ordinary", "first_divergence"))
        if (any(a[k] != b[k] for k in ("initial", "row_schedule", "charged_tokens"))
                or a["final"] == b["final"] or b["preference_rows"] <= 0
                or any(not v["actual_update"] or not v["full_restore"] or v["committed_updates"] != 2
                       or len(v["charged_tokens"]) != 2 or min(v["charged_tokens"] + v["update_us"]) <= 0
                       for v in (a, b))):
            raise ValueError("matched physical update and exact restore required")
    return producer, end["elapsed_us"]


CPU_MEASURES = ("gradient_l2", "gradient_loss", "gradient_probe_rows", "native_verified_terms", "parameter_bytes")
ARM_MEASURES = ("charged_tokens", "update_us", "preference_rows", "save_us", "restore_us",
                "replay_us", "checkpoint_bytes", "checkpoint_files", "full_restore", "actual_update")
TELEMETRY = ("samples", "readable_samples", "covered_us", "busy_us", "role_wall_us")


def stage_measurements(phase, producer, cost, *, image, build_us, image_reused):
    details = producer["details"]
    common = dict(cost, image=image, build_us=build_us, image_reused=image_reused, producer_us=producer["elapsed_us"])
    if phase == "cpu":
        return dict(common, **{k:details[k] for k in CPU_MEASURES})
    arms = {arm:{k:details["arms"][arm][k] for k in ARM_MEASURES} for arm in ("ordinary", "first_divergence")}
    return dict(common, arms=arms, matched=True, peak_device_bytes=details["peak_device_bytes"],
                telemetry={k:details["telemetry"][k] for k in TELEMETRY})


def training_forecast(arms):
    # Actual complete update wall times include synchronization. A two-update sample
    # is a planning observation; case/portfolio and replication timing remain separate.
    rates = {k: [t / n for t, n in zip(v["update_us"], v["charged_tokens"], strict=True)] for k, v in arms.items()}
    pilot = {k: math.ceil(4_000_000 * max(v)) for k, v in rates.items()}
    target = 8 * 3600 * 1_000_000
    return dict(pilot_training_us=pilot, combined_training_us=sum(pilot.values()), shift_us=target,
                evaluation_replication_reserve_us=target // 2,
                projected_training_excess_us=max(0, sum(pilot.values()) - target // 2),
                basis="slower observed complete update per arm; evaluation and replication unmeasured")


def candidate_image(host, source, area, revision, phase):
    """Use the same immutable candidate image and preserve its original build charge."""
    import subprocess
    opts = dict(revision=revision, phase=phase, engine=host.engine.name,
                base=host.args.gpu_base_image if phase == "hip" else None,
                override=host.args.image if phase == "hip" else None)
    path = area / ("image-" + digest(opts)[7:] + ".json")
    if path.exists():
        saved = load(path)
        if seal(saved) != saved or saved["options"] != opts:
            raise ValueError("candidate image acquisition changed")
        try:
            if host.engine.image(saved["image"]) == saved["image"]:
                return saved["image"], saved["elapsed_us"], True
        except subprocess.CalledProcessError:
            pass
    began = time.monotonic_ns()
    image = (host.engine.build(source, target="qualification", tag="oeis-learn:round028-cpu-" + revision,
                               seconds=IMAGE_SECONDS) if phase == "cpu" else host.ensure())
    us = (time.monotonic_ns() - began) // 1000
    revise(path, seal(dict(options=opts, image=image, elapsed_us=us)))
    return image, us, False


def rehearse(args, root, source, folder, store, revision, audit_result, *, host_factory, output_area=None):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from .progressive_handoff import stage as invoke
    from .progressive_round028 import reopen
    from .progressive_round028_support import training_package
    if audit_result["decision"]["action"] != "MATCHED_GENERATOR":
        return dict(mechanical_status="SKIPPED", reason="GENERATOR_NOT_SELECTED", gpu_dispatches=0), 0
    cut = reopen(root, folder, store.campaign, store.state["audit"], recheck_inputs=False)
    donor = Donor(root, folder, store.campaign)
    original = donor.saved(ORIGINAL + "/prepare-research/frozen/campaign.json", donor.policy["campaign"],
                           maximum=CAMPAIGN_DOCUMENT_BYTES)
    refs = {r["name"]: r for r in original["payload"]["references"]}
    package = training_package(donor, refs)
    sample = sample_stream(package, store.campaign["identity"], cut["support"]["input"])
    output = dict(mechanical_status="BLOCKED", scientific_status="PILOT_UNMEASURED", candidate=revision,
                  audit=store.state["publication"], decision=audit_result["decision"],
                  gpu_dispatches=0, scope="training_only_rehearsal", filesystem=space(folder),
                  original_scale=original_scale(cut))
    if sample is None:
        return dict(output, reason="NO_TRAINING_SUPPORT_OUTSIDE_PROBES"), 2
    area = confined(folder, output_area or ("private/generator-rehearsal/" + revision), exists=False)
    area.mkdir(parents=True, exist_ok=True)
    base_path = confined(area, "training.json", exists=False)
    retain(base_path, sample)
    output["support"] = sample["support"]
    src = digest(collect_snapshot(source))
    active_path = area / "progress.json"
    parent = None
    for phase in ("cpu", "hip"):
        output["phase"] = phase
        revise(active_path, seal(output))
        try:
            if phase == "cpu":
                host = host_factory(args, source_root=source, workspace=root, revision=revision)
            before = space(folder)
            # Report a planning estimate for two arm checkpoints and overlapping writes.
            # The inherited operator free-space floor owns the stop decision.
            checkpoint_room = 12 * output["cpu"]["parameter_bytes"] if phase == "hip" else 0
            output["estimated_checkpoint_bytes"] = checkpoint_room
            required = host.policy["free_disk_floor_bytes"]
            if before["free_bytes"] < required:
                return dict(output, reason="FREE_SPACE_FLOOR_DEFICIT", required_bytes=required,
                            deficit_bytes=required-before["free_bytes"], filesystem=before), 2
            physical = physical_scope(host, phase)
            image, build_us, build_reused = candidate_image(host, source, area, revision, phase)
            if space(folder)["free_bytes"] < required:
                return dict(output, reason="POST_BUILD_STORAGE_DEFICIT", required_bytes=required,
                            deficit_bytes=required-space(folder)["free_bytes"]), 2
            request = seal(dict(version="round028-probe-request/v1", source=src, image=image,
                                engine=host.engine.name, stage=phase, data=sample["identity"], parent=parent,
                                physical=physical))
            producer, cost = measure_stage(host, source, area, base_path, phase, image, request, invoke=invoke)
            parent = producer["identity"]
            output[phase] = stage_measurements(phase, producer, cost, image=image, build_us=build_us,
                                               image_reused=build_reused)
            if phase == "hip":
                output["gpu_dispatches"] = int(not cost["reused"])
                output["forecast"] = training_forecast(output["hip"]["arms"])
            output["filesystem"] = space(folder)
        except Exception as exc:
            # Only aggregates and code locations travel. Preserve raw diagnostics on the workstation.
            import traceback
            (area / (phase + "-exception.txt")).write_text(traceback.format_exc())
            output.update(reason=phase.upper()+"_REHEARSAL_FAILED", error_type=type(exc).__name__,
                          failure_locations=failure_locations(source, area))
            if phase == "hip":
                output["gpu_dispatches"] = None  # Failure could have occurred before or after device acquisition.
            revise(active_path, seal(output))
            return output, 2
    result = dict(output, mechanical_status="COMPLETE", phase="complete", reason="MATCHED_REHEARSAL_MEASURED",
                  next_action="REVIEW_TRAINING_FORECAST" if output["forecast"]["projected_training_excess_us"]
                  else "FREEZE_FRESH_PANELS_AND_MEASURE_EVALUATION")
    revise(active_path, seal(result))
    return result, 0
