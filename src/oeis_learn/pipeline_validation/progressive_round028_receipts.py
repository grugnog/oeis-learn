"""Read the pinned workstation's original rehearsal receipts with no engine or learner."""
import fcntl
from pathlib import Path
import re
import runpy
import subprocess
import zipfile

from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES, digest, load, read_bound_record, seal,
)
from oeis_learn.generalization.progressive.program_search_report import verify_archive
from oeis_learn.generalization.progressive.program_search_rehearsal_report import (
    ARMS, PHASES, VERSION, validate_public,
)
from .identity import confined
from .progressive_round028_rehearsal import original_scale, read_stage, stage_measurements, training_forecast


class ReceiptGap(ValueError):
    """A safe, non-private reason plus any independently verified preceding stage."""
    def __init__(self, reason, **observed):
        self.reason = reason
        self.observed = observed
        super().__init__(reason)


def sealed(path, *, maximum=1 << 20):
    value = load(path, maximum=maximum)
    if not isinstance(value, dict) or seal(value) != value:
        raise ReceiptGap("SAVED_RECORD_SEAL_DIFFERS")
    return value


def image_acquisition(area, revision, stage, image):
    matches = []
    for path in sorted(area.glob("image-*.json")):
        value = sealed(confined(area, path.name))
        opts = value["options"]
        if (opts.get("revision"), opts.get("engine"), opts.get("phase"), value["image"]) != (
                revision, "podman", stage, image):
            continue
        if path.name != "image-" + digest(opts)[7:] + ".json" or type(value["elapsed_us"]) is not int:
            raise ReceiptGap("IMAGE_ACQUISITION_DIFFERS")
        matches.append(value["elapsed_us"])
    if len(matches) != 1:
        raise ReceiptGap("IMAGE_ACQUISITION_MISSING_OR_AMBIGUOUS")
    return matches[0]


def verify_role_mounts(item, image, expected):
    from oeis_learn.container_runtime import normalized_image
    if not isinstance(item, list) or len(item) != 1:
        raise ReceiptGap("CONTAINER_INSPECTION_INCOMPLETE")
    record = item[0]
    policy = record["HostConfig"]
    binds = [m for m in record["Mounts"] if m["Type"] == "bind"]
    mounted = {m["Destination"]:m for m in binds}
    if (normalized_image(record["Image"]) != image or policy["NetworkMode"] != "none"
            or policy["ReadonlyRootfs"] is not True or policy["Privileged"] or len(binds) != len(expected)
            or set(mounted) != set(expected)
            or any((mounted[k]["Source"], mounted[k]["RW"]) != (str(path), writable)
                   for k,(path,writable) in expected.items())):
        raise ReceiptGap("ORIGINAL_IMAGE_OR_ROLE_MOUNTS_DIFFER")


def inspection(root, boundary, request, area):
    item = load(confined(root, boundary["inspection"]["path"]), maximum=CAMPAIGN_DOCUMENT_BYTES)
    output = confined(root, Path(boundary["producer"]["path"]).parent)
    expected = {"/work/inputs/training.json": (area / "training.json", False),
                "/work/inputs/request.json": (root / "request.json", False), "/work/output": (output, True)}
    verify_role_mounts(item,request["image"],expected)
    return output


def checkpoint_closures(output, training, details, *, weight_slices=None):
    from oeis_learn.generalization.state import restore
    from .progressive_training import resolve_update_policy

    ids = [[s["positive"]["draw_identity"] for s in batch] for batch in training["batches"]]
    charges = [sum(len(s["positive"]["tokens"]) for s in batch) for batch in training["batches"]]
    preferred = sum(s["label"] is not None for s in training["batches"][0])
    result = {}
    for arm in ARMS:
        measured = details["arms"][arm]
        if (measured["charged_tokens"] != charges or measured["row_schedule"] != digest(ids)
                or measured["preference_rows"] != (preferred if arm == "first_divergence" else 0)):
            raise ReceiptGap("SAMPLED_ROWS_OR_CHARGED_TOKENS_DIFFER")
        try:
            root = confined(output, arm)
            ref = load(confined(root, "generalization-latest.json"))
            state = read_bound_record(root, ref, version="generalization-learner-state/v1")
            refs = [v for k,v in state["states"].items() if k not in {"data", "rng"}]
            refs.extend(state["states"]["rng"].values())
            if any(not isinstance(v, dict) or set(v) != {"path", "bytes", "sha256"} for v in refs):
                raise ReceiptGap("INCOMPLETE_FULL_STATE_BLOB_REFERENCES")
            for ref in refs:
                confined(root, ref["path"])
            policy = resolve_update_policy(training["profile"], objective=arm, effective_batch=32,
                                           microbatch_size=training["microbatch"], schedule_horizon_tokens=4_000_000,
                                           weight_decay=0)
            contract = digest([policy["identity"], ids])
            # The original restore verifies every blob before returning its ledger. It does not unpickle tensors.
            ledger = restore(state, contract=contract, cap=sum(charges), root=root)["ledger"]
            if ledger.updates != 1 or ledger.consumed != charges[0] or ledger.pending or state["states"]["data"] != {"cursor": 32, "pending_batch": None}:
                raise ReceiptGap("FULL_CHECKPOINT_EXPOSURE_DIFFERS")
            files = [p for p in root.rglob("*") if p.is_file()]
            total = sum(confined(root, p.relative_to(root)).stat().st_size for p in files)
            if (len(files), total) != (measured["checkpoint_files"], measured["checkpoint_bytes"]):
                raise ReceiptGap("ORIGINAL_CHECKPOINT_FILE_ACCOUNTING_DIFFERS")
            if weight_slices is not None:
                from .progressive_research_inputs import checked_file
                weight = state["states"]["weights"]
                weight_slices[arm] = (checked_file(root, weight), weight)
            result[arm] = "VERIFIED"
        except FileNotFoundError:
            result[arm] = "UNAVAILABLE"
    return result


def stage_chain(area, *, revision, source, training, progress, weight_slices=None):
    parent = None
    measurements = {}
    closures = {}
    capacity = None
    inventory = []
    for path in sorted(area.glob("stages/*/request.json")):
        request = sealed(confined(area, path.relative_to(area)))
        if (set(request) != {"version", "identity", "source", "data", "engine", "image", "stage", "parent", "physical"}
                or request["version"] != "round028-probe-request/v1" or path.parent.name != request["identity"][7:]
                or not re.fullmatch("sha256:[0-9a-f]{64}", request["physical"])
                or request.get("engine") != "podman" or request.get("source") != source
                or request.get("data") != training["identity"] or request.get("stage") not in ("cpu", "hip")):
            raise ReceiptGap("SAVED_STAGE_REQUEST_DIFFERS")
        inventory.append((request, path.parent))
    for stage in ("cpu", "hip"):
        matched = []
        for req, root in inventory:
            if req["stage"] != stage or req["parent"] != parent or not (root / "receipt.json").is_file():
                continue
            if progress.get(stage, {}).get("image", req["image"]) != req["image"]:
                continue
            boundary = sealed(confined(root, "receipt.json"))
            producer, cost = read_stage(root, req, reused=True)
            output = inspection(root, boundary, req, area)
            original_build = image_acquisition(area, revision, stage, req["image"])
            result = stage_measurements(stage, producer, cost, image=req["image"], build_us=original_build,
                                        image_reused=True)
            old = progress.get(stage)
            if old and any(old.get(k) != v for k,v in result.items() if k not in ("reused", "image_reused")):
                continue
            matched.append((producer, result, output, root, boundary))
        if len(matched) != 1:
            raise ReceiptGap("MISSING_OR_AMBIGUOUS_" + stage.upper() + "_BOUNDARY",
                             matched_boundaries=len(matched), **measurements)
        producer, result, output, root, boundary = matched[0]
        measurements[stage] = result
        parent = producer["identity"]
        if stage == "hip":
            closures = checkpoint_closures(output, training, producer["details"], weight_slices=weight_slices)
            capacity = sealed(confined(root, boundary["end"]["path"]))["space"]
    return measurements, closures, capacity


def verify_sample_origin(base, training, registration):
    from collections import defaultdict
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.training_rows import first_divergence, pair_seeds
    seeds = pair_seeds(registration, 0)
    expected = resolve_profile(base["profile"]["base_recipe"], shape="reference", seed=seeds["initialization"], dropout=0)
    if (training["profile"] != expected or base["scope"] != "NATIVE_VERIFIED"
            or training["microbatch"] != base["update_policy"]["microbatch_size"]):
        raise ReceiptGap("ORIGINAL_TRAINING_PROFILE_DIFFERS")
    rows = {r["draw_identity"]:r for shard in base["shards"] for r in shard}
    probes = {p["id"] for p in base["layout"]["probes"].values()}
    order = sorted(rows.keys()-probes, key=lambda k:(digest([seeds["training_families"], k]),k))
    offset = training["offset"]
    if not order or type(offset) is not int or not 0 <= offset < len(order) or offset % 32:
        raise ReceiptGap("ORIGINAL_SAMPLE_OFFSET_DIFFERS")
    positives = defaultdict(list)
    for r in rows.values():
        positives[tuple(r["indices"]),tuple(r["values"])].append(r["tokens"])
    for i, sample in enumerate(s for batch in training["batches"] for s in batch):
        p, n = sample["positive"], sample["negative"]
        if (rows[order[(offset+i) % len(order)]] != p or p["membership"] != "training"
                or p["native_verified"] is not True or (sample["label"] is None) != (n is None)):
            raise ReceiptGap("SAMPLED_TRAINING_ROWS_DIFFER")
        if n is not None and (n["draw_identity"] in probes or rows.get(n["draw_identity"]) != n
                or n["membership"] != "training" or n["native_verified"] is not True
                or n["indices"] != p["indices"] or n["values"][:20] != p["values"][:20] or n["values"] == p["values"]
                or first_divergence(p["tokens"], n["tokens"], positives[tuple(p["indices"]),tuple(p["values"])]) != sample["label"]):
            raise ReceiptGap("SAMPLED_PREFERENCE_LABEL_DIFFERS")


def recover(root, folder, prior_source, revision, store, *, weight_slices=None):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from .progressive_round028 import ARCHIVE, reopen

    historical = verify_archive(confined(folder, ARCHIVE))
    if historical["report"] != store.state["publication"] or historical["decision"]["action"] != "MATCHED_GENERATOR":
        raise ReceiptGap("SAVED_AUDIT_DOES_NOT_SELECT_GENERATOR")
    # Reopen the four committed batches and their support metadata, without an audit or new donor execution.
    cut = reopen(root, folder, store.campaign, store.state["audit"], recheck_inputs=False)
    if historical["private_cut"] != cut["identity"]:
        raise ReceiptGap("PRIVATE_DIAGNOSTIC_PARENT_DIFFERS")
    area = confined(folder, "private/generator-rehearsal/" + revision)
    progress = sealed(confined(area, "progress.json")) if (area / "progress.json").is_file() else {}
    if progress and (progress.get("candidate") != revision or progress.get("audit") != historical["report"]):
        raise ReceiptGap("REHEARSAL_OR_AUDIT_IDENTITY_DIFFERS")
    training = sealed(confined(area, "training.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
    if (training["version"] != "round028-training-probe/v1" or training["base"] != cut["support"]["input"]
            or training["registration"] != store.campaign["identity"] or len(training["batches"]) != 2
            or any(len(b) != 32 for b in training["batches"])):
        raise ReceiptGap("TRAINING_PROBE_OR_REGISTRATION_DIFFERS")
    from .progressive_round028_audit import Donor, ORIGINAL
    from .progressive_round028_support import training_package
    donor = Donor(root, folder, store.campaign)
    original = donor.saved(ORIGINAL + "/prepare-research/frozen/campaign.json", donor.policy["campaign"],
                           maximum=CAMPAIGN_DOCUMENT_BYTES)
    base = training_package(donor, {r["name"]:r for r in original["payload"]["references"]})
    if base is None or base["identity"] != training["base"]:
        raise ReceiptGap("ORIGINAL_TRAINING_PRODUCER_UNAVAILABLE")
    verify_sample_origin(base, training, store.campaign["identity"])
    source = digest(collect_snapshot(prior_source))
    stages, closures, filesystem = stage_chain(area, revision=revision, source=source, training=training, progress=progress, weight_slices=weight_slices)
    forecast = training_forecast(stages["hip"]["arms"])
    if "forecast" in progress and forecast != progress["forecast"]:
        raise ReceiptGap("SAVED_FORECAST_DIFFERS")
    costs = cut["original_trials"]["trial-05-1M"]["costs"]
    evidence = seal(dict(version=VERSION, scope="training_only_rehearsal", candidate=revision, source=source,
                         audit=historical["report"], support=training["support"], **stages, forecast=forecast,
                         original_scale=original_scale(cut), filesystem=filesystem,
                         historical_trial05_us={k:costs[k] for k in PHASES}, checkpoint_closures=closures))
    return validate_public(evidence, historical)


def recover_command(root, folder, source, revision):
    from .progressive_round028 import ARCHIVE, audit_store
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("original full rehearsal implementation SHA required")
    if any(not (folder / name).is_file() for name in (ARCHIVE, "run.lock", "campaign.json")):
        return dict(mechanical_status="BLOCKED", reason="ORIGINAL_AUDIT_UNAVAILABLE", gpu_dispatches=0), 2
    with confined(folder, "run.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(mechanical_status="OWNED", reason="ACTIVE_WORKSTATION_OWNER", gpu_dispatches=0), 2
        area = confined(folder, "private/generator-rehearsal/" + revision, exists=False)
        context = dict(candidate=revision, scope="saved_rehearsal", gpu_dispatches=0,
                       stage_requests=len(list(area.glob("stages/*/request.json"))),
                       published_stages=len(list(area.glob("stages/*/receipt.json"))))
        try:
            helper = source / "specs/015-learned-program-search/validation/candidate_source.py"
            prior = runpy.run_path(str(helper))["materialize"](root, revision)
            store = audit_store(folder, revision)
            evidence = recover(root, folder, prior, revision, store)
        except (ValueError, KeyError, OSError, TypeError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
            name = exc.reason if isinstance(exc, ReceiptGap) else type(exc).__name__
            extra = exc.observed if isinstance(exc, ReceiptGap) else {}
            from .progressive_round028_rehearsal import failure_locations
            return dict(context, mechanical_status="BLOCKED", reason=name, **extra,
                        failure_locations=failure_locations(source, area)), 2
        return dict(context, mechanical_status="COMPLETE", scientific_status="PILOT_UNMEASURED",
                    reason="SAVED_REHEARSAL_VERIFIED", evidence=evidence), 0
