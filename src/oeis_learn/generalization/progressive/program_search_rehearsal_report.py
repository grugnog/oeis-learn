"""Portable aggregate of the original CPU and HIP rehearsal, with a replayable forecast."""
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

from .contracts import seal
from .program_search_report import verify_archive as verify_audit
from .report import MAX_BYTES
from oeis_learn.pipeline_validation.progressive_round028_rehearsal import (
    ARM_MEASURES, CPU_MEASURES, TELEMETRY, training_forecast,
)

VERSION = "round028-rehearsal-evidence/v1"
ARMS = ("ordinary", "first_divergence")
COMMON = {"image", "build_us", "image_reused", "reused", "elapsed_us", "all_attempts_us",
          "unknown_intervals", "attempts", "lease_us", "producer_us"}
SCALE = {"campaign_bytes", "candidate_score_bytes", "largest_input_bytes", "input_bytes", "input_files"}
SUPPORT = {"eligible_rows", "eligible_families", "positions"}
PHASES = {"startup", "training", "inference", "search", "verification"}


def require(condition):
    if not condition:
        raise ValueError("inconsistent rehearsal evidence")


def fields(value, keys):
    require(isinstance(value, dict) and set(value) == set(keys))


def count(value, minimum=0):
    require(type(value) is int and value >= minimum)


def validate_public(value, historical):
    fields(value, {"version", "identity", "candidate", "source", "audit", "scope", "support", "cpu", "hip",
                   "forecast", "original_scale", "filesystem", "historical_trial05_us", "checkpoint_closures"})
    require(value["version"] == VERSION and seal(value) == value and value["scope"] == "training_only_rehearsal")
    require(value["audit"] == historical["report"] and historical["denominator"] == 128
            and historical["decision"]["action"] == "MATCHED_GENERATOR")
    require(re.fullmatch("[0-9a-f]{40}", value["candidate"]) and re.fullmatch("sha256:[0-9a-f]{64}", value["source"]))
    for name, keys in (("support", SUPPORT), ("filesystem", {"free_bytes", "free_inodes"}),
                       ("original_scale", SCALE), ("historical_trial05_us", PHASES)):
        fields(value[name], keys)
        for v in value[name].values():
            count(v)
    support = value["support"]
    require(0 < support["eligible_families"] <= support["eligible_rows"] and 0 < support["positions"] <= support["eligible_rows"])
    fields(value["checkpoint_closures"], ARMS)
    require(all(v in ("VERIFIED", "UNAVAILABLE") for v in value["checkpoint_closures"].values()))
    cpu, hip = value["cpu"], value["hip"]
    fields(cpu, COMMON | set(CPU_MEASURES))
    fields(hip, COMMON | {"arms", "matched", "peak_device_bytes", "telemetry"})
    for stage in (cpu, hip):
        require(re.fullmatch("sha256:[0-9a-f]{64}", stage["image"]) and type(stage["image_reused"]) is bool
                and type(stage["reused"]) is bool)
        for k in COMMON - {"image", "image_reused", "reused"}:
            count(stage[k])
        require(stage["attempts"] > stage["unknown_intervals"] and stage["producer_us"] > 0
                and 0 < stage["elapsed_us"] <= stage["all_attempts_us"])
    require(cpu["lease_us"] == 0 and 0 < hip["lease_us"] <= hip["elapsed_us"] and hip["matched"] is True)
    for k in ("gradient_loss", "gradient_l2"):
        require(type(cpu[k]) in (int,float) and math.isfinite(cpu[k]) and cpu[k] >= 0)
    require(cpu["gradient_l2"] > 0 and cpu["gradient_probe_rows"] == 1 and cpu["native_verified_terms"] == 200)
    for k in ("gradient_probe_rows", "native_verified_terms", "parameter_bytes"):
        count(cpu[k], 1)
    count(hip["peak_device_bytes"], 1)
    fields(hip["arms"], ARMS)
    for arm, measured in hip["arms"].items():
        fields(measured, ARM_MEASURES)
        require(measured["actual_update"] is True and measured["full_restore"] is True)
        for k in ("charged_tokens", "update_us"):
            require(isinstance(measured[k], list) and len(measured[k]) == 2)
            for n in measured[k]:
                count(n, 1)
        for k in ("save_us", "restore_us", "replay_us", "checkpoint_bytes", "checkpoint_files"):
            count(measured[k], 1)
        count(measured["preference_rows"])
        require(measured["preference_rows"] == 0 if arm == "ordinary" else 0 < measured["preference_rows"] <= 32)
    require(hip["arms"]["ordinary"]["charged_tokens"] == hip["arms"]["first_divergence"]["charged_tokens"])
    telemetry = hip["telemetry"]
    fields(telemetry, TELEMETRY)
    for k in ("samples", "readable_samples", "covered_us", "role_wall_us"):
        count(telemetry[k])
    require(telemetry["readable_samples"] <= telemetry["samples"] and telemetry["covered_us"] <= telemetry["role_wall_us"])
    if telemetry["covered_us"]:
        count(telemetry["busy_us"])
        require(telemetry["busy_us"] <= telemetry["covered_us"])
    else:
        require(telemetry["busy_us"] is None)
    require(value["forecast"] == training_forecast(hip["arms"]))
    return value


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def audit_context(path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() == ["replay.json", "round028-audit.zip"]:
            from .program_search_report import verify_replay
            receipt = verify_replay(path)
            return receipt, receipt["historical_1m_cost_us"].get("trial-05-1M")
    receipt = verify_audit(path)
    with zipfile.ZipFile(path) as archive:
        audited = json.loads(archive.read("audit.json"))
    return receipt, audited["historical_1m_cost_us"].get("trial-05-1M")


def verify_transport(path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() == ["replay.json", "round028-audit.zip"]:
            from .program_search_report import verify_replay
            return verify_replay(path)
        if archive.namelist() == ["qualification.json", "round028-parent.zip"]:
            from .program_search_pilot import verify_qualification
            return verify_qualification(path)
        if "pilot.json" in archive.namelist():
            from .program_search_pilot import verify_pilot
            return verify_pilot(path)
        if "workstation.json" in archive.namelist():
            from .program_search_outcome import verify_workstation
            return verify_workstation(path)
        if "outcome.json" in archive.namelist():
            from .program_search_outcome import verify_outcome
            return verify_outcome(path)
        if archive.namelist() == ["timing.json", "round028-panels.zip"]:
            from .program_search_timing import verify_timing
            return verify_timing(path)
        if archive.namelist() == ["panels.json", "round028-rehearsal.zip"]:
            from .program_search_panel_report import verify_panels
            return verify_panels(path)
        if archive.namelist() == ["audit.json"]:
            return verify_audit(path)
        if archive.namelist() != ["rehearsal.json", "round028.zip"]:
            raise ValueError("missing verified rehearsal evidence or unknown public archive")
        require(not archive.comment and all(not i.extra and not i.comment and i.file_size <= MAX_BYTES
                                            for i in archive.infolist()))
        historical, costs = audit_context(io.BytesIO(archive.read("round028.zip")))
        raw = archive.read("rehearsal.json")
        value = validate_public(json.loads(raw), historical)
        require(encode(value) == raw and value["historical_trial05_us"] == costs)
    return dict(status="VERIFIED", scope="archive_only_rehearsal", candidate=value["candidate"], audit=value["audit"],
                evidence=value["identity"], support=value["support"], gradient_l2=value["cpu"]["gradient_l2"],
                forecast=value["forecast"], checkpoint_closures=value["checkpoint_closures"],
                gpu_dispatches=0)


def publish(path, value, audit_path):
    path = Path(path)
    historical, costs = audit_context(audit_path)
    validate_public(value, historical)
    require(value["historical_trial05_us"] == costs)
    with tempfile.NamedTemporaryFile(dir=path.parent) as staged:
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("rehearsal.json", encode(value))
            archive.write(audit_path, "round028.zip")
        staged.flush()
        os.fsync(staged.fileno())
        receipt = verify_transport(staged.name)
        os.link(staged.name, path)
    return receipt
