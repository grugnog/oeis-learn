"""Independently check the aggregate workstation evidence and its two parents."""

import errno
import io
import json
import re
import zipfile

from .contracts import digest, seal
from .program_search_outcome import read_canonical
from .program_search_rehearsal_report import encode

MEMBERS = ("release.json", "rehearsal.zip", "pilot.zip")
PHASES = ("source", "wheel", "recovery", "storage", "scale", "hardware")


def guard_inventory(scale, hip):
    from dataclasses import asdict
    from oeis_learn.synthesis.contracts import MAX_FRAME_BYTES, ResourceProfile
    from oeis_learn.synthesis.service_policy import ServiceLimits
    from oeis_learn.tracking.pipeline_profile import Recorder
    from .contracts import CAMPAIGN_DOCUMENT_BYTES
    from .report import MAX_BYTES, MAX_MANIFEST, MAX_PART, MAX_ROWS
    from .storage import JOURNAL_BYTES
    from oeis_learn.pipeline_validation import research_limits as limits
    from oeis_learn.pipeline_validation.progressive_workstation_state import LIMIT, LOG_LIMIT

    def guard(owner, scope, units, limit, observed, harm, exhaustion, source):
        return dict(
            owner=owner,
            scope=scope,
            units=units,
            limit=limit,
            observed=observed,
            margin=limit - observed if observed is not None else None,
            basis=(
                "measured_workstation_release"
                if observed is not None
                else "registered_owner_profile"
            ),
            source=source,
            protects=harm,
            exhaustion=exhaustion,
        )

    largest_batch = max(
        (v["expanded_bytes"] + v["files"] - 1) // v["files"] for v in scale["corpora"].values()
    )
    rows = {
        "campaign": guard(
            "contracts",
            "one campaign",
            "bytes",
            CAMPAIGN_DOCUMENT_BYTES,
            scale["campaign_bytes"],
            "reader memory",
            "reject oversized campaign",
            "contracts.CAMPAIGN_DOCUMENT_BYTES",
        ),
        "producer_reader": guard(
            "workstation collector",
            "one producer",
            "bytes",
            LIMIT,
            largest_batch,
            "reader memory",
            "bound the individual read",
            "progressive_workstation_state.LIMIT",
        ),
        "journal": guard(
            "budget journal",
            "campaign",
            "bytes",
            JOURNAL_BYTES,
            scale["journal"]["bytes"],
            "durable accounting",
            "preserve completed charges",
            "storage.JOURNAL_BYTES",
        ),
        "zip": guard(
            "archive verifier",
            "expanded aggregate",
            "bytes",
            MAX_BYTES,
            None,
            "reader memory",
            "reject oversized archive",
            "report.MAX_BYTES",
        ),
        "projection_part": guard(
            "report transport",
            "one part",
            "bytes",
            MAX_PART,
            None,
            "transport memory",
            "split verified parts",
            "report.MAX_PART",
        ),
        "manifest": guard(
            "report transport",
            "one manifest",
            "bytes",
            MAX_MANIFEST,
            None,
            "reader memory",
            "reject oversized manifest",
            "report.MAX_MANIFEST",
        ),
        "table": guard(
            "report transport",
            "one public table",
            "rows",
            MAX_ROWS,
            None,
            "reader memory",
            "reject oversized table",
            "report.MAX_ROWS",
        ),
        "fit": guard(
            "native fitter",
            "one fit",
            "microseconds",
            4_000_000,
            None,
            "equal service",
            "mark unfinished fit",
            "FR-006",
        ),
        "portfolio": guard(
            "portfolio",
            "one prompt",
            "microseconds",
            60_000_000,
            None,
            "equal search",
            "preserve partial portfolio",
            "FR-006",
        ),
        "learner": guard(
            "acceptance supervisor",
            "one disposable learner role",
            "seconds",
            limits.ACCEPTANCE_SECONDS,
            hip["elapsed_us"] / 1e6,
            "workstation lease",
            "stop and preserve checkpoint",
            "research_limits.ACCEPTANCE_SECONDS",
        ),
        "cleanup": guard(
            "supervisor",
            "one termination",
            "seconds",
            limits.CLEANUP_SECONDS,
            None,
            "stuck worker",
            "verify termination",
            "research_limits.CLEANUP_SECONDS",
        ),
        "samples": guard(
            "telemetry",
            "one role",
            "samples",
            Recorder.MAX_SAMPLES,
            hip["telemetry"]["samples"],
            "log storage",
            "mark coverage gap",
            "Recorder.MAX_SAMPLES",
        ),
        "telemetry_events": guard(
            "telemetry",
            "one role",
            "events",
            Recorder.MAX_EVENTS,
            None,
            "log storage",
            "retain aggregate coverage",
            "Recorder.MAX_EVENTS",
        ),
        "diagnostic_log": guard(
            "workstation collector",
            "one diagnostic log",
            "bytes",
            LOG_LIMIT,
            None,
            "report storage",
            "truncate diagnostic copy",
            "progressive_workstation_state.LOG_LIMIT",
        ),
        "frame": guard(
            "native protocol",
            "one request",
            "bytes",
            MAX_FRAME_BYTES,
            None,
            "decoder memory",
            "reject oversized frame",
            "synthesis.contracts.MAX_FRAME_BYTES",
        ),
    }
    for key, value in asdict(ResourceProfile()).items():
        if type(value) in (float, int):
            rows["native_" + key] = guard(
                "native runtime",
                "one registered execution",
                key,
                value,
                None,
                "bounded exact execution",
                "record native disposition",
                "ResourceProfile." + key,
            )
    for key, value in asdict(ServiceLimits()).items():
        units = "bytes" if key.endswith("bytes") else "entries" if "cache" in key else "batches"
        rows["service_" + key] = guard(
            "native service",
            "one owning role",
            units,
            value,
            None,
            "bounded queue and cache",
            "expire queued work or evict cache entries",
            "ServiceLimits." + key,
        )
    return rows


def verify_release(path):
    from oeis_learn.pipeline_validation.progressive_release_checks import HISTORICAL, release_cases
    from .program_search_rehearsal_report import verify_transport
    from .program_search_pilot import verify_pilot

    rows = read_canonical(path, MEMBERS)
    doc = json.loads(rows["release.json"])

    def need(ok):
        if not ok:
            raise ValueError("inconsistent workstation release evidence")

    def shape(value, fields):
        need(isinstance(value, dict) and set(value) == set(fields.split()))

    def numeric(value, minimum=0):
        need(type(value) is int and value >= minimum)

    shape(
        doc,
        "version identity candidate source registration pilot rehearsal stages hardware targets guards",
    )
    need(
        doc == seal(doc)
        and encode(doc) == rows["release.json"]
        and doc["version"] == "round028-release/v1"
        and re.fullmatch(r"[0-9a-f]{40}", doc["candidate"])
        and re.fullmatch(r"sha256:[0-9a-f]{64}", doc["source"])
    )
    rehearsal = verify_transport(io.BytesIO(rows["rehearsal.zip"]))
    with zipfile.ZipFile(io.BytesIO(rows["rehearsal.zip"])) as nested:
        qualification = json.loads(nested.read("rehearsal.json"))
    pilot = verify_pilot(io.BytesIO(rows["pilot.zip"]))
    need(
        pilot["report"] == doc["pilot"]
        and pilot["assessment"] == "SEALED"
        and rehearsal["evidence"] == doc["rehearsal"]
        and rehearsal["candidate"] == doc["candidate"]
        and qualification["source"] == doc["source"]
        and set(rehearsal["checkpoint_closures"].values()) == {"VERIFIED"}
    )
    shape(
        doc["targets"],
        "candidate_score_bytes campaign_bytes calibration_bytes tasks journal_events",
    )
    for k, floor in HISTORICAL.items():
        numeric(doc["targets"][k], max(floor, qualification["original_scale"].get(k, 0)))
    need(
        doc["registration"]
        == seal(
            dict(
                version="round028-release-registration/v1",
                candidate=doc["candidate"],
                source=doc["source"],
                pilot=doc["pilot"],
                targets=doc["targets"],
                suite=digest(release_cases()),
            )
        )["identity"]
    )
    stages = doc["stages"]
    shape(stages, "source wheel recovery storage scale interrupt")
    cpu_image = qualification["cpu"]["image"]
    for stage, evidence in stages.items():
        shape(
            evidence,
            "image native producer wall_us result" + (" lease_us" if stage == "interrupt" else ""),
        )
        expected_image = qualification["hip"]["image"] if stage == "interrupt" else cpu_image
        need(
            evidence["image"] == expected_image
            and all(
                re.fullmatch(r"sha256:[0-9a-f]{64}", evidence[k]) for k in ("producer", "native")
            )
        )
        numeric(evidence["wall_us"])
    need(len({s["native"] for k, s in stages.items() if k != "interrupt"}) == 1)
    fault = stages["interrupt"]
    shape(
        fault["result"],
        "final useful_updates discarded_updates useful_tokens failed_us restored_us",
    )
    need(
        re.fullmatch(r"sha256:[0-9a-f]{64}", fault["result"]["final"])
        and 0 < fault["lease_us"] <= fault["wall_us"]
        and fault["result"]["useful_updates"] == 2
        and fault["result"]["discarded_updates"] == 1
        and fault["result"]["useful_tokens"]
        == sum(qualification["hip"]["arms"]["ordinary"]["charged_tokens"])
    )
    for key in ("failed_us", "restored_us"):
        numeric(fault["result"][key], 1)
    for mode in ("source", "wheel"):
        value = stages[mode]["result"]
        shape(value, "mode files roster tests elapsed_us junit")
        need(
            value["mode"] == mode
            and value["files"] == len(release_cases())
            and value["roster"] == digest(release_cases())
        )
        numeric(value["tests"], value["files"])
        numeric(value["elapsed_us"], 1)
        need(re.fullmatch(r"sha256:[0-9a-f]{64}", value["junit"]))
    need(stages["source"]["result"]["tests"] == stages["wheel"]["result"]["tests"])
    recovery = stages["recovery"]["result"]
    shape(recovery, "tests report")
    numeric(recovery["tests"], 1)
    need(re.fullmatch(r"sha256:[0-9a-f]{64}", recovery["report"]))
    storage = stages["storage"]["result"]
    shape(storage, "errno bytes_exercised restored")
    need(storage["errno"] == errno.ENOSPC and storage["restored"] is True)
    numeric(storage["bytes_exercised"], 1)
    scale = stages["scale"]["result"]
    shape(
        scale,
        "targets corpora journal campaign_bytes calibration_bytes peak_allocated_bytes peak_inodes",
    )
    need(scale["targets"] == doc["targets"])
    for key in ("campaign_bytes", "calibration_bytes"):
        numeric(scale[key], scale["targets"][key])
    shape(scale["corpora"], "compressible varied")
    for item in scale["corpora"].values():
        shape(item, "expanded_bytes compressed_bytes files cold_us warm_us")
        need(
            item["files"] == scale["targets"]["tasks"]
            and item["expanded_bytes"] == scale["targets"]["candidate_score_bytes"]
        )
        for k in ("compressed_bytes", "cold_us", "warm_us"):
            numeric(item[k], 1)
    need(
        scale["corpora"]["varied"]["compressed_bytes"]
        > scale["corpora"]["compressible"]["compressed_bytes"]
    )
    journal = scale["journal"]
    shape(journal, "events committed bytes cold_us warm_us")
    need(journal["events"] == scale["targets"]["journal_events"] == 2 * journal["committed"])
    for key in ("bytes", "cold_us", "warm_us"):
        numeric(journal[key], 1)
    numeric(scale["peak_allocated_bytes"], 1)
    numeric(scale["peak_inodes"], scale["targets"]["tasks"])
    need(
        doc["hardware"] == qualification["hip"]["image"]
        and doc["guards"] == guard_inventory(scale, qualification["hip"])
    )
    return dict(
        status="VERIFIED",
        scope="archive_only_release_acceptance",
        candidate=doc["candidate"],
        report=doc["identity"],
        tests_per_install=stages["wheel"]["result"]["tests"],
        pilot=pilot["report"],
        scientific_action=pilot["next_action"],
        assessment=pilot["assessment"],
    )
