"""Independently check the aggregate workstation evidence and its two parents."""

import errno
import io
import json
import re
import zipfile

from oeis_learn.pipeline_validation.progressive_research_resources import guard_inventory
from .contracts import digest, seal
from .program_search_outcome import read_canonical
from .program_search_rehearsal_report import encode

MEMBERS = ("release.json", "rehearsal.zip", "pilot.zip")
PHASES = ("source", "wheel", "recovery", "storage", "scale", "hardware")


def release_forecast(training, pilot):
    """Project both pairs from the new HIP updates and original measured end-to-end role times."""
    from .program_search_pilot import ARMS, LOOKS

    forecast = training["forecast"]
    observed = {(row["block"], row["role"]): row["charged_us"] for row in pilot["costs"]}
    estimates = {(block, role): cost for block, role, cost in pilot["design"]["prices"]}
    rows = []

    def evaluation(split, pair, arm, target):
        block = f"{arm}-{target}"
        for role in ("search", "score"):
            key = (block, role)
            if (
                split == "development"
                and arm == ARMS[1]
                and target == 0
                and pilot["looks"][0]["zero_reused"]
            ):
                cost, basis = 0, "verified shared zero-look"
            elif key in observed:
                cost, basis = observed[key], "original full pilot role elapsed"
            else:
                cost, basis = estimates[key], "original registered cold role forecast"
            rows.append(
                dict(
                    split=split, pair=pair, arm=arm, target=target, role=role, us=cost, basis=basis
                )
            )

    for arm in ARMS:
        previous = 0
        for target in LOOKS:
            key = (f"{arm}-{target}", "train")
            increment = (
                forecast["pilot_training_us"][arm] * (target - previous) + LOOKS[-1] - 1
            ) // LOOKS[-1]
            rows.append(
                dict(
                    split="development",
                    pair=0,
                    arm=arm,
                    target=target,
                    role="train",
                    us=max(increment, observed.get(key, estimates[key])),
                    basis="larger of candidate update projection and historical whole role",
                )
            )
            evaluation("development", 0, arm, target)
            previous = target
    evaluation("development", 0, "incumbent", LOOKS[-1])
    for pair in (0, 1):
        for arm in ARMS:
            if pair:
                reference = sum(
                    r["us"]
                    for r in rows
                    if r["split"] == "development" and r["arm"] == arm and r["role"] == "train"
                )
                rows.append(
                    dict(
                        split="replication",
                        pair=pair,
                        arm=arm,
                        target=LOOKS[-1],
                        role="train",
                        us=reference,
                        basis="independent seed at candidate and historical whole-role rate",
                    )
                )
            evaluation("replication", pair, arm, LOOKS[-1])
        evaluation("replication", pair, "incumbent", LOOKS[-1])
    shift = forecast["shift_us"]
    development = sum(r["us"] for r in rows if r["split"] == "development")
    repeat = sum(r["us"] for r in rows if r["split"] == "replication")
    return dict(
        scope="conditional planning only",
        roles=rows,
        shift_us=shift,
        development_us=development,
        replication_us=repeat,
        half_shift_us=shift // 2,
        deficit_us=max(0, development + repeat - shift),
        accrued_pilot_lease_us=pilot["research_gpu_us"],
    )


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
        "version identity candidate source registration pilot rehearsal stages hardware targets guards forecast admission",
    )
    need(
        doc == seal(doc)
        and encode(doc) == rows["release.json"]
        and doc["version"] == "round028-release/v1"
        and re.fullmatch(r"[0-9a-f]{40}", doc["candidate"])
        and re.fullmatch(r"sha256:[0-9a-f]{64}", doc["source"])
    )
    rehearsal = verify_transport(io.BytesIO(rows["rehearsal.zip"]))
    nested = read_canonical(io.BytesIO(rows["rehearsal.zip"]), ("rehearsal.json", "round028.zip"))
    qualification = json.loads(nested["rehearsal.json"])
    audit = read_canonical(io.BytesIO(nested["round028.zip"]), ("audit.json",))["audit.json"]
    pilot = verify_pilot(io.BytesIO(rows["pilot.zip"]))
    with zipfile.ZipFile(io.BytesIO(rows["pilot.zip"])) as child:
        need(
            doc["forecast"] == release_forecast(qualification, json.loads(child.read("pilot.json")))
        )
        with zipfile.ZipFile(io.BytesIO(child.read("round028-parent.zip"))) as parent:
            with zipfile.ZipFile(io.BytesIO(parent.read("round028.zip"))) as original:
                need(audit == original.read("audit.json"))
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
    admission = doc["admission"]
    shape(
        admission,
        "free_bytes free_inodes required_bytes required_inodes deficit_bytes deficit_inodes credited_scale floor_bytes identity",
    )
    for k in ("free_bytes", "free_inodes", "required_bytes", "required_inodes", "floor_bytes"):
        numeric(admission[k])
    need(
        admission == seal(admission)
        and admission["credited_scale"] is False
        and admission["deficit_bytes"] == admission["deficit_inodes"] == 0
        and admission["free_bytes"] >= admission["floor_bytes"] + admission["required_bytes"]
        and admission["required_bytes"] >= scale["peak_allocated_bytes"]
        and admission["free_inodes"] >= admission["required_inodes"] >= scale["peak_inodes"]
    )
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
