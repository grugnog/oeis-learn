"""Cost the registered generator comparison with exposed prompts and recovered weight slices."""

import io
import json
import runpy
import time
import zipfile
from pathlib import Path

from oeis_learn.generalization.progressive.budgets import (
    KEYS,
    BudgetJournal,
    role_work_requirements,
)
from oeis_learn.generalization.progressive.cohorts import STRATA, validate_prompt
from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    digest,
    load,
    read_bound_record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.program_search_panel_report import (
    publish_child,
    verify_panels,
)
from oeis_learn.generalization.progressive.program_search_timing import (
    BLOCKS,
    LEARNERS,
    MEMBERS,
    PANEL,
    PANEL_REVISION,
    VERSION,
    parent_plan,
    project,
    verify_timing,
)

from .identity import confined
from .pilot_clocks import CORRECT_FULL, DESCRIPTIVE
from .preparation_recovery import retain, revise
from .progressive_panel_sources import require
from .progressive_research_inputs import _json, checked_file
from .progressive_round028_panels import PARENT, PRODUCER, NativeRoles, bind_private
from .progressive_round028_rehearsal import space
from .progressive_timing_worker import ROLE_SECONDS, observations

PANEL_ARCHIVE = "share/panel-supply/" + PANEL_REVISION + "/round028-panels.zip"
AREA = "private/evaluation-timing"
SHARE = "share/evaluation-timing"


def timing_cases(cut):
    rows = cut["cases"]
    require(
        len(rows) == len({r["task"] for r in rows}) == 128
        and all(sum(r["stratum"] == s for r in rows) == 32 for s in STRATA),
        "original 128-case development roster required",
    )
    used, selected = set(), []
    for stratum in STRATA:
        # Hash only frozen family and case keys. Never inspect native outcomes or durations.
        ordered = sorted(
            (r for r in rows if r["stratum"] == stratum),
            key=lambda r: (digest([PANEL, stratum, r["group"]]), digest([PANEL, r["task"]])),
        )
        count = 0
        for row in ordered:
            if row["group"] not in used:
                used.add(row["group"])
                selected.append(row)
                count += 1
                if count == 2:
                    break
        require(count == 2, "two distinct exposed timing families per stratum required")
    return selected


def original_channels(root, rows):
    prompts, truths = [], []
    for r in rows:
        for name, role, kind, out in (
            ("visible", "generator", "visible_prompt", prompts),
            ("truth", "scorer", "development_truth", truths),
        ):
            item = read_bound_record(root, r["channels"][name], version="discovery-channel/v1")
            require(
                (item["role"], item["kind"], item["data"]["id"]) == (role, kind, r["task"]),
                "historical channel authority differs",
            )
            out.append(item["data"])
        p, t = validate_prompt(prompts[-1]), truths[-1]
        require(
            set(t) == set(p)
            and all(len(t[k]) == 100 and t[k][:20] == p[k] for k in ("indices", "values")),
            "historical visible/truth prefix differs",
        )
    return prompts, truths


def original_weights(root, source, folder, store):
    from .progressive_round028_audit import ORIGINAL, Donor
    from .progressive_round028_receipts import recover, sealed

    prior = runpy.run_path(
        str(source / "specs/015-learned-program-search/validation/candidate_source.py")
    )["materialize"](root, PRODUCER)
    weights = {}
    recovery = recover(root, folder, prior, PRODUCER, store, weight_slices=weights)
    require(
        recovery["identity"] == PARENT and set(weights) == set(LEARNERS),
        "both exact full checkpoint closures required",
    )
    training = sealed(
        confined(folder, "private/generator-rehearsal/" + PRODUCER + "/training.json"),
        maximum=CAMPAIGN_DOCUMENT_BYTES,
    )
    donor = Donor(root, folder, store.campaign)
    frozen = donor.saved(
        ORIGINAL + "/prepare-research/frozen/campaign.json",
        donor.policy["campaign"],
        maximum=CAMPAIGN_DOCUMENT_BYTES,
    )
    refs = {r["name"]: r for r in frozen["payload"]["references"]}
    recipe, incumbent = refs["recipe"], refs["retained-D-evaluation-weights"]
    require(
        (recipe["role"], recipe["kind"], incumbent["role"], incumbent["kind"])
        == ("custodian", "recipe", "generator", "checkpoint"),
        "bound incumbent and base recipe required",
    )
    weights["incumbent"] = checked_file(root, incumbent["artifact"]), incumbent["artifact"]
    profiles = dict.fromkeys(LEARNERS, training["profile"])
    profiles["incumbent"] = _json(checked_file(root, recipe["artifact"]))
    return profiles, weights


class TimingRole(NativeRoles):
    worker = "oeis_learn.pipeline_validation.progressive_timing_worker"
    request_version = "round028-timing-role/v1"
    producer_version = "round028-timing-output/v1"
    fixed_attempt = True


def cost(name, role):
    generate = role == "search"
    kind = "visible_prompt" if generate else "development_truth"
    values = [{"kind": kind} for _ in range(8)]
    if generate:
        values.append({"kind": "oeis_generation"})
    result = role_work_requirements("generate" if generate else "score", values)
    if generate and name.endswith("-one"):
        result.update(draws=8, attempts=8, rejections=8)
    # Includes the supervised cleanup allowance. Model tokens for training remain zero.
    result["elapsed_us"] = (ROLE_SECONDS + 120) * 1_000_000
    return result


def timed_role(runner, journal, name, role, data, mounts):
    debit = journal.reserve(
        name,
        role,
        0,
        cost(name, role),
        boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        monotonic_us=time.monotonic_ns() // 1000,
    )
    result = runner(role, seal(data), seconds=ROLE_SECONDS, inputs=mounts)
    journal.commit(debit["key"])
    return result


def run_blocks(area, roles, journal, profiles, weights, rows, prompts, truths):
    blocks, refs = {}, []
    gpu, cpu = roles
    members = {r["task"]: r["stratum"] for r in rows}
    for name in BLOCKS:
        arm, look = name.split("-")
        weight_path, original = weights[arm]
        weight = dict(original, path="inputs/weights.pt")
        generated = timed_role(
            gpu,
            journal,
            name,
            "search",
            {"block": name, "rows": prompts, "recipe": profiles[arm], "weights": weight},
            [(weight_path, "/work/inputs/weights.pt")],
        )
        gen_ref = generated["candidates"]
        require(
            gen_ref["path"] == "generation/candidates.json", "unexpected generator output location"
        )
        gen_file = checked_file(gpu.last_output, gen_ref)
        gen = read_bound_record(gpu.last_output, gen_ref, version="discovery-candidates/v1")
        require(
            gen["checkpoint"] == weight["sha256"]
            and gen["recipe"] == digest(profiles[arm])
            and gen["prompts"] == digest(sorted(prompts, key=lambda row: row["id"]))
            and gen["model_control"]
            == ("retained_D" if arm == "incumbent" else "earlier_checkpoint"),
            "generated candidate differs from frozen weight, recipe or prompt roster",
        )
        scored = timed_role(
            cpu,
            journal,
            name,
            "score",
            {
                "block": name,
                "rows": truths,
                "candidates": dict(gen_ref, path="inputs/candidates.json"),
            },
            [(gen_file, "/work/inputs/candidates.json")],
        )
        score_ref = scored["scores"]
        require(score_ref["path"] == "scored/scores.json", "unexpected scorer output location")
        scores = read_bound_record(cpu.last_output, score_ref, version="discovery-scoring/v1")
        cells = observations(
            gen,
            scores,
            members,
            generated["case_us"],
            gen_ref["sha256"],
            DESCRIPTIVE if look == "one" else CORRECT_FULL,
        )
        blocks[name] = {
            "cells": cells,
            "gpu_wall_us": gpu.last_wall_us,
            "cpu_wall_us": cpu.last_wall_us,
            "lease_us": gpu.last_lease_us,
            "gpu_worker_us": gpu.last_producer["elapsed_us"],
            "cpu_worker_us": cpu.last_producer["elapsed_us"],
            "telemetry": generated["telemetry"],
            "peak_device_bytes": generated["peak_device_bytes"],
        }
        for role in roles:
            boundary = load(role.last_boundary)
            parent = role.last_boundary.parent
            for path in (role.last_boundary, parent / "request.json", parent / "data.json"):
                refs.append(reference(area, path))
            for k in ("producer", "inspection", "timing", "lease"):
                if k in boundary:
                    refs.append(
                        dict(
                            boundary[k], path=str((parent / boundary[k]["path"]).relative_to(area))
                        )
                    )
        refs.extend(
            [
                dict(ref, path=str((out / ref["path"]).relative_to(area)))
                for out, ref in ((gpu.last_output, gen_ref), (cpu.last_output, score_ref))
            ]
        )
        revise(
            area / "progress.json",
            seal({"completed_blocks": len(blocks), "portfolios": 8 * len(blocks)}),
        )
    return blocks, refs


def timing_command(args, source, root, folder, revision, store, host_factory):
    from oeis_learn.experiments.source_snapshot import collect_snapshot

    from .progressive_round028 import reopen

    parent = confined(folder, PANEL_ARCHIVE)
    verified = verify_panels(parent)
    if verified["action"] != "MEASURE_CANDIDATE_EVALUATION_COST" or verified["deficits"]:
        return {
            "mechanical_status": "BLOCKED",
            "scientific_status": "PILOT_UNMEASURED",
            "reason": verified["action"],
            "gpu_dispatches": 0,
        }, 2
    require(
        verified["evidence"] == PANEL and store.state["route"] == "MATCHED_GENERATOR",
        "verified supply must warrant physical timing",
    )
    with zipfile.ZipFile(parent) as archive:
        panel = json.loads(archive.read("panels.json"))
        with zipfile.ZipFile(io.BytesIO(archive.read("round028-rehearsal.zip"))) as old:
            rehearsal = json.loads(old.read("rehearsal.json"))
    require(
        panel["audit"] == store.state["publication"],
        "panel and published audit belong to different campaigns",
    )
    supply = confined(folder, "private/panel-supply/" + PANEL_REVISION)
    confined(supply, "boundary.json")
    bind_private(supply, panel)
    area = confined(folder, AREA, exists=False)
    area.mkdir(exist_ok=True, mode=0o700)
    archive = confined(folder, SHARE + "/round028-timing.zip", exists=False)
    archive.parent.mkdir(parents=True, exist_ok=True)

    def completed():
        boundary = load(confined(area, "boundary.json"))
        require(
            seal(boundary) == boundary and boundary["panel"] == PANEL,
            "private timing boundary changed",
        )
        for ref in boundary["files"]:
            checked_file(area, ref)
        public = read_bound_record(area, boundary["public"], version=VERSION)
        reg = read_bound_record(area, boundary["registration"], identity=public["registration"])
        require(
            reg["candidate"] == public["candidate"] and reg["panel"] == PANEL,
            "timing registration differs",
        )
        receipt = (
            verify_timing(archive)
            if archive.exists()
            else publish_child(archive, public, parent, MEMBERS, verify_timing)
        )
        require(
            receipt["evidence"] == public["identity"], "private and copied timing result differ"
        )
        return {
            "mechanical_status": "COMPLETE",
            "scientific_status": "PILOT_UNMEASURED",
            "reason": receipt["action"],
            "archive": str(archive.relative_to(folder)),
            "receipt": receipt,
            "gpu_dispatches": 5,
        }, 0

    if (area / "boundary.json").exists():
        return completed()
    require(not archive.exists(), "timing publication without its private boundary")
    cut = reopen(root, folder, store.campaign, store.state["audit"], recheck_inputs=False)
    rows = timing_cases(cut)
    registration = seal(
        {
            "version": "round028-candidate-timing-registration/v1",
            "candidate": revision,
            "source": digest(collect_snapshot(source)),
            "panel": PANEL,
            "rehearsal": PARENT,
            "audit": store.state["publication"],
            "roster": [{k: r[k] for k in ("task", "stratum", "group")} for r in rows],
            "blocks": list(BLOCKS),
            "attempts": {"one": 1, "full": 8},
            "scope": "previously_exposed_timing_only",
            "role_seconds": ROLE_SECONDS,
        }
    )
    retain(area / "registration.json", registration)
    if not (area / "start-space.json").exists():
        retain(area / "start-space.json", seal({"space": space(area)}))
    prompts, truths = original_channels(root, rows)
    profiles, weights = original_weights(root, source, folder, store)
    caps = {
        k: sum(cost(name, role)[k] for name in BLOCKS for role in ("search", "score")) for k in KEYS
    }
    journal = BudgetJournal(area / "accounting", registration["identity"], caps)
    gpu, cpu = (
        TimingRole(args, source, root, revision, area, registration, host_factory) for _ in range(2)
    )
    gpu.phase = "hip"
    blocks, refs = run_blocks(area, (gpu, cpu), journal, profiles, weights, rows, prompts, truths)
    accounting = journal.snapshot()
    require(
        len(accounting["committed"]) == 10
        and not accounting["pending"]
        and accounting["consumed"]["draws"] == 208,
        "incomplete timing budget settlement",
    )
    if not (area / "end-space.json").exists():
        retain(area / "end-space.json", seal({"space": space(area)}))
    result = seal(
        {
            "version": VERSION,
            "scope": "previously_exposed_timing_only",
            "panel": PANEL,
            "candidate": revision,
            "registration": registration["identity"],
            "blocks": blocks,
            "forecast": project(blocks, parent_plan(panel, rehearsal)),
            "image_build_us": {"hip": gpu.build_us, "cpu": cpu.build_us},
            "space_before": load(area / "start-space.json")["space"],
            "space_after": load(area / "end-space.json")["space"],
        }
    )
    retain(area / "public.json", result)
    refs.extend(
        reference(area, area / p)
        for p in (
            "accounting/budget.jsonl",
            "accounting/budget-binding.json",
            "start-space.json",
            "end-space.json",
        )
    )
    retain(
        area / "boundary.json",
        seal(
            {
                "panel": PANEL,
                "registration": reference(area, area / "registration.json"),
                "public": reference(area, area / "public.json"),
                "files": refs,
            }
        ),
    )
    return completed()


def diagnostic_roles(area):
    """Fixed projections of the original role charges, including interrupted attempts."""
    result = {}
    for request in sorted(area.glob("roles/*/request.json")):
        try:
            req = load(confined(area, request))
            data = load(
                confined(area, request.with_name("data.json")), maximum=CAMPAIGN_DOCUMENT_BYTES
            )
            if (
                seal(req) != req
                or seal(data) != data
                or data["identity"] != req["data"]
                or data["block"] not in BLOCKS
                or req["role"] not in ("search", "score")
            ):
                continue
            entry = {
                "attempts": 0,
                "unknown_intervals": 0,
                "finished_us": 0,
                "lease_us": 0,
                "completed": (request.parent / "boundary.json").is_file(),
            }
            result[data["block"] + "-" + req["role"]] = entry
            for start in request.parent.glob("attempt-*/start.json"):
                entry["attempts"] += 1
                end = start.with_name("end.json")
                lease = start.parent / "output/gpu-lease.json"
                if end.is_file():
                    saved = load(confined(area, end))
                    if (
                        seal(saved) == saved
                        and saved["request"] == req["identity"]
                        and type(saved["elapsed_us"]) is int
                    ):
                        entry["finished_us"] += saved["elapsed_us"]
                    else:
                        entry["unknown_intervals"] += 1
                else:
                    entry["unknown_intervals"] += 1
                if lease.is_file():
                    us = load(confined(area, lease)).get("elapsed_us")
                    if type(us) is int and us >= 0:
                        entry["lease_us"] += us
        except (KeyError, ValueError, TypeError, OSError):
            continue
    return result
