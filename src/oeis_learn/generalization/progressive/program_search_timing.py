"""Public pricing for a fixed eight-case candidate timing sample on exposed families."""

import io
import json
import re
import zipfile

from .cohorts import STRATA
from .contracts import seal
from .program_search_costing import PROXIES
from .program_search_rehearsal_report import count, encode, fields, require

PANEL_REVISION = "8954cbd0644273bed4a8cf413df7573519c86d10"
PANEL = "sha256:a6556a276840a845219c2df53e22fb18c4cb9d95ca0c956c30fa4d04986621bd"
LEARNERS = ("ordinary", "first_divergence")
BLOCKS = tuple(f"{arm}-{look}" for look in ("one", "full") for arm in LEARNERS) + (
    "incumbent-full",
)
MEMBERS = ["timing.json", "round028-panels.zip"]
VERSION = "round028-candidate-evaluation-timing/v1"
CELL = (
    "cases",
    "closed",
    "started",
    "selected",
    "scored",
    "executed",
    "search_us",
    "max_search_us",
    "score_us",
    "max_score_us",
    "portfolio_us",
    "decode_us",
    "native_us",
)


def parent_plan(panel, rehearsal):
    historical = panel["historical_forecast"]
    hip = rehearsal["hip"]
    checkpoint = sum(v["save_us"] + v["restore_us"] for v in hip["arms"].values())
    updates = sum(sum(v["update_us"]) + v["replay_us"] for v in hip["arms"].values())
    # Includes cold setup, model construction, waiting, inspection and cleanup.
    # Saved update and checkpoint intervals are charged independently.
    training_overhead = max(0, hip["elapsed_us"] - checkpoint - updates)
    return {
        "training_us": historical["training_us"],
        "shift_us": historical["shift_us"],
        "reserve_us": historical["reserve_us"],
        "probes": panel["descriptive_cases_per_arm_look"],
        "training_role_overhead_us": training_overhead,
        "checkpoint_cycle_us": checkpoint,
        "score_floor_us": {
            arm: {
                s: (cell["score_us"] + 31) // 32
                for s, cell in panel["historical_clocks"][proxy].items()
            }
            for arm, proxy in zip(("learner", "incumbent"), PROXIES, strict=True)
        },
    }


def estimate_role(role, data, strata, timing, rehearsal, plan):
    """Price the next batch from the original role overhead and the slower observed case."""
    from .program_search_pilot import LOOKS
    from fractions import Fraction
    from math import ceil

    arm, target, *_ = data["block"].split("-")
    target = int(target)
    if role == "train":
        a = rehearsal["hip"]["arms"][arm]
        earlier = LOOKS[max(0, LOOKS.index(target) - 1)]
        tokens = max(0, target - max(earlier, data.get("resume_charge") or 0))
        rate = max(Fraction(t, n) for t, n in zip(a["update_us"], a["charged_tokens"], strict=True))
        return (
            ceil(tokens * rate) + a["save_us"] + a["restore_us"] + plan["training_role_overhead_us"]
        )
    block = timing["blocks"][arm + ("-full" if target == LOOKS[-1] else "-one")]
    phase = "search" if role == "search" else "score"
    device = "gpu" if role == "search" else "cpu"
    cells = block["cells"]
    overhead = max(0, block[device + "_wall_us"] - sum(c[phase + "_us"] for c in cells.values()))
    floors = plan["score_floor_us"]["incumbent" if arm == "incumbent" else "learner"]
    rates = {
        s: max(c["max_" + phase + "_us"], floors[s] if phase == "score" else 0)
        for s, c in cells.items()
    }
    return overhead + sum(
        rates["parameter" if strata[r["id"]] == "retention" else strata[r["id"]]]
        for r in data["rows"]
    )


def pilot_prices(graph, strata, timing, rehearsal, plan):
    """One physical role per arm and look, including both zero-token checkpoints."""
    prices = {}
    for target, looks in graph.items():
        for arm, look in looks.items():
            block = f"{arm}-{target}"
            rows = [{"id": r["task"]} for r in look["members"]]
            for role in (("search", "score") if arm == "incumbent" else ("train", "search", "score")):
                prices[(block, role)] = estimate_role(role, dict(block=block, rows=rows),
                                                      strata, timing, rehearsal, plan)
    return prices


def remaining_price(prices, completed, current, estimate):
    return estimate + sum(price for key, price in prices.items() if key != current and key not in completed)


def prices(block, *, full, score_floors):
    fields(
        block,
        {
            "cells",
            "gpu_wall_us",
            "cpu_wall_us",
            "lease_us",
            "gpu_worker_us",
            "cpu_worker_us",
            "telemetry",
            "peak_device_bytes",
        },
    )
    for name in (
        "gpu_wall_us",
        "cpu_wall_us",
        "lease_us",
        "gpu_worker_us",
        "cpu_worker_us",
        "peak_device_bytes",
    ):
        count(block[name], 1)
    require(
        block["gpu_worker_us"] <= block["lease_us"] <= block["gpu_wall_us"]
        and block["cpu_worker_us"] <= block["cpu_wall_us"]
    )
    telemetry = block["telemetry"]
    fields(telemetry, {"samples", "readable_samples", "covered_us", "busy_us"})
    for k in ("samples", "readable_samples", "covered_us"):
        count(telemetry[k])
    require(
        telemetry["readable_samples"] <= telemetry["samples"]
        and telemetry["covered_us"] <= block["gpu_worker_us"]
    )
    if telemetry["covered_us"]:
        count(telemetry["busy_us"])
        require(telemetry["busy_us"] <= telemetry["covered_us"])
    else:
        require(telemetry["busy_us"] is None)
    fields(block["cells"], STRATA)
    rates, complete = {}, True
    for s, c in block["cells"].items():
        fields(c, CELL)
        for v in c.values():
            count(v)
        require(
            c["cases"] == 2
            and c["closed"] <= 2
            and c["scored"] <= 2
            and c["executed"] <= c["selected"] <= 2
            and c["started"] <= (16 if full else 2)
            and c["max_search_us"] <= c["search_us"] <= 2 * c["max_search_us"]
            and c["max_score_us"] <= c["score_us"] <= 2 * c["max_score_us"]
            and c["portfolio_us"] <= c["search_us"] + 2
        )
        # Each stratum is priced at its slower observed case for each phase. A
        # quick empty selection must not remove the original independent scorer allowance.
        rates[s] = c["max_search_us"] + max(c["max_score_us"], score_floors[s])
        complete &= c["closed"] == c["scored"] == 2 and c["executed"] == c["selected"]
    totals = [sum(c[field] for c in block["cells"].values()) for field in ("search_us", "score_us")]
    require(totals[0] <= block["gpu_worker_us"] + 8 and totals[1] <= block["cpu_worker_us"] + 8)
    overhead = max(0, block["gpu_wall_us"] - totals[0]) + max(0, block["cpu_wall_us"] - totals[1])
    return rates, overhead, complete


def project(blocks, plan):
    fields(blocks, BLOCKS)
    rates, overheads, batch_counts, work = {}, {}, {}, {}
    measured = True
    for name in BLOCKS:
        arm, look = name.split("-")
        full = look == "full"
        rates[name], overheads[name], ok = prices(
            blocks[name],
            full=full,
            score_floors=plan["score_floor_us"]["incumbent" if arm == "incumbent" else "learner"],
        )
        measured &= ok
        counts = dict.fromkeys(STRATA, 32) if full else plan["probes"]
        # The separate 32-case retention reservation uses the familiar parameter domain.
        work[name] = sum(n * rates[name][s] for s, n in counts.items()) + (
            32 * rates[name]["parameter"] if full else 0
        )
        batch_counts[name] = (sum(counts.values()) + (32 if full else 0) + 7) // 8
    details = {
        "rates_us": rates,
        "role_overheads_us": overheads,
        "work_us": work,
        "training_us": plan["training_us"],
        "shift_us": plan["shift_us"],
        "reserve_us": plan["reserve_us"],
    }
    if not measured:
        return dict(details, action="RESOLVE_INCOMPLETE_TIMING")

    def scenario(batches):
        costs = {name: work[name] + overheads[name] * batches[name] for name in BLOCKS}
        early = 2 * sum(costs[arm + "-one"] for arm in LEARNERS)
        endpoint = sum(costs[name] for name in BLOCKS if name.endswith("-full"))
        repeat_eval = endpoint + sum(costs[arm + "-full"] for arm in LEARNERS)
        # One save/restore cycle per arm and training segment. The earlier HIP
        # receipt supplies the observed host startup/cleanup cost per training role.
        cycle = plan["checkpoint_cycle_us"] + 2 * plan["training_role_overhead_us"]
        pilot = plan["training_us"] + 2 * cycle + early + endpoint
        repeat = plan["training_us"] + cycle + repeat_eval
        return {
            "descriptive_us": early,
            "endpoint_us": endpoint,
            "pilot_us": pilot,
            "repeat_us": repeat,
            "pilot_remaining_us": plan["shift_us"] - pilot,
            "remaining_us": plan["shift_us"] - pilot - repeat,
        }

    logical = scenario(dict.fromkeys(BLOCKS, 1))
    chunks = scenario(batch_counts)
    if logical["pilot_remaining_us"] < 0:
        action = "REPORT_PILOT_COST_DEFICIT"
    elif logical["remaining_us"] < 0:
        action = "QUALIFY_PILOT_ONLY"
    else:
        action = "QUALIFY_PROSPECTIVE_PILOT"
    return dict(details, **logical, eight_case_batches=chunks, action=action)


def verify_timing(path):
    from .program_search_panel_report import verify_panels

    with zipfile.ZipFile(path) as archive:
        require(
            archive.namelist() == MEMBERS
            and not archive.comment
            and all(
                not i.extra and not i.comment and i.file_size < 1 << 20 for i in archive.infolist()
            )
        )
        raw = archive.read(MEMBERS[0])
        value = json.loads(raw)
        require(encode(value) == raw and seal(value) == value)
        parent = io.BytesIO(archive.read(MEMBERS[1]))
    receipt = verify_panels(parent)
    require(
        receipt["evidence"] == PANEL and receipt["action"] == "MEASURE_CANDIDATE_EVALUATION_COST"
    )
    with zipfile.ZipFile(parent) as panels:
        panel = json.loads(panels.read("panels.json"))
        with zipfile.ZipFile(io.BytesIO(panels.read("round028-rehearsal.zip"))) as prior:
            rehearsal = json.loads(prior.read("rehearsal.json"))
    fields(
        value,
        {
            "version",
            "identity",
            "registration",
            "candidate",
            "panel",
            "scope",
            "blocks",
            "forecast",
            "image_build_us",
            "space_before",
            "space_after",
        },
    )
    require(
        value["version"] == VERSION
        and value["panel"] == receipt["evidence"]
        and value["scope"] == "previously_exposed_timing_only"
        and re.fullmatch(r"[0-9a-f]{40}", value["candidate"])
        and re.fullmatch(r"sha256:[0-9a-f]{64}", value["registration"])
    )
    fields(value["image_build_us"], {"cpu", "hip"})
    for v in value["image_build_us"].values():
        count(v)
    for name in ("space_before", "space_after"):
        fields(value[name], {"free_bytes", "free_inodes"})
        for v in value[name].values():
            count(v)
    forecast = project(value["blocks"], parent_plan(panel, rehearsal))
    require(forecast == value["forecast"])
    return {
        "status": "VERIFIED",
        "scope": "archive_only_candidate_timing",
        "evidence": value["identity"],
        "candidate": value["candidate"],
        "registration": value["registration"],
        "panel": PANEL,
        "portfolios": 40,
        "gpu_dispatches": 5,
        "new_training_updates": 0,
        "forecast": forecast,
        "action": forecast["action"],
    }
