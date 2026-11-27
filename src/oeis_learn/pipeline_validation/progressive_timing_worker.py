"""Installed, isolated search or scorer role for a two-family-per-stratum cost measurement."""

import math
import os
import time
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    atomic_write,
    digest,
    load,
    seal,
)
from oeis_learn.generalization.progressive.program_search_costing import CLOCK_FIELDS
from oeis_learn.generalization.progressive.program_search_timing import BLOCKS

from .pilot_clocks import CORRECT_FULL, DESCRIPTIVE, clock_record
from .progressive_evaluation import generate, score

WORK_SECONDS = 540
ROLE_SECONDS = WORK_SECONDS + 120


def search(work, data, client):
    from oeis_learn.tracking.pipeline_profile import Recorder

    from .progressive_generator_probe import usage_summary
    from .progressive_qualification import require_hip

    if (
        set(data) != {"identity", "block", "rows", "recipe", "weights"}
        or data["weights"]["path"] != "inputs/weights.pt"
    ):
        raise ValueError("visible search scope required")
    arm, look = data["block"].split("-")
    policy = DESCRIPTIVE if look == "one" else CORRECT_FULL
    torch = require_hip()
    torch.cuda.reset_peak_memory_stats()
    telemetry_dir = work / "output/telemetry"
    nodes = sorted(Path("/dev/dri").glob("renderD*"))
    case_us = {}
    started = time.monotonic_ns()
    with (
        Recorder(
            telemetry_dir,
            instrumented=True,
            render_node=nodes[0] if len(nodes) == 1 else None,
            host_counters=False,
        ) as recorder,
        recorder.phase("search"),
    ):
        ref = generate(
            work,
            "generation",
            data["rows"],
            data["recipe"],
            client=client,
            model_control="retained_D" if arm == "incumbent" else "earlier_checkpoint",
            reference_recipe=None if arm == "incumbent" else data["recipe"],
            checkpoint=data["weights"],
            device="cuda",
            seconds=WORK_SECONDS,
            output_root=work / "output",
            methods=("correct",),
            evaluation_policy=policy,
            case_elapsed_us=case_us,
        )
        torch.cuda.synchronize()
    usage = usage_summary(telemetry_dir, (time.monotonic_ns() - started) // 1000)
    return {
        "candidates": ref,
        "case_us": case_us,
        "peak_device_bytes": torch.cuda.max_memory_allocated(),
        "telemetry": {
            k: usage[k] for k in ("samples", "readable_samples", "covered_us", "busy_us")
        },
    }


def scoring(work, data, client):
    if (
        set(data) != {"identity", "block", "rows", "candidates"}
        or data["candidates"]["path"] != "inputs/candidates.json"
    ):
        raise ValueError("committed candidate scorer scope required")
    score(
        work,
        data["candidates"],
        data["rows"],
        client=client,
        output="output/scored",
        seconds=WORK_SECONDS,
        selected_only=True,
    )
    from oeis_learn.generalization.progressive.contracts import reference

    return {"scores": reference(work / "output", work / "output/scored/scores.json")}


def observations(gen, scores, members, case_us, gen_hash, policy):
    from oeis_learn.generalization.progressive.cohorts import STRATA
    from oeis_learn.generalization.progressive.program_search_timing import CELL

    from .progressive_panel_sources import require

    clocks = gen["clock"]
    require(
        gen["methods"] == ["correct"]
        and seal(clocks) == clocks
        and clocks["policy"] == policy
        and clocks["work"] == gen["model"]
        and clocks["source"] == gen["recipe"]
        and clocks["expected_portfolios"] == len(members)
        and clocks["missing_portfolios"] == 0
        and scores["candidates"] == gen_hash
        and scores["denominator"] == len(members)
        and set(scores["outcomes"]) == set(gen["records"]) == set(case_us) == set(members),
        "original candidate and separate scorer timing parents required",
    )
    cells = {s: dict.fromkeys(CELL, 0) for s in STRATA}
    seen = set()
    for clock in clocks["clocks"]:
        key = clock["prompt"]
        require(
            key in members
            and key not in seen
            and clock["mode"] == "correct"
            and clock["policy"] == policy
            and clock["snapshot"] == gen["model"]
            and clock == clock_record(**{k: clock[k] for k in CLOCK_FIELDS}),
            "candidate clock differs",
        )
        seen.add(key)
        c = cells[members[key]]
        row = gen["records"][key]
        details = row["costs"]["correct"]["details"]
        chosen = row["selections"]["equal_budget"] is not None
        scoring = scores["outcomes"][key]["equal_budget"]["scoring"]
        require(
            type(case_us[key]) is int
            and case_us[key] >= 0
            and len(details) == clock["started"]
            and (chosen or scoring["outcome"] == "no_selected_candidate"),
            "timed case ledger differs",
        )
        c["cases"] += 1
        c["closed"] += int(clock["cost_complete"])
        c["started"] += clock["started"]
        c["selected"] += int(chosen)
        c["executed"] += int(chosen and scoring["complete"])
        c["scored"] += int(not chosen or scoring["complete"])
        c["search_us"] += case_us[key]
        c["max_search_us"] = max(c["max_search_us"], case_us[key])
        c["portfolio_us"] += math.ceil((clock["finish"] - clock["start"]) * 1e6)
        c["decode_us"] += sum(math.ceil(d["generation_seconds"] * 1e6) for d in details)
        c["native_us"] += sum(math.ceil(d.get("native_seconds", 0) * 1e6) for d in details)
        score_us = math.ceil(scoring["elapsed_seconds"] * 1e6)
        c["score_us"] += score_us
        c["max_score_us"] = max(c["max_score_us"], score_us)
    require(seen == set(members), "original clock missing")
    return cells


def main():
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.synthesis.client import SynthesisClient

    start = time.monotonic_ns()
    work = Path("/work")
    req = load(work / "inputs/request.json")
    data = load(work / "inputs/data.json", maximum=CAMPAIGN_DOCUMENT_BYTES)
    if (
        seal(req) != req
        or req["version"] != "round028-timing-role/v1"
        or req["source"] != digest(source_snapshot())
        or req["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
        or req["engine"] != os.environ["OEIS_QUALIFICATION_ENGINE"]
        or data["identity"] != req["data"]
        or seal(data) != data
        or data["block"] not in BLOCKS
        or len(data["rows"]) != 8
    ):
        raise ValueError("registered candidate timing role and eight original cases required")
    with SynthesisClient(digest(["round028-candidate-timing", req["identity"]])) as client:
        native = client.runtime_identity("bigint-body/v1")
        if req["candidate_native"] not in (None, native):
            raise ValueError("immutable candidate native implementation differs")
        result = {"search": search, "score": scoring}[req["role"]](work, data, client)
    atomic_write(
        work / "output/producer.json",
        seal(
            {
                "version": "round028-timing-output/v1",
                "request": req["identity"],
                "native": native,
                "result": result,
                "elapsed_us": (time.monotonic_ns() - start) // 1000,
            }
        ),
    )


if __name__ == "__main__":
    main()
