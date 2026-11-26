"""Measure a cold resident learner and independently score its exposed choices."""

from oeis_learn.generalization.progressive.contracts import read_bound_record, reference, seal

from .preparation_recovery import retain
from .progressive_panel_sources import require
from oeis_learn.generalization.progressive.program_search_pilot import (
    ARMS,
    LOOKS,
    RESIDENT_SUPERVISOR_GRACE_SECONDS,
    resident_seconds,
)
from .progressive_pilot_recovery import qualification_role
from .progressive_research_inputs import checked_file


def score_choices(cpu, block, truth, candidates, path, seconds):
    """Reopen sealed choices when an otherwise successful scorer missed a selected program."""
    from .progressive_round028_pilot import PilotDeficit

    ref = dict(reference(path.parent, path), path="inputs/candidates.json")
    chosen = {
        key
        for key, row in candidates["records"].items()
        if row["selections"]["equal_budget"] is not None
    }
    ordinal = 0
    while True:
        name = block if ordinal == 0 else f"{block}-rescore-{ordinal}"
        data = seal(dict(block=name, rows=truth, candidates=ref, seconds=seconds))
        result = qualification_role(
            cpu,
            "score",
            data,
            seconds=seconds + RESIDENT_SUPERVISOR_GRACE_SECONDS,
            inputs=[(path, "/work/inputs/candidates.json")],
        )
        scores = read_bound_record(
            cpu.last_output, result["scores"], version="discovery-scoring/v1"
        )
        outcomes = scores["outcomes"]
        require(
            scores["candidates"] == ref["sha256"]
            and scores["denominator"] == len(truth)
            and set(outcomes) == set(candidates["records"]) == {r["id"] for r in truth},
            "qualification score belongs to different choices or truth",
        )
        missing = 0
        for key, value in outcomes.items():
            row = value["equal_budget"]
            selection = candidates["records"][key]["selections"]["equal_budget"]
            expected = selection["selected"]["source_sha256"] if selection is not None else None
            require(
                row["candidate_sha256"] == expected,
                "independent scorer checked a different program",
            )
            checked = row["scoring"]
            if key in chosen:
                if not (
                    checked.get("complete") is True
                    and all(
                        checked.get(engine, {}).get("outcome") == "ok"
                        and checked[engine].get("coverage") == 100
                        for engine in ("production", "reference")
                    )
                ):
                    missing += 1
            else:
                require(
                    checked["outcome"] == "no_selected_candidate",
                    "unexpected qualification selection",
                )
        if not missing:
            return scores, result["scores"], len(chosen)
        # Retain the original timing and choices. Each invocation may try the scorer once more.
        failure = cpu.area / "qualification-costs" / f"{name}-incomplete-score.json"
        already_reported = failure.exists()
        retain(
            failure,
            seal(
                dict(
                    request=data["identity"],
                    producer=cpu.last_producer["identity"],
                    role="score",
                    scores=result["scores"]["sha256"],
                    lease_us=0,
                    wall_us=cpu.last_wall_us,
                    unscored_choices=missing,
                )
            ),
        )
        if not already_reported:
            raise PilotDeficit(
                "QUALIFICATION_SCORER_INCOMPLETE",
                selected_choices=len(chosen),
                unscored_choices=missing,
            )
        ordinal += 1


def resident_scale(cpu, gpu, prompts, truth, recipe, weight, weights, seed, progress):
    from .pilot_clocks import CORRECT_FULL
    from .progressive_round028_pilot import score_projection

    count = 160
    seconds = resident_seconds(count)
    allowance = seconds * 1_000_000
    supervisor = (seconds + RESIDENT_SUPERVISOR_GRACE_SECONDS) * 1_000_000
    ordinal = 0
    while True:
        # After a segmented recovery, keep and score those choices. Use different exposed
        # aliases for the single-process timing, which cannot be inferred from the pieces.
        group = "qualification" if ordinal == 0 else f"qualification-cold-{ordinal}"
        block = (
            "qualify-resident-endpoint"
            if ordinal == 0
            else f"qualify-resident-endpoint-cold-{ordinal}"
        )
        visible = [dict(prompts[i % len(prompts)], id=f"{group}-{i:03d}") for i in range(count)]
        truths = [dict(truth[i % len(truth)], id=f"{group}-{i:03d}") for i in range(count)]
        material = seal(
            dict(
                block=block,
                arm=ARMS[0],
                target=LOOKS[-1],
                inference_seed=seed,
                rows=visible,
                recipe=recipe,
                weights=weight,
                seconds=seconds,
            )
        )
        progress("accept_frozen_pilot", role="full_size_search", cold_attempt=ordinal)
        result = qualification_role(
            gpu,
            "search",
            material,
            seconds=seconds + RESIDENT_SUPERVISOR_GRACE_SECONDS,
            inputs=[(weights, "/work/inputs/weights.pt")],
        )
        ref = result["candidates"]
        path = checked_file(gpu.last_output, ref)
        generated = read_bound_record(gpu.last_output, ref, version="discovery-candidates/v1")
        search = gpu.last_producer
        progress("accept_frozen_pilot", role="full_size_score", cold_attempt=ordinal)
        scored, score_ref, selected = score_choices(cpu, block, truths, generated, path, seconds)
        _, closed = score_projection(
            generated,
            scored,
            ref["sha256"],
            visible,
            weight,
            recipe,
            CORRECT_FULL,
            inference_seed=seed,
        )
        require(closed, "exposed endpoint-sized learner qualification incomplete")
        if "recovered" not in search:
            break
        ordinal += 1

    clocks = generated["clock"]["clocks"]
    # Host cleanup can be proven after a reboot while its elapsed wall time remains unknown.
    search_wall, score_wall = gpu.last_wall_us, cpu.last_wall_us
    score_measurements = cpu.last_producer["result"].get("measurements")
    return seal(
        dict(
            cases=len(clocks),
            exposed_problems=len(prompts),
            selected_choices=selected,
            no_selection=count - selected,
            cold_attempt=ordinal,
            candidates_bytes=ref["bytes"],
            scored_bytes=score_ref["bytes"],
            candidate=ref["sha256"],
            scores=score_ref["sha256"],
            search_us=generated["elapsed_us"],
            search_allowance_us=allowance,
            search_margin_us=allowance - generated["elapsed_us"],
            slowest_case_us=max(round((c["finish"] - c["start"]) * 1e6) for c in clocks),
            search_worker_us=search["elapsed_us"],
            search_supervisor_us=search_wall,
            search_supervisor_allowance_us=supervisor,
            search_supervisor_margin_us=None if search_wall is None else supervisor - search_wall,
            score_allowance_us=allowance,
            score_margin_us=None
            if score_measurements is None
            else allowance - score_measurements["score_choices_us"],
            score_worker_us=cpu.last_producer["elapsed_us"],
            score_supervisor_us=score_wall,
            score_supervisor_allowance_us=supervisor,
            score_supervisor_margin_us=None if score_wall is None else supervisor - score_wall,
            search_measurements=result.get("measurements"),
            score_measurements=score_measurements,
        )
    )
