"""Isolated native CPU work for outcome-blind supply and dedicated calibration families."""

import os
import time
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal

METHODS = ("templates", "polynomial", "linear_recurrence")
CHUNK = 256
CHUNKS = 8


def native_draws(data, client):
    from oeis_learn.generalization.progressive.complexity import measure
    from oeis_learn.generalization.progressive.generator import default_policy, draw
    from oeis_learn.generalization.progressive.prepared_support import transfer_support
    from oeis_learn.generalization.progressive.research_sampling import Sampler

    from .progressive_preparation import verify_draw
    from .progressive_research_census import classify

    count, start = data["count"], data["start"]
    if count != CHUNK or start not in range(0, CHUNK * CHUNKS, CHUNK):
        raise ValueError("unregistered prospective draw interval")
    package = data["support"]
    transfer, _ = transfer_support(package)
    sampler = Sampler(package, seed=data["seed"])
    law = default_policy(seed=data["seed"])
    reasons, rows = {}, []
    for i in range(start, start + count):
        sample = sampler.draw(i // 2) if i % 2 == 0 else draw(law, i // 2)
        measurements = {}
        row, reason = verify_draw(sample, client=client, measurements=measurements)
        if row is None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        rows.append(
            seal(
                dict(
                    row,
                    id="mechanical:" + row["draw_identity"],
                    membership="evaluation_candidate",
                    stratum=classify(row, transfer),
                    difficulty=measure(
                        row["source"], reference_steps=measurements["reference_steps"]
                    ),
                )
            )
        )
    return {"attempted": count, "rejected": reasons, "rows": rows}


def conventional_choice(method, prompt, client):
    from oeis_learn.decoder.bigint_codec import fitting_policy
    from oeis_learn.fresh.bigint_diagnostics import _seal
    from oeis_learn.generalization.progressive.baselines import Portfolio, iter_candidates
    from oeis_learn.synthesis.admission import VisiblePrompt

    visible = VisiblePrompt(tuple(prompt["indices"]), tuple(prompt["values"]))
    portfolio = Portfolio()
    admitted, complete = [], True
    for proposal in iter_candidates(method, prompt, portfolio):
        if not proposal["applicable"]:
            continue
        source = proposal["source"]
        deadline = proposal["fit_deadline"]
        began = time.monotonic()
        try:
            result = client.fit(source, visible, policy=fitting_policy(source), deadline=deadline)
        except TimeoutError:
            result = {"complete": False, "admitted": [], "outcome": "deadline"}
        finish = time.monotonic()
        portfolio.observe(
            method,
            candidate=digest(source),
            started=began,
            ended=finish,
            outcome=result.get("outcome"),
            phase="fit",
        )
        if finish < deadline:
            admitted.extend(dict(r, _original_source=source) for r in result.get("admitted", []))
        complete &= finish < deadline and result.get("complete") is True
    complete &= time.monotonic() < portfolio.deadline
    return {
        "choice": _seal(admitted, visible, cost=portfolio.report()["total_elapsed_us"] / 1e6),
        "complete": complete,
        "attempts": portfolio.attempts,
    }


def calibration_choices(data, client):
    # This process has no truth or working source for any calibration family.
    return {
        r["id"]: {m: conventional_choice(m, r, client) for m in METHODS} for r in data["prompts"]
    }


def calibration_scores(data, client):
    from oeis_learn.fresh.bigint_diagnostics import score
    from oeis_learn.synthesis.reference import evaluate

    choices = data["choices"]
    truth = {r["id"]: r for r in data["truth"]}
    if set(choices) != set(truth) or any(set(v) != set(METHODS) for v in choices.values()):
        raise ValueError("calibration scorer input differs from the sealed choices")
    if set(data["witnesses"]) != set(truth):
        raise ValueError("calibration witness roster changed")
    for key, row in truth.items():
        source = data["witnesses"][key]
        native = client.execute(source, row["indices"], deadline=time.monotonic() + 4)
        independent = evaluate(source, row["indices"], client.limits, deadline=time.monotonic() + 4)
        if (
            native.get("outcome") != "ok"
            or independent.outcome != "ok"
            or not (native["values"] == row["values"] == list(independent.values))
        ):
            raise ValueError("calibration witness differs on the original 100 indices")
    scored = {}
    for method in METHODS:
        inputs = {
            k: {
                "selections": {"equal_budget": v[method]["choice"]},
                "arm_evidence": {
                    "equal_budget": {"complete": v[method]["complete"], "reason": "conventional"}
                },
            }
            for k, v in choices.items()
        }
        raw = score(inputs, truth, client, deadline=time.monotonic() + len(truth) * 30 + 1)[
            "outcomes"
        ]
        for key, result in raw.items():
            verdict = result["equal_budget"]
            measured = verdict["success"] or (
                choices[key][method]["complete"] and verdict["complete"]
            )
            scored.setdefault(key, {})[method] = {"exact": verdict["success"], "measured": measured}
    return scored


def main():
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.synthesis.client import SynthesisClient

    root = Path("/work/inputs")
    request = load(root / "request.json")
    if (
        seal(request) != request
        or request["version"] != "round028-panel-role/v1"
        or request["source"] != digest(source_snapshot())
        or request["engine"] != os.environ["OEIS_QUALIFICATION_ENGINE"]
        or request["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
    ):
        raise ValueError("candidate panel role authority differs")
    data = load(root / "data.json", maximum=128 << 20)
    if seal(data) != data or data["identity"] != request["data"]:
        raise ValueError("panel role data changed")
    from .progressive_round028_replay import replay_attempts
    from .progressive_round028_selector import selector_choice, selector_score
    operations = {"selector_choice": selector_choice, "selector_score": selector_score,
                  "fit_replay": replay_attempts, "draws": native_draws, "choose": calibration_choices, "score": calibration_scores}
    began = time.monotonic_ns()
    with SynthesisClient(digest(["round028-panel-role", request["identity"]])) as client:
        native = client.runtime_identity("bigint-body/v1")
        if request["candidate_native"] is not None and native != request["candidate_native"]:
            raise ValueError("native identity differs between the candidate roles")
        result = operations[request["role"]](data, client)
    atomic_write(
        Path("/work/output/producer.json"),
        seal(
            {
                "version": "round028-panel-output/v1",
                "request": request["identity"],
                "native": native,
                "result": result,
                "elapsed_us": (time.monotonic_ns() - began) // 1000,
            }
        ),
    )


if __name__ == "__main__":
    main()
