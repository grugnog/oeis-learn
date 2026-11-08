"""Installed visible generation and separate final-source scoring adapters.

The generator API has no truth parameter. Its immutable candidate artifact must
be referenced and verified before the scorer can inspect continuation truth.
"""

import time

from oeis_learn.generalization.progressive.baselines import (
    Portfolio,
    iter_candidates,
    load_baseline,
)
from oeis_learn.generalization.progressive.cohorts import validate_prompt
from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    verify_refs,
)
from oeis_learn.pipeline_validation.identity import confined, seal


def generate(
    root,
    output,
    prompts,
    recipe,
    *,
    client,
    checkpoint=None,
    model_control="retained_D",
    reference_recipe=None,
    seed=None,
    device="cpu",
    seconds=3600,
    output_root=None,
    budget=None,
    methods=("correct", "shuffled", "absent", "template", "polynomial", "linear_recurrence"),
):
    import torch

    from oeis_learn.decoder.bigint_codec import fitting_policy
    from oeis_learn.fresh.bigint_diagnostics import _seal, model_snapshot, portfolio
    from oeis_learn.fresh.conditioning_controls import condition_memory
    from oeis_learn.generalization.progressive.baselines import conditioning
    from oeis_learn.synthesis.admission import VisiblePrompt

    from .pilot_clocks import DRAIN, FullClockEvidence

    if type(seconds) not in (int, float) or not 0 < seconds <= 21600:
        raise ValueError("bounded installed generation required")
    if (
        not isinstance(methods, tuple)
        or not methods
        or len(set(methods)) != len(methods)
        or any(
            m
            not in ("correct", "shuffled", "absent", "template", "polynomial", "linear_recurrence")
            for m in methods
        )
    ):
        raise ValueError("frozen registered method roster required")
    prompts = list(prompts)
    for p in prompts:
        validate_prompt(p)
    prompts.sort(key=lambda p: p["id"])
    if not prompts or len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("nonempty unique visible roster required")
    conditioning(prompts, "shuffled", seed=recipe["model"]["seed"])
    began = time.monotonic()
    deadline = min(began + seconds, budget.deadline) if budget else began + seconds
    output_root = root if output_root is None else output_root
    output = confined(output_root, output, exists=False)
    if output.exists():
        raise ValueError("fresh generation output required")
    output.mkdir(parents=True)
    encoder, decoder, counts = load_baseline(
        model_control,
        recipe,
        root=root if checkpoint else None,
        ref=checkpoint,
        seed=seed,
        reference_recipe=reference_recipe,
        device=device,
    )
    snapshot = model_snapshot(encoder, decoder)
    controlled = conditioning(prompts, "shuffled", seed=recipe["model"]["seed"])
    rows = {}
    evidence = FullClockEvidence(
        output / "attempts", prompts, source=digest(recipe), work=snapshot, policy=DRAIN
    )
    try:
        for ordinal, p in enumerate(prompts):
            memories = {}
            with torch.no_grad():
                for mode, values in [
                    ("correct", p["values"]),
                    ("shuffled", controlled[ordinal]["values"]),
                ]:
                    memories[mode] = encoder.forward_from_sequences(
                        [[int(v) for v in values]], device=device
                    )
                memory = memories["correct"]
                padding = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
                memories["absent"] = condition_memory(memory, padding, "absent")[0]
            record = {"selections": {}, "arm_evidence": {}, "costs": {}}
            for mode in ("correct", "shuffled", "absent", "template"):
                if mode not in methods:
                    for key in ("raw", "rescue", "equal_budget") if mode == "correct" else (mode,):
                        record["selections"][key] = None
                        record["arm_evidence"][key] = {"complete": False, "reason": "NOT_SELECTED"}
                    continue
                # The canonical synchronous driver drains native work before
                # asking the model for another draw. No second scheduler.
                if budget:
                    budget.consume(draws=8, attempts=8, rejections=8)
                value = portfolio(
                    decoder,
                    memories.get(mode),
                    p,
                    recipe,
                    client,
                    deadline=deadline,
                    attempt_count=8,
                    template=mode == "template",
                    mode=mode,
                    snapshot=snapshot,
                    evidence=evidence,
                    evaluation_policy=DRAIN,
                )
                for name in ("raw", "rescue", "equal_budget") if mode == "correct" else (mode,):
                    key = name if mode == "correct" else "equal_budget"
                    record["selections"][name] = value[key]
                    record["arm_evidence"][name] = value["arm_evidence"][key]
                record["costs"][mode] = value
            visible = VisiblePrompt(tuple(p["indices"]), tuple(p["values"]))
            for method in ("polynomial", "linear_recurrence"):
                portfolio_budget = Portfolio()
                admitted = []
                complete = True
                stream = iter_candidates(method, p, portfolio_budget) if method in methods else ()
                if method not in methods or time.monotonic() >= deadline:
                    complete = False
                    stream = ()
                if budget and method in methods:
                    budget.consume(attempts=8, rejections=8)
                for row in stream:
                    if not row["applicable"]:
                        continue
                    start = time.monotonic()
                    result = client.fit(
                        row["source"],
                        visible,
                        policy=fitting_policy(row["source"]),
                        deadline=min(deadline, row["fit_deadline"]),
                    )
                    finish = time.monotonic()
                    if finish >= min(deadline, row["fit_deadline"]):
                        complete = False
                    else:
                        admitted.extend(
                            dict(item, _original_source=row["source"])
                            for item in result.get("admitted", [])
                        )
                        complete &= result.get("complete") is True
                    portfolio_budget.observe(
                        method,
                        candidate=digest(row["source"]),
                        started=start,
                        ended=finish,
                        outcome=result.get("outcome"),
                        phase="fit",
                    )
                record["selections"][method] = _seal(
                    admitted, visible, cost=portfolio_budget.report()["total_elapsed_us"] / 1e6
                )
                record["arm_evidence"][method] = {
                    "complete": complete and time.monotonic() < deadline,
                    "reason": "conventional_visible_only",
                }
                record["costs"][method] = portfolio_budget.report()
            rows[p["id"]] = record
    finally:
        evidence.finish()
    if model_snapshot(encoder, decoder) != snapshot:
        raise RuntimeError("generator mutated frozen weights")
    result = seal(
        {
            "version": "discovery-candidates/v1",
            "model": snapshot,
            "checkpoint": checkpoint["sha256"] if checkpoint else None,
            "model_control": model_control,
            "recipe": digest(recipe),
            "methods": list(methods),
            "prompts": digest(prompts),
            "records": rows,
            "clock": evidence.clock_summary,
            "counts": counts,
            "elapsed_us": int((time.monotonic() - began) * 1e6),
        }
    )
    atomic_write(output / "candidates.json", result)
    return reference(output_root, output / "candidates.json")


def score(root, candidates, truth, *, client, output, seconds=3600, budget=None):
    from oeis_learn.fresh.bigint_diagnostics import score as canonical_score
    from oeis_learn.synthesis.contracts import canonical_integer

    verify_refs(root, [candidates])
    value = load(confined(root, candidates["path"]), maximum=512 << 20)
    if (
        value.get("version") != "discovery-candidates/v1"
        or seal(value) != value
        or set(value)
        != {
            "version",
            "identity",
            "model",
            "model_control",
            "checkpoint",
            "recipe",
            "methods",
            "prompts",
            "records",
            "clock",
            "counts",
            "elapsed_us",
        }
    ):
        raise ValueError("sealed generator candidate artifact required")
    if type(seconds) not in (int, float) or not 0 < seconds <= 21600:
        raise ValueError("bounded scorer required")
    truth = list(truth)
    by_id = {}
    for row in truth:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "indices", "values"}
            or row["id"] in by_id
        ):
            raise ValueError("closed unique scoring truth required")
        if len(row["indices"]) != 100 or len(row["values"]) != 100:
            raise ValueError("100 exact terms required")
        xs = [canonical_integer(x) for x in row["indices"]]
        for y in row["values"]:
            canonical_integer(y)
        if xs != list(range(xs[0], xs[0] + 100)):
            raise ValueError("exact source indices required")
        by_id[row["id"]] = row
    if set(by_id) != set(value["records"]):
        raise ValueError("candidate/truth membership differs")
    truth = sorted(truth, key=lambda row: row["id"])
    prompts = [
        {k: r[k][:20] if k != "id" else r[k] for k in ("id", "indices", "values")} for r in truth
    ]
    if digest(prompts) != value["prompts"]:
        raise ValueError("visible truth differs from generation")
    deadline = (
        min(time.monotonic() + seconds, budget.deadline) if budget else time.monotonic() + seconds
    )
    methods = (
        "raw",
        "rescue",
        "equal_budget",
        "shuffled",
        "absent",
        "template",
        "polynomial",
        "linear_recurrence",
    )
    outcomes = {r["id"]: {} for r in truth}
    for method in methods:
        if budget:
            budget.consume(attempts=len(truth), rejections=len(truth))
        projected = {
            k: {
                "selections": {"equal_budget": v["selections"][method]},
                "arm_evidence": {"equal_budget": v["arm_evidence"][method]},
            }
            for k, v in value["records"].items()
        }
        scored = canonical_score(projected, by_id, client, deadline=deadline)
        for key, row in scored["outcomes"].items():
            v = row["equal_budget"]
            upstream = value["records"][key]["arm_evidence"][method]
            v["complete"] = v["complete"] and upstream.get("complete") is True
            v["outcome"] = (
                "success" if v["success"] else "failure" if v["complete"] else "unresolved"
            )
            outcomes[key][method] = v
    result = seal(
        {
            "version": "discovery-scoring/v1",
            "candidates": candidates["sha256"],
            "outcomes": outcomes,
            "denominator": len(truth),
            "scientific_status": "UNMEASURED",
            "claim": "finite_agreement_only",
        }
    )
    output = confined(root, output, exists=False)
    atomic_write(output / "scores.json", result)
    return result
