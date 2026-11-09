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
    forced=None,
    component_library=None,
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
    if component_library is not None:
        from oeis_learn.generalization.progressive.component_search import validate_input

        validate_input(component_library)
        if methods != ("correct",) or forced is not None:
            raise ValueError("component comparison requires one ordinary matched portfolio")
    prompts = list(prompts)
    for p in prompts:
        validate_prompt(p)
    prompts.sort(key=lambda p: p["id"])
    if not prompts or len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("nonempty unique visible roster required")
    if "shuffled" in methods:
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
    runtime_recipe = recipe.get("base_recipe", recipe)
    if forced is not None and (
        set(forced) != {p["id"] for p in prompts}
        or any(
            not isinstance(v, list) or len(v) > 8 or any(not isinstance(s, str) for s in v)
            for v in forced.values()
        )
    ):
        raise ValueError("bounded diagnostic candidate roster required")
    controlled = (
        conditioning(prompts, "shuffled", seed=recipe["model"]["seed"])
        if "shuffled" in methods
        else prompts
    )
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
                    *(
                        [("shuffled", controlled[ordinal]["values"])]
                        if "shuffled" in methods
                        else []
                    ),
                ]:
                    memories[mode] = encoder.forward_from_sequences(
                        [[int(v) for v in values]], device=device
                    )
                memory = memories["correct"]
                padding = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
                memories["absent"] = condition_memory(memory, padding, "absent")[0]
            record = {"selections": {}, "arm_evidence": {}, "costs": {}}
            if forced is not None:
                visible = VisiblePrompt(tuple(p["indices"]), tuple(p["values"]))
                admitted = []
                for source in forced[p["id"]]:
                    if budget:
                        budget.consume(draws=1, attempts=1, rejections=1)
                    result = client.fit(
                        source,
                        visible,
                        policy=fitting_policy(source),
                        deadline=min(deadline, time.monotonic() + 4),
                    )
                    admitted.extend(result.get("admitted", []))
                chosen = _seal(admitted, visible, cost=0)
                for method in (
                    "raw",
                    "rescue",
                    "equal_budget",
                    "shuffled",
                    "absent",
                    "template",
                    "polynomial",
                    "linear_recurrence",
                ):
                    selected = method in {"raw", "rescue", "equal_budget"}
                    record["selections"][method] = chosen if selected else None
                    record["arm_evidence"][method] = dict(
                        complete=selected, reason="DIAGNOSTIC_ONLY"
                    )
                record["costs"]["diagnostic"] = dict(autonomous_discoveries=0)
                rows[p["id"]] = record
                continue
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
                proposals = None
                if component_library is not None:
                    from oeis_learn.generalization.progressive.component_search import Proposals

                    proposals = Proposals(
                        component_library["library"],
                        decoder=decoder,
                        memory=memories[mode],
                        charge=budget.consume if budget else None,
                    )
                value = portfolio(
                    decoder,
                    memories.get(mode),
                    p,
                    runtime_recipe,
                    client,
                    deadline=deadline,
                    attempt_count=8,
                    template=mode == "template",
                    mode=mode,
                    snapshot=snapshot,
                    evidence=evidence,
                    evaluation_policy=DRAIN,
                    component_proposals=proposals,
                )
                for name in ("raw", "rescue", "equal_budget") if mode == "correct" else (mode,):
                    key = name if mode == "correct" else "equal_budget"
                    record["selections"][name] = value[key]
                    record["arm_evidence"][name] = value["arm_evidence"][key]
                record["costs"][mode] = value
                if proposals is not None:
                    record["costs"]["components"] = dict(
                        proposals.report,
                        discovery_work=component_library["discovery_work"],
                        commit=component_library["commit"],
                        cost_scope="discovery_plus_search",
                    )
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


def generate_short(encoder, decoder, prompts, *, seed, seconds=4, max_tokens=64):
    """Visible-only four-input generation through the shared constrained sampler."""
    import torch
    from oeis_learn.generalization.progressive.short_profile import collate
    from oeis_learn.decoder.bigint_generation import GenerationRequest, sample_batch

    if (
        type(seconds) not in (int, float)
        or not 0 < seconds <= 60
        or type(max_tokens) is not int
        or not 1 <= max_tokens <= 1024
    ):
        raise ValueError("bounded short generation required")
    if not 1 <= len(prompts) <= 8:
        raise ValueError("short prompt bucket 1..8 required")
    rows = collate(prompts)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        memory = encoder.forward_from_sequences(
            [[int(v) for v in r] for r in rows["values"]], device=next(encoder.parameters()).device
        )
    if memory.shape[1] != 4:
        raise ValueError("short encoder input positions differ")
    deadline = time.monotonic() + seconds
    requests = [
        GenerationRequest(memory[i : i + 1], seed + i, 0.8, deadline, max_tokens)
        for i in range(len(prompts))
    ]
    return sample_batch(decoder, requests)


def score_short(source, truth, *, client):
    """Separate nine-term scorer; truth is never an input to generation."""
    from oeis_learn.generalization.progressive.short_profile import validate_terms
    from oeis_learn.synthesis.reference import evaluate

    validate_terms(truth)
    if client is None:
        raise ValueError("native short score required")
    deadline = time.monotonic() + 4
    reference_result = evaluate(source, truth["indices"], deadline=deadline)
    native = client.execute(source, truth["indices"], deadline=deadline)
    if (
        native.get("outcome") == "ok"
        and reference_result.outcome == "ok"
        and native["values"] != list(reference_result.values)
    ):
        raise ValueError("short scorer native/reference disagreement")
    agreed = (
        native.get("outcome") == "ok"
        and reference_result.outcome == "ok"
        and native["values"] == truth["values"]
    )
    return seal(
        dict(
            version="progressive-short-score/v1",
            source=digest(source),
            truth=digest(truth),
            tested_indices=truth["indices"],
            agreement=agreed,
            primary_scaling_eligible=False,
        )
    )
