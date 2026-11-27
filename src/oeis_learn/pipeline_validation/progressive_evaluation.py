"""Installed visible generation and separate final-source scoring adapters.

The generator API has no truth parameter. Its immutable candidate artifact must
be referenced and verified before the scorer can inspect continuation truth.
"""

import time
from pathlib import Path

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
from .pilot_clocks import CORRECT_FULL, DESCRIPTIVE, DRAIN, FullClockEvidence, attempts


def model_receipt(look, *, profile, weights, checkpoint, origin):
    """The custodian commits the model before a visible-only probe is issued."""
    from oeis_learn.generalization.progressive.contracts import _check, schema

    zero = look["look"] == 0 and look["method"] != "D"
    if (checkpoint is None) != zero:
        raise ValueError("a trained or incumbent probe requires its registered checkpoint")
    for ref in (look["registration"], profile, weights, origin, *(() if zero else (checkpoint,))):
        _check(schema("common")["$defs"]["hash"], ref, {})
    return seal(dict(version="round029-probe-model/v1",
                     **{k: look[k] for k in ("registration", "method", "seed", "look")},
                     profile=profile, weights=weights, checkpoint=checkpoint, origin=origin))


def initial_probe_model(look, profile, start):
    from oeis_learn.generalization.progressive.velocity_trials import validate_start

    origin = validate_start(start)
    if (look["look"] != 0 or profile["model"]["seed"] != look["seed"]
            or origin["initialization"] != look["seed"] or origin["codec"] != profile["codec"]):
        raise ValueError("probe requires its independently prepared random initialization")
    return model_receipt(look, profile=digest(profile), weights=origin["initial_weights"],
                         checkpoint=None, origin=origin["identity"])


def donor_probe_model(root, look, profile, checkpoint):
    """Reopen the original donor inventory before projecting weights to the search role."""
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from .progressive_velocity_donor import inventory

    method = look["method"]
    cuts = inventory(root)["cuts"]
    cut = next((r for r in cuts if r["method"] == method and
                r["look"] == (None if method == "D" else look["look"])), None)
    if (cut is None or not cut["inference"] or cut["weights"] != checkpoint["sha256"]
            or method == "D" and cut["recipe"] != digest(profile)
            or method != "D" and (method != "00" or cut["seeds"]["initialization"] != look["seed"]
                                   or cut["codec"] != profile["codec"])):
        raise ValueError("registered donor weights or model profile differs")
    control = model_control_for(look)
    enc, dec, _ = load_baseline(control, profile, root=root, ref=checkpoint,
                                reference_recipe=profile if method != "D" else None)
    return model_receipt(look, profile=digest(profile), weights=model_snapshot(enc, dec),
                         checkpoint=cut["weights"], origin=digest([cut["registration"], cut["lineage"], cut["weights"]]))


def checked_probe_model(look, model):
    from oeis_learn.generalization.progressive.campaign import velocity_policy

    if look["method"] in velocity_policy()["controls"]:
        if model is not None:
            raise ValueError("conventional probe received a neural model")
    elif not isinstance(model, dict) or model != model_receipt(
        look, **{k: model.get(k) for k in ("profile", "weights", "checkpoint", "origin")}
    ):
        raise ValueError("registered model for this method, seed and look required")
    return model


def model_control_for(look):
    return "retained_D" if look["method"] == "D" else "random" if look["look"] == 0 else "earlier_checkpoint"


def velocity_probe(look, prompts, *, mode, inference_seed, model=None):
    """Freeze encoder conditioning separately from the recipient used by the fitter and scorer."""
    from collections import defaultdict
    from oeis_learn.generalization.progressive.campaign import velocity_policy
    from oeis_learn.generalization.progressive.graph import velocity_look
    from oeis_learn.generalization.progressive.metrics import _seed

    cfg = velocity_policy()
    _seed(inference_seed)
    rebuilt = velocity_look(
        registration=look["registration"],
        method=look["method"],
        seed=look["seed"],
        look=look["look"],
        profile=look["profile"],
        extension=look["look"] == cfg["conditional_look"],
        development=[row for row in look["members"] if row["stratum"] != "retention"],
        retention=[row for row in look["members"] if row["stratum"] == "retention"],
    )
    if (
        look != rebuilt
        or mode not in {"true", "shuffled"}
        or look["method"] in cfg["controls"]
        and (look["profile"] != "full" or mode != "true")
        or [r["task"] for r in look["members"]] != [validate_prompt(p)["id"] for p in prompts]
        or not prompts
        or len({p["id"] for p in prompts}) != len(prompts)
    ):
        raise ValueError("registered ordered recipient roster and probe opportunity required")
    donors = {p["id"]: p["id"] for p in prompts}
    if mode == "shuffled":
        strata = defaultdict(list)
        for member, prompt in zip(look["members"], prompts, strict=True):
            strata[member["stratum"]].append(prompt)
        for stratum, recipients in strata.items():
            groups = defaultdict(list)
            for p in recipients:
                groups[tuple(p["values"])].append(p["id"])
            shift = max(map(len, groups.values()))
            if 2 * shift > len(recipients):
                raise ValueError(f"no differing donor permutation in registered stratum {stratum}")
            order = [
                key
                for group in sorted(groups, key=lambda v: digest([inference_seed, v]))
                for key in sorted(groups[group], key=lambda v: digest([inference_seed, v]))
            ]
            donors.update({key: order[(i + shift) % len(order)] for i, key in enumerate(order)})
    return seal(
        dict(
            version="round029-probe/v2",
            model=checked_probe_model(look, model),
            look=look,
            recipients=digest(prompts),
            mode=mode,
            inference_seed=inference_seed,
            donors=donors,
        )
    )


def probe_inference(probe):
    """Identity of the actual opportunity, independent of aliases for D or a shared zero look."""
    look = probe["look"]
    method = look["method"]
    if method == "D":
        coordinate = "D"
    elif look["look"] == 0:
        coordinate = ["initial", look["seed"]]
    else:
        coordinate = [method, look["seed"], look["look"]]
    model = probe["model"]
    return digest([coordinate, look["profile"], look["opportunity"], look["attempts"],
                   probe["recipients"], probe["mode"], probe["inference_seed"], probe["donors"],
                   [model[k] for k in ("profile", "weights", "checkpoint")] if model else method])


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
    evaluation_policy=DRAIN,
    case_elapsed_us=None,
    case_boundary_dir=None,
    inference_seed=None,
    measurements=None,
    registered_probe=None,
):
    prompts = list(prompts)
    if type(seconds) not in (int, float) or not 0 < seconds <= _maximum_seconds(budget):
        raise ValueError("bounded installed generation required")
    if registered_probe is not None:
        from oeis_learn.generalization.progressive.campaign import velocity_policy

        look = registered_probe["look"]
        if (
            registered_probe
            != velocity_probe(
                look,
                prompts,
                mode=registered_probe["mode"],
                inference_seed=registered_probe["inference_seed"],
                model=registered_probe.get("model"),
            )
            or inference_seed is not None
            or evaluation_policy != DRAIN
            or forced is not None
            or component_library is not None
        ):
            raise ValueError("original registered probe required")
        if look["method"] in velocity_policy()["controls"]:
            from .progressive_velocity_controls import generate_control

            if methods != (look["method"],) or checkpoint is not None or device != "cpu":
                raise ValueError(
                    "registered CPU control requires its own method and visible inputs"
                )
            return generate_control(
                output_root if output_root is not None else root,
                output,
                prompts,
                recipe,
                probe=registered_probe,
                client=client,
                seconds=seconds,
                budget=budget,
                case_elapsed_us=case_elapsed_us,
                case_boundary_dir=case_boundary_dir,
                measurements=measurements,
            )
        if methods != ("correct",) or look["method"] not in {
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "D",
        }:
            raise ValueError("original registered neural probe required")
        registered = registered_probe["model"]
        if (model_control != model_control_for(look) or digest(recipe) != registered["profile"]
                or (checkpoint["sha256"] if checkpoint else None) != registered["checkpoint"]
                or model_control == "random" and seed != look["seed"]):
            raise ValueError("model profile, checkpoint or initialization differs from the registered probe")
        inference_seed = registered_probe["inference_seed"]
        evaluation_policy = DESCRIPTIVE if look["profile"] == "common" else CORRECT_FULL
    if evaluation_policy not in (DRAIN, DESCRIPTIVE, CORRECT_FULL) or (
        evaluation_policy in (DESCRIPTIVE, CORRECT_FULL)
        and (methods != ("correct",) or forced is not None or component_library is not None)
    ):
        raise ValueError("timed candidate generation requires the registered visible-only arm")

    import torch
    from oeis_learn.decoder.bigint_codec import fitting_policy
    from oeis_learn.fresh.bigint_diagnostics import _seal, model_snapshot, portfolio
    from oeis_learn.fresh.conditioning_controls import condition_memory
    from oeis_learn.generalization.progressive.baselines import conditioning
    from oeis_learn.synthesis.admission import VisiblePrompt

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
    for p in prompts:
        validate_prompt(p)
    if registered_probe is None:
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
    if measurements is not None:
        load_started = time.monotonic_ns()
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
    if registered_probe is not None and snapshot != registered_probe["model"]["weights"]:
        raise ValueError("resolved model tensors differ from the registered probe")
    if measurements is not None:
        measurements["load_and_identify_checkpoint_us"] = (time.monotonic_ns() - load_started) // 1000
    runtime_recipe = recipe.get("base_recipe", recipe)
    if inference_seed is not None:
        if type(inference_seed) is not int or inference_seed < 0:
            raise ValueError("invalid registered inference seed")
        runtime_recipe = dict(runtime_recipe, model=dict(runtime_recipe["model"], seed=inference_seed))
    draw_identity = {} if inference_seed is None else {"inference_seed": inference_seed}
    encoder_values = {p["id"]: p["values"] for p in prompts}
    if registered_probe is not None:
        draw_identity["velocity_probe"] = registered_probe["identity"]
        encoder_values = {key: encoder_values[donor] for key, donor in registered_probe["donors"].items()}
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
        output / "attempts", prompts, source=digest(recipe), work=snapshot, policy=evaluation_policy
    )
    try:
        for ordinal, p in enumerate(prompts):
            if inference_seed is not None and time.monotonic() >= deadline:
                raise TimeoutError("resident search stopped before opening the next portfolio")
            case_start = time.monotonic_ns() if case_elapsed_us is not None else None
            if case_boundary_dir is not None:
                boundary = Path(case_boundary_dir) / digest(p["id"])[7:]
                boundary.mkdir(parents=True, exist_ok=False)
                atomic_write(boundary / "start.json", seal(dict(prompt=digest(p), model=snapshot, policy=evaluation_policy, **draw_identity)))
            memories = {}
            with torch.no_grad():
                for mode, values in [
                    ("correct", encoder_values[p["id"]]),
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
                complete=True
                expired=0
                for source in forced[p["id"]]:
                    if budget:
                        budget.consume(draws=1, attempts=1, rejections=1)
                    fit_deadline=min(deadline,time.monotonic()+4)
                    try:
                        result = client.fit(source,visible,policy=fitting_policy(source),deadline=fit_deadline)
                    except TimeoutError:
                        complete=False
                        expired+=1
                        continue
                    if time.monotonic()>=fit_deadline:
                        complete=False
                        expired+=1
                        continue
                    complete &= result.get('complete') is True
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
                        complete=selected and complete, reason="DIAGNOSTIC_ONLY" if complete else "NATIVE_SERVICE_INCOMPLETE"
                    )
                record["costs"]["diagnostic"] = dict(autonomous_discoveries=0,expired_fits=expired)
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
                count = attempts(evaluation_policy)[mode]
                if budget:
                    budget.consume(draws=count, attempts=count, rejections=count)
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
                    attempt_count=count,
                    template=mode == "template",
                    mode=mode,
                    snapshot=snapshot,
                    evidence=evidence,
                    evaluation_policy=evaluation_policy,
                    component_proposals=proposals,
                    **({"independent_case_clock": True} if inference_seed is not None and budget is None else {}),
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
            if case_boundary_dir is not None:
                atomic_write(boundary / "choice.json", seal(dict(prompt=digest(p), model=snapshot, policy=evaluation_policy,
                                                           clock=evidence.clocks[(p["id"], "correct")], record=record, **draw_identity)))
            if case_elapsed_us is not None:
                case_elapsed_us[p["id"]] = (time.monotonic_ns() - case_start + 999) // 1000
    finally:
        evidence.finish()
    if model_snapshot(encoder, decoder) != snapshot:
        raise RuntimeError("generator mutated frozen weights")
    if draw_identity:
        evidence.clock_summary = seal(dict(evidence.clock_summary, **draw_identity))
    if measurements is not None:
        publish_started = time.monotonic_ns()
    result = seal(
        {
            "version": "discovery-candidates/v1",
            "model": snapshot,
            "checkpoint": checkpoint["sha256"] if checkpoint else None,
            "model_control": model_control,
            "recipe": digest(recipe),
            "methods": list(methods),
            "prompts": digest(sorted(prompts, key=lambda p: p["id"])),
            "records": rows,
            "clock": evidence.clock_summary,
            "counts": counts,
            "elapsed_us": int((time.monotonic() - began) * 1e6),
        }
    )
    atomic_write(output / "candidates.json", result)
    ref = reference(output_root, output / "candidates.json")
    if measurements is not None:
        measurements["publish_candidates_us"] = (time.monotonic_ns() - publish_started) // 1000
    return ref


def score(root, candidates, truth, *, client, output, seconds=3600, budget=None, selected_only=False,
          measurements=None, independent_case_clock=False, registered_probe=None):
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
    if type(seconds) not in (int, float) or not 0 < seconds <= _maximum_seconds(budget):
        raise ValueError("bounded scorer required")
    if independent_case_clock and (budget is not None or not selected_only):
        raise ValueError("independent scoring clocks require a supervised single-choice roster")
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
    if measurements is not None:
        scoring_started = time.monotonic_ns()
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
    if selected_only:
        methods = ("equal_budget",)
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
        scored = canonical_score(projected, by_id, client, deadline=deadline,
                                 independent_case_clock=independent_case_clock)
        for key, row in scored["outcomes"].items():
            v = row["equal_budget"]
            upstream = value["records"][key]["arm_evidence"][method]
            v["complete"] = v["complete"] and upstream.get("complete") is True
            v["outcome"] = (
                "success" if v["success"] else "failure" if v["complete"] else "unresolved"
            )
            if registered_probe is not None:
                from oeis_learn.synthesis.admission import reference_identity

                choice = value["records"][key]["selections"][method]
                selected_exact = v["success"] if choice is not None and v["scoring"]["complete"] else None
                choice_ref = digest(choice) if choice is not None else None
                scorer_ref = digest(["round029-selected-20-80/v1", reference_identity(),
                                     getattr(getattr(client, "limits", None), "identity", None),
                                     client.runtime_identity("bigint-body/v1") if hasattr(client, "runtime_identity") else None])
                truth_ref = digest(by_id[key])
                observation = velocity_outcome(selected_exact=selected_exact, closed=upstream["complete"],
                                              complete=v["complete"])
                # Identical sealed choices at a genuinely shared inference opportunity can be reused.
                inference = probe_inference(registered_probe)
                evidence = dict(outcome_ref=digest([key, inference, choice_ref, scorer_ref, truth_ref,
                                                    observation]), choice_ref=choice_ref,
                                scorer_ref=scorer_ref, truth_ref=truth_ref)
                from oeis_learn.generalization.progressive.velocity_metrics import case_outcome_receipt

                v.update(observation, probe_ref=registered_probe["identity"],
                         receipt=case_outcome_receipt(key, status=observation["status"], **evidence))
                v["outcome"] = {"solve": "success", "miss": "failure", "unresolved": "unresolved"}[v["status"]]
            outcomes[key][method] = v
    if measurements is not None:
        publish_started = time.monotonic_ns()
        measurements["score_choices_us"] = (publish_started - scoring_started) // 1000
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
    if measurements is not None:
        measurements["publish_scores_us"] = (time.monotonic_ns() - publish_started) // 1000
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


def score_velocity(root, candidates, truth, *, client, output, probe, seconds=3600):
    """The scored choice must have been issued at this look, mode and frozen opportunity."""
    from .progressive_research_inputs import checked_file
    from oeis_learn.generalization.progressive.campaign import velocity_policy

    generated = load(checked_file(root, candidates), maximum=candidates["bytes"])
    clock = generated["clock"]
    method = probe["look"]["method"]
    control = method in velocity_policy()["controls"]
    policy = (
        "round029-conventional-full/v1"
        if control
        else DESCRIPTIVE if probe["look"]["profile"] == "common" else CORRECT_FULL
    )
    if (
        clock != seal(clock)
        or clock.get("velocity_probe") != probe["identity"]
        or clock.get("inference_seed") != probe["inference_seed"]
        or clock["policy"] != policy
        or generated["methods"] != ([method] if control else ["correct"])
    ):
        raise ValueError("sealed choices come from another probe profile or conditioning")
    truth = list(truth)
    by_id = {row["id"]: row for row in truth}
    if len(by_id) != len(truth) or set(by_id) != set(probe["donors"]):
        raise ValueError("original recipient truth roster required")
    prompts = [
        dict(
            id=m["task"],
            indices=by_id[m["task"]]["indices"][:20],
            values=by_id[m["task"]]["values"][:20],
        )
        for m in probe["look"]["members"]
    ]
    if (
        velocity_probe(
            probe["look"], prompts, mode=probe["mode"], inference_seed=probe["inference_seed"], model=probe.get("model")
        )
        != probe
    ):
        raise ValueError("scoring truth differs from frozen recipient prompt")
    if not control:
        model = checked_probe_model(probe["look"], probe.get("model"))
        if (generated["model"] != model["weights"] or generated["checkpoint"] != model["checkpoint"]
                or generated["recipe"] != model["profile"] or generated["model_control"] != model_control_for(probe["look"])):
            raise ValueError("sealed choices were generated by a different model")
    return score(
        root,
        candidates,
        truth,
        client=client,
        output=output,
        seconds=seconds,
        selected_only=True,
        independent_case_clock=True,
        registered_probe=probe,
    )


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


def _maximum_seconds(budget):
    from oeis_learn.generalization.progressive.budgets import RoleWorkBudget
    return budget.evaluation_seconds if isinstance(budget,RoleWorkBudget) else 21600


def velocity_outcome(*, selected_exact, closed, complete, evidence=None):
    """Private selected-choice outcome independent of other unfinished portfolio work."""
    if (type(complete) is not bool or type(closed) is not bool or
            selected_exact is not None and type(selected_exact) is not bool):
        raise ValueError("verified selected result and explicit closure flags required")
    if selected_exact is True:
        status = "solve"
    elif selected_exact is False or closed and complete:
        status = "miss"
    else:
        status = "unresolved"
    observation = dict(status=status, complete=complete, closed=closed)
    if evidence is not None:
        from oeis_learn.generalization.progressive.velocity_metrics import case_outcome_receipt

        observation["receipt"] = case_outcome_receipt(status=status, **evidence)
    return observation
