"""Run the frozen visible-only conventional comparators at the round029 full endpoint."""

import time
from pathlib import Path

from oeis_learn.decoder.bigint_codec import fitting_policy
from oeis_learn.fresh.bigint_diagnostics import _seal
from oeis_learn.fresh.template_baseline import template_bank
from oeis_learn.generalization.progressive.baselines import Portfolio, iter_candidates
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, reference, seal
from oeis_learn.synthesis.admission import VisiblePrompt

from .identity import confined


def generate_control(
    root,
    output,
    prompts,
    recipe,
    *,
    client,
    probe,
    seconds,
    budget,
    case_elapsed_us,
    case_boundary_dir,
    measurements,
    recovered=None,
    recover_control=None,
):
    began_ns = time.monotonic_ns()
    look = probe["look"]
    method = look["method"]
    rule = "templates" if method == "template" else method
    policy = recipe.get("base_recipe", recipe).get("pipeline")
    selection = "visible-only/v2" if policy else "visible-only/v1"
    snapshot = digest(
        [
            method,
            Portfolio().report()["conventional_identity"],
            template_bank()["sha256"] if rule == "templates" else None,
        ]
    )
    output = confined(root, output, exists=False)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = min(start + seconds, budget.deadline) if budget else start + seconds
    records, clocks = {}, []
    if measurements is not None:
        measurements["prepare_control_us"] = (time.monotonic_ns() - began_ns) // 1000
    recovered = recovered or {}
    if not set(recovered) <= {p["id"] for p in prompts}:
        raise ValueError("unknown recovered control recipient")
    for prompt in prompts:
        if time.monotonic() >= deadline:
            raise TimeoutError("conventional search stopped before opening the next portfolio")
        if budget:
            budget.consume(
                draws=look["attempts"], attempts=look["attempts"], rejections=look["attempts"]
            )
        account = Portfolio()
        visible = VisiblePrompt(tuple(prompt["indices"]), tuple(prompt["values"]))
        binding = dict(prompt=digest(prompt), model=snapshot, velocity_probe=probe["identity"])
        if prompt["id"] in recovered:
            # Supplied by the bound evaluation adapter for historical installed wheels.
            row, clock = recover_control(recovered[prompt["id"]], binding)
            records[prompt["id"]] = row
            if clock is not None:
                clocks.append(clock)
            continue
        if case_boundary_dir is not None:
            boundary = Path(case_boundary_dir) / digest(prompt["id"])[7:]
            boundary.mkdir(parents=True, exist_ok=False)
            atomic_write(boundary / "start.json", seal(binding))
        admitted = []
        reasons = []
        finished_fits = applicable = 0
        for ordinal, row in enumerate(
            iter_candidates(rule, prompt, account, deadline=deadline if budget else None)
        ):
            if not row["applicable"]:
                reasons.append(row["reason"])
                continue
            applicable += 1
            began = time.monotonic()
            deadline_fit = min(row["fit_deadline"], deadline) if budget else row["fit_deadline"]
            job = digest([probe["identity"], prompt["id"], ordinal, row["source"]])
            if case_boundary_dir is not None:
                atomic_write(boundary / f"issued-{ordinal:04d}.json",
                             seal(dict(job=job, ordinal=ordinal, elapsed_us=int((began-account.started)*1e6))))
            try:
                if began >= deadline_fit:
                    raise TimeoutError
                fit = client.fit(
                    row["source"],
                    visible,
                    policy=fitting_policy(row["source"]),
                    deadline=deadline_fit,
                    seed=int(job[7:15], 16),
                )
            except TimeoutError:
                fit = dict(outcome="resource_exhausted", complete=False, admitted=[])
            end = time.monotonic()
            timely = end < deadline_fit
            account.observe(
                rule,
                candidate=digest(row["source"]),
                started=began,
                ended=end,
                outcome=fit["outcome"] if timely else "resource_exhausted",
                phase="fit",
            )
            if timely:
                finished_fits += int(fit.get("complete") is True)
                admitted.extend(
                    dict(
                        item,
                        _original_source=row["source"],
                        _cost=end - began,
                        _attempt_ordinal=ordinal,
                        _job_id=job,
                    )
                    for item in fit["admitted"]
                )
        elapsed = time.monotonic() - account.started
        complete = (
            applicable > 0
            and applicable == finished_fits
            and not reasons
            and (rule != "templates" or account.attempts == look["attempts"])
        )
        reason = (
            "complete" if complete else reasons[0] if reasons and not applicable else "incomplete"
        )
        choice = _seal(
            admitted, visible, cost=elapsed, selection=selection, snapshot=snapshot, policies=policy
        )
        report = account.report()
        records[prompt["id"]] = dict(
            selections={"equal_budget": choice},
            arm_evidence={"equal_budget": dict(complete=complete, reason=reason)},
            costs={method: report},
        )
        wall = report["total_elapsed_us"]
        clock_complete = (
            account.attempts == (look["attempts"] if rule == "templates" else 1)
            or time.monotonic() >= account.deadline
        )
        clocks.append(
            dict(
                prompt=prompt["id"],
                allowed=look["attempts"],
                started=account.attempts,
                unissued=look["attempts"] - account.attempts,
                applicable=applicable,
                completed_fits=finished_fits,
                cost_complete=clock_complete,
                elapsed_us=wall,
            )
        )
        if case_boundary_dir is not None:
            atomic_write(
                boundary / "choice.json",
                seal(dict(binding, clock=clocks[-1], record=records[prompt["id"]])),
            )
        if case_elapsed_us is not None:
            case_elapsed_us[prompt["id"]] = wall
    publish_started = time.monotonic_ns()
    generated = seal(
        dict(
            version="discovery-candidates/v1",
            model=snapshot,
            model_control=method,
            checkpoint=None,
            recipe=digest(recipe),
            methods=[method],
            prompts=digest(sorted(prompts, key=lambda p: p["id"])),
            records=records,
            counts=dict(cases=len(prompts)),
            clock=seal(
                dict(
                    policy="round029-conventional-full/v1",
                    velocity_probe=probe["identity"],
                    cost_complete=len(clocks) == len(prompts) and all(c["cost_complete"] for c in clocks),
                    inference_seed=probe["inference_seed"],
                    clocks=clocks,
                )
            ),
            elapsed_us=int((time.monotonic() - start) * 1_000_000),
        )
    )
    path = output / "candidates.json"
    atomic_write(path, generated)
    ref = reference(root, path)
    if measurements is not None:
        measurements["publish_candidates_us"] = (time.monotonic_ns() - publish_started) // 1000
    return ref
