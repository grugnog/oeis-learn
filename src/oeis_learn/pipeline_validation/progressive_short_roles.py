"""Installed four-visible/nine-total roles with separate truth projections."""

import json
import time

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
    verify_refs,
)
from oeis_learn.generalization.progressive.short_profile import (
    admit,
    collate,
    validate_terms,
)

from .progressive_evidence import publish


def validate_channel(kind, data):
    if kind in {"short_prompt", "short_training_truth", "short_score_truth"} and (
        type(data.get("id")) is not str or not 1 <= len(data["id"]) <= 256
    ):
        raise ValueError("bounded short member identity required")
    if kind == "short_prompt":
        if set(data) != {"id", "indices", "values"}:
            raise ValueError("closed short prompt required")
        collate([{k: data[k] for k in ("indices", "values")}])
    elif kind in {"short_training_truth", "short_score_truth"}:
        if set(data) != {"id", "indices", "values"}:
            raise ValueError("closed short truth required")
        validate_terms({k: data[k] for k in ("indices", "values")})
    elif kind == "short_candidates":
        if (
            set(data) != {"version", "identity", "checkpoint", "model_control", "members"}
            or seal(data) != data
            or data["version"] != "progressive-short-candidates/v1"
        ):
            raise ValueError("closed short candidates required")
        if not isinstance(data["members"], dict) or len(data["members"]) > 128:
            raise ValueError("bounded short candidate roster required")
        for member, row in data["members"].items():
            if (
                type(member) is not str
                or len(member) > 256
                or set(row) != {"source", "complete"}
                or type(row["complete"]) is not bool
                or (
                    row["source"] is not None
                    and (type(row["source"]) is not str or len(row["source"]) > 65536)
                )
            ):
                raise ValueError("closed bounded short candidate required")
    elif kind == "short_policy":
        if (
            set(data) != {"seed", "max_tokens", "seconds", "forced"}
            or type(data["seed"]) is not int
            or not 0 <= data["seed"] < 2**63
            or type(data["max_tokens"]) is not int
            or not 1 <= data["max_tokens"] <= 1024
            or type(data["seconds"]) is not int
            or not 1 <= data["seconds"] <= 60
        ):
            raise ValueError("finite short generation policy required")
        if data["forced"] is not None and not isinstance(data["forced"], dict):
            raise ValueError("diagnostic short candidates required")
        if data["forced"] is not None and (
            len(data["forced"]) > 128
            or any(
                type(k) is not str or len(k) > 256 or type(v) is not str or len(v) > 65536
                for k, v in data["forced"].items()
            )
        ):
            raise ValueError("bounded diagnostic short candidate inventory required")


def execute(operation, inputs, output, values, authority, budget):
    from .progressive_device_acceptance import device

    task = load(authority, maximum=32 << 20)["task"]
    if task["track"] != "external_assisted":
        raise ValueError("separate assisted short profile required")
    if operation == "generate":
        if any(
            v["kind"] not in {"short_prompt", "short_policy", "checkpoint", "recipe"}
            for v in values
        ):
            raise ValueError("closed short generator channels required")
        prompts = [v["data"] for v in values if v["kind"] == "short_prompt"]
        policies = [v["data"] for v in values if v["kind"] == "short_policy"]
        recipes = [v["data"] for v in values if v["kind"] == "recipe"]
        checkpoints = [v["data"] for v in values if v["kind"] == "checkpoint"]
        if (
            not prompts
            or len(prompts) > 128
            or len(policies) != 1
            or len(recipes) != 1
            or len(checkpoints) != 1
        ):
            raise ValueError("one short generator model/policy required")
        policy = policies[0]
        validate_channel("short_policy", policy)
        if policy["forced"] is not None and task["purpose"] != "acceptance":
            raise ValueError("forced short candidates cannot authorize research")
        if recipes[0]["compatibility"] != checkpoints[0]["compatibility"]:
            raise ValueError("short model compatibility differs")
        root = inputs / "model"
        profile = json.loads(verify_refs(root, [recipes[0]["artifact"]])[0])
        from oeis_learn.generalization.progressive.baselines import load_baseline

        control = checkpoints[0].get("control", "retained_D")
        encoder, decoder, _ = load_baseline(
            control,
            profile,
            root=None if control == "random" else root,
            ref=None if control == "random" else checkpoints[0]["artifact"],
            seed=profile["model"]["seed"] if control == "random" else None,
            device=device(task["purpose"]),
        )
        from .progressive_evaluation import generate_short

        members = {}
        for start in range(0, len(prompts), 8):
            subset = prompts[start : start + 8]
            for p in subset:
                validate_channel("short_prompt", p)
            generated = generate_short(
                encoder,
                decoder,
                [{k: p[k] for k in ("indices", "values")} for p in subset],
                seed=policy["seed"] + start,
                max_tokens=policy["max_tokens"],
                seconds=min(policy["seconds"], max(0.001, budget.deadline - time.monotonic())),
            )
            for p, c in zip(subset, generated):
                budget.consume(draws=1, attempts=1, rejections=1)
                if policy["forced"] is not None:
                    source = policy["forced"].get(p["id"])
                    c = dict(c, source=source, complete=source is not None)
                members[p["id"]] = {
                    "source": c.get("source") if c.get("complete") else None,
                    "complete": bool(c.get("complete")),
                }
        data = seal(
            {
                "version": "progressive-short-candidates/v1",
                "checkpoint": None if control == "random" else checkpoints[0]["artifact"]["sha256"],
                "model_control": control,
                "members": members,
            }
        )
        atomic_write(output / "short-candidates.json", data)
        return publish(
            output,
            operation=operation,
            mechanical_status="COMPLETE",
            reason="FOUR_VISIBLE_GENERATION",
            metrics={"short_candidates": reference(output, output / "short-candidates.json")},
        )
    candidates = [v["data"] for v in values if v["kind"] == "short_candidates"]
    truth = [
        v["data"]
        for v in values
        if v["kind"]
        == ("short_training_truth" if operation == "admit-training" else "short_score_truth")
    ]
    if (
        len(candidates) != 1
        or not truth
        or any(
            v["kind"]
            not in {
                "short_candidates",
                "short_training_truth" if operation == "admit-training" else "short_score_truth",
            }
            for v in values
        )
    ):
        raise ValueError("isolated short candidate/truth channels required")
    data = candidates[0]
    validate_channel("short_candidates", data)
    from oeis_learn.synthesis.client import SynthesisClient

    from .progressive_evaluation import score_short

    targets = []
    scores = {}
    with SynthesisClient(digest(data)) as client:
        budget.client(client)
        for t in truth:
            kind = "short_training_truth" if operation == "admit-training" else "short_score_truth"
            validate_channel(kind, t)
            source = data["members"].get(t["id"], {}).get("source")
            if not source:
                scores[t["id"]] = {"outcome": "unresolved"}
                continue
            budget.consume(attempts=1, rejections=1)
            terms = {k: t[k] for k in ("indices", "values")}
            if operation == "admit-training":
                target = admit(
                    source, terms, member=t["id"], client=client, purpose=task["purpose"]
                )
                if target is not None:
                    targets.append(target)
            else:
                score = score_short(source, terms, client=client)
                scores[t["id"]] = {
                    "outcome": "success" if score["agreement"] else "failure",
                    "primary_scaling_eligible": False,
                }
    if operation == "admit-training":
        data = seal({"version": "progressive-short-admissions/v1", "targets": targets})
        atomic_write(output / "short-admissions.json", data)
        return publish(
            output,
            operation=operation,
            mechanical_status="COMPLETE",
            reason="SHORT_TARGETS_ADMITTED" if targets else "SOURCE_YIELD_DEFICIT",
            metrics={"short_admissions": reference(output, output / "short-admissions.json")},
        )
    atomic_write(
        output / "short-scores.json",
        seal(
            {
                "version": "progressive-short-scores/v1",
                "outcomes": scores,
                "primary_scaling_eligible": False,
            }
        ),
    )
    return publish(
        output,
        operation=operation,
        mechanical_status="COMPLETE",
        reason="FINITE_NINE_TERM_SCORING",
        metrics={"short_scores": reference(output, output / "short-scores.json")},
    )
