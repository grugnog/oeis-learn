"""Installed closed discovery operations; never accepts user Python or callbacks."""

import json
import sys
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import atomic_write, load, reference

from .progressive_evidence import failure, publish
from .progressive_roles import IMPLEMENTED, REGISTRY


def execute(operation, *, inputs=Path("/inputs"), output=Path("/output/discovery")):
    if operation not in IMPLEMENTED:
        raise ValueError("unsupported discovery operation")
    spec = REGISTRY[operation]
    values = []
    for channel in spec.channels:
        folder = inputs / channel
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError("required installed channel absent")
        for path in sorted(folder.glob("*.json")):
            import re

            if re.fullmatch(r"[0-9a-f]{64}\.json", path.name) is None:
                continue
            value = load(path, maximum=512 << 20)
            if value.get("version") != "discovery-channel/v1" or value.get("role") not in {
                spec.role,
                "public",
            }:
                raise ValueError("installed role/type mismatch")
            from oeis_learn.pipeline_validation.identity import seal

            if seal(value) != value:
                raise ValueError("installed channel identity mismatch")
            values.append(value)
    if operation == "propose-trial":
        if len(values) != 1 or values[0]["kind"] != "search_space":
            raise ValueError("one closed scalar-history proposal request required")
        from oeis_learn.generalization.progressive.tuning import propose_request
        proposal = propose_request(values[0]["data"], output / "journal")
        atomic_write(output / "proposal.json", proposal)
        return publish(output, operation=operation, mechanical_status="COMPLETE",
                       reason="PROPOSAL_DURABLE_BEFORE_DISPATCH", metrics={"proposal": proposal["identity"]})
    if operation == "prepare-stream":
        policies = [v["data"] for v in values if v["kind"] == "generation_policy"]
        if len(policies) != 1 or any(v["kind"] != "generation_policy" for v in values):
            raise ValueError("one frozen structural stream law required")
        from oeis_learn.pipeline_validation.identity import digest
        from oeis_learn.synthesis.client import SynthesisClient

        from .progressive_preparation import prepare_stream

        with SynthesisClient(digest(policies[0])) as client:
            import os

            shards = prepare_stream(
                output / "stream",
                policies[0],
                client=client,
                purpose=os.environ.get("OEIS_DISCOVERY_PURPOSE", "acceptance"),
            )
        complete = bool(shards) and all(s["payload"]["native_verified"] for s in shards)
        return publish(
            output,
            operation=operation,
            mechanical_status="COMPLETE" if complete else "PARTIAL",
            reason="SEALED_STRUCTURAL_STREAM",
            metrics={"shards": [s["identity"] for s in shards]},
        )
    if operation == "generate":
        if any(v["kind"] not in {"checkpoint", "recipe", "visible_prompt"} for v in values):
            raise ValueError("closed generation types required")
        recipes = [v["data"] for v in values if v["kind"] == "recipe"]
        checkpoints = [v["data"] for v in values if v["kind"] == "checkpoint"]
        prompts = [v["data"] for v in values if v["kind"] == "visible_prompt"]
        if (
            len(recipes) != 1
            or len(checkpoints) != 1
            or recipes[0]["compatibility"] != checkpoints[0]["compatibility"]
        ):
            raise ValueError("one compatible recipe/checkpoint required")
        from oeis_learn.generalization.progressive.contracts import verify_refs
        from oeis_learn.pipeline_validation.identity import confined, digest, seal
        from oeis_learn.synthesis.client import SynthesisClient

        from .progressive_evaluation import generate

        model_root = inputs / "model"
        verify_refs(
            model_root, [recipes[0]["artifact"], checkpoints[0]["artifact"]], max_bytes=64 << 30
        )
        recipe = load(confined(model_root, recipes[0]["artifact"]["path"]))
        from oeis_learn.fresh.bigint_config import resolve

        if resolve(recipe) != recipe:
            raise ValueError("resolved installed recipe required")
        control = checkpoints[0].get("control", "retained_D")
        with SynthesisClient(digest(recipe)) as client:
            artifact = generate(
                model_root,
                "generation",
                prompts,
                recipe,
                client=client,
                output_root=output,
                checkpoint=None if control == "random" else checkpoints[0]["artifact"],
                model_control=control,
                reference_recipe=recipe if control == "earlier_checkpoint" else None,
                seed=recipe["model"]["seed"] if control == "random" else None,
                device="cuda",
            )
        return publish(
            output,
            operation=operation,
            mechanical_status="COMPLETE",
            reason="SEALED_VISIBLE_CANDIDATES",
            metrics={"candidates": artifact},
        )
    if operation == "score":
        if any(
            v["kind"] not in {"candidate", "development_truth", "confirmation_truth"}
            for v in values
        ):
            raise ValueError("closed scorer inputs required")
        candidates = [v["data"] for v in values if v["kind"] == "candidate"]
        truth = [
            v["data"] for v in values if v["kind"] in {"development_truth", "confirmation_truth"}
        ]
        if len(candidates) != 1:
            raise ValueError("one frozen candidate collection required")
        from oeis_learn.pipeline_validation.identity import digest
        from oeis_learn.synthesis.client import SynthesisClient

        from .progressive_evaluation import score

        atomic_write(output / "input-candidates.json", candidates[0])
        with SynthesisClient(digest(candidates[0])) as client:
            result = score(
                output,
                reference(output, output / "input-candidates.json"),
                truth,
                client=client,
                output="scoring",
            )
        return publish(
            output,
            operation=operation,
            mechanical_status="COMPLETE",
            reason="FINITE_SCORING_COMPLETE",
            metrics={"scores": result["identity"]},
        )
    if operation == "census":
        cohorts = [v["data"] for v in values if v["kind"] == "cohort"]
        exposures = [v["data"] for v in values if v["kind"] == "exposure"]
        if len(cohorts) != 1 or len(exposures) != 1:
            raise ValueError("one closed cohort and exposure cut required")
        from .progressive_preparation import census

        result = census(cohorts[0], exposure=exposures[0])
        atomic_write(output / "census.json", result)
        return publish(
            output,
            operation=operation,
            mechanical_status="COMPLETE" if result["status"] == "COMPLETE" else "PARTIAL",
            scientific_status="UNMEASURED",
            reason="OUTCOME_BLIND_CENSUS",
            missing=[
                s for s, v in result["strata"].items() if v["task_deficit"] or v["group_deficit"]
            ],
        )
    if operation == "audit-language":
        witnesses = [v["data"] for v in values if v["kind"] == "coverage_witness"]
        if not witnesses:
            raise ValueError("constructive witness request required")
        from oeis_learn.generalization.progressive.coverage import audit
        from oeis_learn.pipeline_validation.identity import digest
        from oeis_learn.synthesis.client import SynthesisClient

        results = []
        private = output / "private"
        private.mkdir(exist_ok=True)
        private.chmod(0o700)
        with SynthesisClient(digest("discovery-coverage/v1")) as client:
            for ordinal, value in enumerate(witnesses):
                witness = {k: value[k] for k in ("source", "indices", "values")}
                path = private / (str(ordinal) + ".json")
                atomic_write(path, witness)
                result = audit(
                    output,
                    value["task"],
                    reference(output, path),
                    client=client,
                    operator_class=value["operator_class"],
                    missing_operations=value["missing_operations"],
                    created_at=value["created_at"],
                )
                # Public summary excludes witness bytes and local path.
                results.append({k: v for k, v in result["payload"].items() if k != "witness"})
        atomic_write(output / "coverage.json", results)
        return publish(
            output, operation=operation, mechanical_status="COMPLETE", reason="FINITE_WITNESS_AUDIT"
        )
    if operation == "publish":
        if any(v["kind"] != "public_evidence" for v in values):
            raise ValueError("public-only publication")
        # Initial collection only, not T080 independent bundle verification.
        atomic_write(output / "inventory.json", [v["identity"] for v in values])
        return publish(
            output,
            operation=operation,
            mechanical_status="PARTIAL",
            reason="INITIAL_COLLECTION_ONLY",
            missing=["independent_reconciliation"],
        )
    raise ValueError("unimplemented discovery role")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in IMPLEMENTED:
        raise ValueError("registered installed operation required")
    output = Path("/output/discovery")
    try:
        result = execute(args[0], output=output)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        # Private payload values must never appear in an exported exception.
        values = []
        for path in Path("/inputs").rglob("*.json"):
            try:
                value = load(path)

                def strings(v):
                    if isinstance(v, dict):
                        for key, child in v.items():
                            if key in {"source", "values", "indices", "witness"}:
                                if isinstance(child, str):
                                    values.append(child)
                                elif isinstance(child, list):
                                    values.extend(x for x in child if isinstance(x, str))
                            else:
                                strings(child)
                    elif isinstance(v, list):
                        for child in v:
                            strings(child)

                strings(value)
            except (ValueError, OSError):
                pass
        result = failure(
            output, args[0], exc, producer="installed-" + args[0], private_values=values
        )
    print(json.dumps(result, sort_keys=True))
    return (
        0
        if result["mechanical_status"] == "COMPLETE"
        else 2
        if result["mechanical_status"] in {"PARTIAL", "BLOCKED"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
