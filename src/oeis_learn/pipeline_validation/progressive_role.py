"""Installed closed discovery operations; never accepts user Python or callbacks."""

import json
import sys
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import atomic_write, load, reference

from .progressive_evidence import failure, publish
from .progressive_roles import IMPLEMENTED, REGISTRY


def _execute(
    operation,
    *,
    inputs=Path("/inputs"),
    output=Path("/output/discovery"),
    authority=Path("/authority/supervisor.json"),
    accounting=Path("/accounting"),
    control=Path("/control"),
    state_root=Path("/learning-state"),
    predecessor_state=Path("/predecessor-state"),
    component_state=Path("/component-state"),
):
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
    from oeis_learn.generalization.progressive.budgets import WORK_ROLES, RoleWorkBudget

    budget = (
        RoleWorkBudget(load(authority), operation, values, output)
        if operation in WORK_ROLES
        else None
    )
    if operation == "train":
        if len(values) != 1 or values[0]["kind"] != "training_request":
            raise ValueError("one frozen training-only request required")
        from oeis_learn.generalization.progressive.contracts import verify_refs
        from oeis_learn.generalization.progressive.projections import _validate_channel
        from oeis_learn.pipeline_validation.identity import confined

        from .progressive_learner import compatibility, run

        data = values[0]["data"]
        _validate_channel("training_request", data)
        verify_refs(inputs / "training", [data["artifact"]], max_bytes=256 << 20)
        package = load(confined(inputs / "training", data["artifact"]["path"]), maximum=256 << 20)
        if data["compatibility"] != compatibility(package):
            raise ValueError("typed training compatibility differs")
        return run(
            package,
            load(authority),
            accounting=accounting,
            control=control,
            output=output,
            state_root=state_root,
            predecessor_state=predecessor_state,
            initializer_root=inputs / "training",
        )
    if operation == "admit-training":
        if any(v["kind"] not in {"candidate", "training_continuation"} for v in values):
            raise ValueError("admission role received excluded truth")
        candidates = [v["data"] for v in values if v["kind"] == "candidate"]
        members = [v["data"] for v in values if v["kind"] == "training_continuation"]
        if (
            len(candidates) != 1
            or len(members) != 1
            or set(candidates[0]) != {"task", "source"}
            or candidates[0]["task"] != members[0]["id"]
        ):
            raise ValueError("one bound concrete candidate/training member required")
        import os

        from oeis_learn.generalization.progressive.admission import admit
        from oeis_learn.pipeline_validation.identity import digest
        from oeis_learn.synthesis.client import SynthesisClient

        with SynthesisClient(digest(members[0])) as client:
            budget.client(client)
            admission = admit(
                members[0],
                candidates[0]["source"],
                client=client,
                purpose=os.environ.get("OEIS_DISCOVERY_PURPOSE", "acceptance"),
                budget=budget,
            )
        atomic_write(output / "admission.json", admission)
        return publish(
            output,
            operation=operation,
            mechanical_status="COMPLETE",
            reason="FINITE_TRAINING_ADMISSION",
            metrics={"admission": admission["identity"]},
        )
    if operation in {"extract-components", "verify-components"}:
        from oeis_learn.generalization.progressive.component_mining import LibraryStore
        from oeis_learn.generalization.progressive.contracts import verify_refs
        import time
        from oeis_learn.generalization.progressive.projections import _validate_channel
        from oeis_learn.pipeline_validation.identity import confined, digest
        from oeis_learn.synthesis.client import SynthesisClient

        requests = [v["data"] for v in values if v["kind"] == "component_policy"]
        if len(requests) != 1 or any(
            v["kind"]
            not in {"component_policy", "typed_training_source", "verified_training_archive"}
            for v in values
        ):
            raise ValueError("training-only component source channels required")
        request = requests[0]
        _validate_channel("component_policy", request)
        root = inputs / "archive"
        refs = []
        for value in values:
            if value["kind"] == "component_policy":
                continue
            _validate_channel(value["kind"], value["data"])
            original = value["data"]["artifact"]
            verify_refs(root, [original], max_bytes=256 << 20)
            if value["kind"] == "typed_training_source":
                refs.append(original)
            else:
                from oeis_learn.generalization.progressive.contracts import validate

                cut = validate("archive", load(confined(root, original["path"])))
                base = confined(root, original["path"]).parent
                for child in cut["payload"]["entries"]:
                    verify_refs(base, [child], max_bytes=4 << 20)
                    refs.append(reference(root, confined(base, child["path"])))
        refs = list({r["path"]: r for r in refs}.values())
        authority_value = load(authority)
        store = LibraryStore(
            root,
            None,
            store_root=component_state,
            purpose=authority_value["task"]["purpose"],
            track=authority_value["task"]["track"],
            selected=request["policy"],
            generation_policies=request.get("generation_policies", {}),
            max_bytes=min(512 << 20, authority_value["task"]["payload"]["caps"]["archive_bytes"]),
            max_inodes=authority_value.get("storage", {}).get(
                "max_inodes", authority_value["campaign"]["payload"]["storage"]["required_inodes"]
            ),
        )
        previous = None
        if request["previous"] is not None:
            found = [v for v in store.committed() if v["identity"] == request["previous"]]
            if len(found) != 1:
                raise ValueError("original committed parent library required")
            previous = found[0]
        with SynthesisClient(digest(request)) as client:
            budget.client(client)
            library = store.publish(
                refs,
                client=client,
                previous=previous,
                seconds=min(21600, max(0.001, budget.deadline - time.monotonic())),
                budget=budget,
            )
        atomic_write(output / "library.json", library)
        from oeis_learn.generalization.progressive.component_search import search_input

        atomic_write(output / "search-library.json", search_input(store, library))
        return publish(
            output,
            operation=operation,
            mechanical_status="COMPLETE",
            reason="VERIFIED_COMPONENT_LIBRARY"
            if library["payload"]["components"]
            else "NO_SUPPORTED_COMPONENT",
            metrics=dict(
                library=library["identity"],
                components=len(library["payload"]["components"]),
                scientific_status="UNMEASURED",
                charged_mining_work=store.spent(),
                native_requests=budget.used["native_work"],
            ),
        )
    if operation == "propose-trial":
        if len(values) != 1 or values[0]["kind"] != "search_space":
            raise ValueError("one closed scalar-history proposal request required")
        from oeis_learn.generalization.progressive.tuning import propose_request

        proposal = propose_request(values[0]["data"], output / "journal")
        atomic_write(output / "proposal.json", proposal)
        return publish(
            output,
            operation=operation,
            mechanical_status="COMPLETE",
            reason="PROPOSAL_DURABLE_BEFORE_DISPATCH",
            metrics={"proposal": proposal["identity"]},
        )
    if operation == "prepare-stream":
        policies = [v["data"] for v in values if v["kind"] == "generation_policy"]
        if len(policies) != 1 or any(v["kind"] != "generation_policy" for v in values):
            raise ValueError("one frozen structural stream law required")
        from oeis_learn.pipeline_validation.identity import digest
        from oeis_learn.synthesis.client import SynthesisClient

        from .progressive_preparation import prepare_stream

        with SynthesisClient(digest(policies[0])) as client:
            budget.client(client)
            import os

            shards = prepare_stream(
                output / "stream",
                policies[0],
                client=client,
                purpose=os.environ.get("OEIS_DISCOVERY_PURPOSE", "acceptance"),
                budget=budget,
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
        if any(
            v["kind"]
            not in {"checkpoint", "recipe", "visible_prompt", "oeis_generation", "library"}
            for v in values
        ):
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

        if recipe.get("version") == "progressive-research-model/v1":
            from oeis_learn.generalization.progressive.models import validate_profile

            validate_profile(recipe)
        elif resolve(recipe) != recipe:
            raise ValueError("resolved installed recipe required")
        generation_policy = [v["data"] for v in values if v["kind"] == "oeis_generation"]
        if len(generation_policy) > 1:
            raise ValueError("one OEIS generation policy required")
        mode_policy = generation_policy[0] if generation_policy else None
        if mode_policy:
            from oeis_learn.generalization.progressive.projections import _validate_channel

            _validate_channel("oeis_generation", mode_policy)
            if (
                mode_policy["forced"] is not None
                and load(authority)["task"]["purpose"] != "acceptance"
            ):
                raise ValueError("forced candidates are DIAGNOSTIC_ONLY")
        libraries = [v["data"] for v in values if v["kind"] == "library"]
        if len(libraries) > 1:
            raise ValueError("one frozen component library required")
        component_library = None
        if libraries:
            from oeis_learn.generalization.progressive.component_search import validate_input

            if (
                libraries[0]["compatibility"] != recipes[0]["compatibility"]
                or libraries[0]["profile"] != "progressive-component-search/v1"
            ):
                raise ValueError("compatible committed component search input required")
            verify_refs(model_root, [libraries[0]["artifact"]], max_bytes=256 << 20)
            component_library = validate_input(
                load(confined(model_root, libraries[0]["artifact"]["path"]), maximum=256 << 20)
            )
            if (
                load(authority)["task"]["purpose"] == "research"
                and component_library["library"]["purpose"] != "research"
            ):
                raise ValueError("research requires a native research library")
            if mode_policy is not None:
                raise ValueError("component and OEIS policies cannot be combined")
        control = checkpoints[0].get("control", "retained_D")
        with SynthesisClient(digest(recipe)) as client:
            budget.client(client)
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
                device="cpu" if load(authority)["task"]["purpose"] == "acceptance" else "cuda",
                budget=budget,
                component_library=component_library,
                **(
                    {"methods": tuple(mode_policy["methods"]), "forced": mode_policy["forced"]}
                    if mode_policy
                    else {"methods": ("correct",)}
                    if component_library
                    else {}
                ),
            )
        from .progressive_evidence import generation_timing

        timing = generation_timing(load(confined(output, artifact["path"]), maximum=512 << 20))
        atomic_write(output / "timing.json", timing)
        return publish(
            output,
            operation=operation,
            mechanical_status="COMPLETE",
            reason="SEALED_VISIBLE_CANDIDATES",
            metrics={"candidates": artifact, "timing": reference(output, output / "timing.json")},
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
            budget.client(client)
            result = score(
                output,
                reference(output, output / "input-candidates.json"),
                truth,
                client=client,
                output="scoring",
                budget=budget,
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
        if len(values) == 1 and values[0]["data"].get("version") == "progressive-decision-input/v1":
            from oeis_learn.generalization.progressive.decisions import publish as decide
            from oeis_learn.pipeline_validation.identity import seal

            data = values[0]["data"]
            if set(data) != {
                "version",
                "source_slot",
                "target_slot",
                "rows",
                "roster",
                "seeds",
                "design",
                "policy",
                "remaining",
                "kwargs",
            }:
                raise ValueError("closed decision producer request required")
            if set(data["kwargs"]) - {"target_id", "action", "trigger", "costs", "confirmation"}:
                raise ValueError("closed comparison arguments required")
            policy = data["policy"]
            decision = decide(
                data["rows"],
                data["roster"],
                data["seeds"],
                data["design"],
                policy=policy,
                remaining=data["remaining"],
                created_at=policy["created_at"],
                purpose=policy["purpose"],
                track=policy["track"],
                **data["kwargs"],
            )
            bundle = seal(
                {
                    "version": "progressive-decision-bundle/v1",
                    "source_slot": data["source_slot"],
                    "target_slot": data["target_slot"],
                    "decision": decision,
                    "rows": data["rows"],
                    "roster": data["roster"],
                    "seeds": data["seeds"],
                    "design": data["design"],
                    "kwargs": data["kwargs"],
                }
            )
            atomic_write(output / "decision-bundle.json", bundle)
            return publish(
                output,
                operation=operation,
                mechanical_status="COMPLETE",
                scientific_status=decision["payload"]["evidence_verdict"],
                next_action=decision["payload"]["next_action"],
                target_policy=policy["identity"],
                reason="RECOMPUTED_REGISTERED_DECISION",
                metrics={"decision_bundle": reference(output, output / "decision-bundle.json")},
            )
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


def execute(operation, **kwargs):
    from oeis_learn.generalization.progressive.budgets import WORK_ROLES
    from oeis_learn.generalization.progressive.storage import owned_guard

    if operation in WORK_ROLES:
        authority = load(kwargs.get("authority", Path("/authority/supervisor.json")))
        with owned_guard(
            authority,
            output=kwargs.get("output", Path("/output/discovery")),
            state_root=kwargs.get("component_state", Path("/component-state"))
            if operation in {"extract-components", "verify-components"}
            else None,
        ).activate():
            return _execute(operation, **kwargs)
    return _execute(operation, **kwargs)


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
        from contextlib import nullcontext
        from oeis_learn.generalization.progressive.storage import owned_guard

        authority_path = Path("/authority/supervisor.json")
        guard = (
            owned_guard(
                load(authority_path),
                output=output,
                state_root=Path("/learning-state") if args[0] == "train" else None,
            )
            if authority_path.is_file()
            else None
        )
        with guard.activate() if guard else nullcontext():
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
