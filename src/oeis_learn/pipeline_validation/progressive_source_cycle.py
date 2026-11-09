"""Frozen assisted source cycle over isolated generation/admission/learning/scoring."""

from pathlib import Path

from oeis_learn.generalization.progressive.contracts import atomic_write, load, reference, seal
from oeis_learn.pipeline_validation.identity import confined

from .progressive_oeis import Cycle, _read


def validate_plan(root, campaign, plan):
    from oeis_learn.generalization.progressive.short_learning import RULE
    from oeis_learn.generalization.progressive.sources import pilot_support

    from .progressive_learner import compatibility, validate_input

    if (
        set(plan) != {"version", "identity", "roster", "training", "seconds", "generation"}
        or seal(plan) != plan
        or plan["version"] != "progressive-source-cycle/v1"
        or campaign["track"] != "external_assisted"
    ):
        raise ValueError("closed frozen assisted source cycle required")
    if plan["roster"] not in [r["artifact"] for r in campaign["payload"]["references"]]:
        raise ValueError("original source roster required")
    roster = _read(root, plan["roster"])
    if set(roster) != {"sources", "roles"}:
        raise ValueError("closed source roster required")
    support = pilot_support(roster["sources"], roster["roles"])
    if support["status"] != "ELIGIBLE":
        raise ValueError("source support deficit: " + str(support["deficits"]))
    if set(plan["seconds"]) != {"generate", "admit-training", "train", "score"} or any(
        type(v) is not int or not 1 <= v <= 600 for v in plan["seconds"].values()
    ):
        raise ValueError("finite short role allowances required")
    from .progressive_short_roles import validate_channel

    validate_channel("short_policy", plan["generation"])
    if campaign["purpose"] == "research" and plan["generation"]["forced"] is not None:
        raise ValueError("forced short fixtures cannot authorize research")
    if len(plan["training"]) != 4:
        raise ValueError("two paired short source seeds required")
    tasks = {p["id"]: p for p in campaign["payload"]["tasks"]}
    pairs = {}
    ids = set()
    slots = set()
    allowed = [r["artifact"] for r in campaign["payload"]["references"]]
    for entry in plan["training"]:
        if (
            set(entry) != {"task", "channel", "model"}
            or entry["task"] in ids
            or entry["task"] not in tasks
        ):
            raise ValueError("original unique short child task required")
        ids.add(entry["task"])
        p = tasks[entry["task"]]
        slots.add(p["slot"])
        if (
            p["kind"] != "training"
            or p["stage"] != "source_pilot"
            or p["arm"] not in {"with_source", "without_source"}
        ):
            raise ValueError("short source stage/arm required")
        if (
            reference(root, entry["channel"]) not in allowed
            or len(entry["model"]) != 2
            or any(reference(root, path) not in allowed for path in entry["model"])
        ):
            raise ValueError("original short learner/model channels required")
        channel = load(confined(root, entry["channel"]))
        package = _read(root, channel["data"]["artifact"])
        if (
            package.get("task_profile") != "combinatorial-short/v1"
            or package.get("short_rules") != RULE
        ):
            raise ValueError("frozen short learning package required")
        validate_input(
            package,
            {"purpose": campaign["purpose"], "track": campaign["track"], "payload": p},
            allow_pending=True,
        )
        if compatibility(package) != channel["data"]["compatibility"]:
            raise ValueError("short learner compatibility differs")
        models = [load(confined(root, path)) for path in entry["model"]]
        if {v["kind"] for v in models} != {"checkpoint", "recipe"}:
            raise ValueError("one identified short model required")
        recipe = next(v["data"] for v in models if v["kind"] == "recipe")
        checkpoint = next(v["data"] for v in models if v["kind"] == "checkpoint")
        if (
            _read(root, recipe["artifact"]) != package["profile"]
            or recipe["compatibility"] != compatibility(package)
            or checkpoint["compatibility"] != compatibility(package)
        ):
            raise ValueError("short generation must use actual learner model")
        if p["initializer"]["mode"] != "random" or checkpoint.get("control") != "random":
            raise ValueError("initial source pilot uses paired fresh seeds")
        pair = pairs.setdefault(p["seed"], {})
        if p["arm"] in pair:
            raise ValueError("duplicate source arm")
        pair[p["arm"]] = p
    if (
        len(slots) != 1
        or len(pairs) != 2
        or any(set(pair) != {"with_source", "without_source"} for pair in pairs.values())
    ):
        raise ValueError("paired short source tasks required")
    for pair in pairs.values():
        a, b = pair.values()
        if (
            a["initializer"] != b["initializer"]
            or a["token_target"] != b["token_target"]
            or a["caps"] != b["caps"]
        ):
            raise ValueError("matched short initialization/exposure required")
    training = [s for s, r in zip(roster["sources"], roster["roles"]) if r == "training"]
    evaluation = [s for s, r in zip(roster["sources"], roster["roles"]) if r == "evaluation"]
    if len(training) > 128 or len(evaluation) > 128:
        raise ValueError("finite source cycle supports at most 128 members per role")
    return training, evaluation


def validate_request(root, request, *, plans, refs, graph, purpose, track):
    from oeis_learn.generalization.progressive.budgets import vector

    if (
        set(request) != {"operation", "artifacts", "image", "seconds", "reservation"}
        or set(request["artifacts"]) != {"cycle"}
        or len(request["artifacts"]["cycle"]) != 1
    ):
        raise ValueError("closed source host request required")
    vector(request["reservation"])
    from oeis_learn.generalization.progressive.contracts import _check,schema
    _check(schema('common')['$defs']['hash'],request['image'],schema('common'))
    if (
        any(request["reservation"].values())
        or type(request["seconds"]) is not int
        or not 0 < request["seconds"] <= 86400
    ):
        raise ValueError("finite child-owned source reservation required")
    path = request["artifacts"]["cycle"][0]
    if reference(root, path) not in [r["artifact"] for r in refs]:
        raise ValueError("source cycle plan must be frozen")
    plan = load(confined(root, path))
    validate_plan(
        root,
        {
            "purpose": purpose,
            "track": track,
            "payload": {"tasks": plans, "references": refs, "graph": graph},
        },
        plan,
    )
    children = [p for p in plans if p["id"] in {e["task"] for e in plan["training"]}]
    policy = next(
        r["artifact"]["sha256"]
        for r in refs
        if r["kind"] == "policy" and _read(root, r["artifact"]) == request
    )
    if (
        any(p["policy"] != policy for p in children)
        or request["seconds"] * 1000000 > children[0]["caps"]["elapsed_us"]
    ):
        raise ValueError("source child authority differs")
    return plan


class SourceCycle(Cycle):
    def run(self):
        if (self.output / "result.json").exists():
            return load(self.output / "result.json")
        from oeis_learn.generalization.progressive.recovery import Recovery
        from oeis_learn.generalization.progressive.short_learning import admitted_rows

        from .progressive_evidence import publish
        from .progressive_learner import compatibility

        completed = []
        results = {}
        for entry in self.plan["training"]:
            task = self.tasks[entry["task"]]
            base_channel = load(confined(self.root, entry["channel"]))
            package = _read(self.root, base_channel["data"]["artifact"])
            accepted = []
            if task["payload"]["arm"] == "with_source":
                candidates = self.generate_sources(self.pool, entry["model"], task)
                for offset in range(0, len(self.pool), 128):
                    truth = [
                        self.channel(
                            "short_training_truth",
                            "admission",
                            {
                                "id": s["identity"],
                                "indices": list(map(str, range(9))),
                                "values": s["values"],
                            },
                        )
                        for s in self.pool[offset : offset + 128]
                    ]
                    _, target, _ = self.phase(
                        "admit-training",
                        {
                            "candidates": [
                                self.channel("short_candidates", "admission", candidates)
                            ],
                            "admission_truth": truth,
                        },
                        task,
                    )
                    accepted.extend(load(target / "role/short-admissions.json")["targets"])
                if not accepted:
                    return publish(
                        self.output,
                        operation="source-cycle",
                        campaign=self.campaign["identity"],
                        mechanical_status="COMPLETE",
                        scientific_status="INCOMPLETE",
                        reason="SOURCE_YIELD_DEFICIT",
                        metrics={
                            "completed_tasks": completed,
                            "task_status": "COMPLETE",
                            "primary_scaling_eligible": False,
                        },
                    )
                rows = admitted_rows(accepted[:64], purpose=task["purpose"])
                base_rows = [r for shard in package["shards"] for r in shard]
                if max(len(r["tokens"]) for r in rows) > max(
                    len(r["tokens"]) for r in base_rows
                ) or any(
                    max(abs(int(r["values"][i])).bit_length() for r in rows)
                    > max(abs(int(r["values"][i])).bit_length() for r in base_rows)
                    for i in range(4)
                ):
                    raise ValueError("new short target exceeds frozen calibrated input envelope")
                package = seal(dict(package, short_rows=rows))
            path = self.host_root / "packages" / (task["identity"][7:] + ".json")
            if path.exists():
                if load(path) != package:
                    raise ValueError("short recovery cut differs")
            else:
                atomic_write(path, package)
            channel = self.channel(
                "training_request",
                "learner",
                dict(base_channel["data"], artifact=reference(self.root, path)),
            )
            trained, _, _ = self.phase("train", {"training": [channel]}, task)
            state = (
                self.root
                / "run-state/discovery-learning"
                / self.campaign["identity"][7:]
                / task["identity"][7:]
            )
            with Recovery(self.journal, task, compatibility(package), state_root=state) as owner:
                owner.restore()
                record = owner.state["checkpoint"]
                full = _read(state / "checkpoints", record["payload"]["full_state"], 64 << 20)
                checkpoint = self.channel(
                    "checkpoint",
                    "generator",
                    {
                        "artifact": reference(
                            self.root,
                            confined(state / "checkpoints", full["states"]["weights"]["path"]),
                        ),
                        "compatibility": compatibility(package),
                        "profile": "bigint-body/v1",
                    },
                )
            recipes = [
                v for v in entry["model"] if load(confined(self.root, v))["kind"] == "recipe"
            ]
            candidates = self.generate_sources(self.panels, [checkpoint, *recipes], task)
            truth = [
                self.channel(
                    "short_score_truth",
                    "scorer",
                    {
                        "id": s["identity"],
                        "indices": list(map(str, range(9))),
                        "values": s["values"],
                    },
                )
                for s in self.panels
            ]
            _, target, _ = self.phase(
                "score",
                {
                    "candidates": [self.channel("short_candidates", "scorer", candidates)],
                    "score_truth": truth,
                },
                task,
            )
            scores = load(target / "role/short-scores.json")
            results[task["payload"]["id"]] = {
                "scores": scores,
                "accepted": len(accepted),
                "checkpoint": record["identity"],
                "exposure": {
                    k: trained["metrics"][k]
                    for k in ("charged_tokens", "useful_tokens", "discarded_tokens")
                },
            }
            completed.append(task["payload"]["id"])
        atomic_write(
            self.host_root / "source-results.json",
            seal(
                {
                    "version": "progressive-source-results/v1",
                    "results": results,
                    "receipts": self.receipts,
                    "scope": "DIAGNOSTIC_ONLY"
                    if self.plan["generation"]["forced"] is not None
                    else "CHECKPOINT_NATIVE",
                    "primary_scaling_eligible": False,
                }
            ),
        )
        return publish(
            self.output,
            operation="source-cycle",
            campaign=self.campaign["identity"],
            mechanical_status="COMPLETE",
            scientific_status="UNMEASURED",
            reason="MATCHED_FINITE_SOURCE_PILOT",
            metrics={
                "completed_tasks": completed,
                "task_status": "COMPLETE",
                "primary_scaling_eligible": False,
                "source_results": reference(self.root, self.host_root / "source-results.json"),
            },
        )

    def generate_sources(self, sources, model, task):
        prompts = [
            self.channel(
                "short_prompt",
                "generator",
                {"id": s["identity"], "indices": ["0", "1", "2", "3"], "values": s["values"][:4]},
            )
            for s in sources
        ]
        policy = self.channel("short_policy", "generator", self.plan["generation"])
        _, target, _ = self.phase("generate", {"visible": prompts, "model": [*model, policy]}, task)
        return load(target / "role/short-candidates.json")


def run_cycle(
    root, output, campaign, manager, request, journal, *, host=None, runtime=None, pythonpath=None
):
    from oeis_learn.generalization.progressive.storage import owned_guard, register_invocation

    cycle = SourceCycle(
        root,
        output,
        campaign,
        manager,
        request,
        journal,
        host=host,
        runtime=runtime,
        pythonpath=pythonpath,
    )
    proof = register_invocation(Path(root), campaign, manager, journal.root, cycle.host_root)
    guard = owned_guard(
        {"campaign": campaign, "task": manager, "storage": proof}, output=cycle.host_root
    )
    guard.roots.append(cycle.output / "result.json")
    cycle.host_guard = guard
    with guard.activate():
        return cycle.run()
