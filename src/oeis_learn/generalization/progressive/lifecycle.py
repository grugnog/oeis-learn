"""Finite campaign transitions; worker identity is owned by the existing supervisor."""

from contextlib import contextmanager
from pathlib import Path

from .campaign import validate_campaign, validate_task
from .contracts import atomic_write, digest, load, record, seal


class RunStore:
    def __init__(self, root, campaign):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.campaign = validate_campaign(campaign)
        binding = self.root / "campaign.json"
        if binding.exists():
            if load(binding) != self.campaign:
                raise ValueError("run campaign changed")
        else:
            atomic_write(binding, self.campaign)
        self.read()

    @contextmanager
    def lease(self):
        import fcntl

        path = self.root / "run.lock"
        if path.is_symlink():
            raise ValueError("run lease alias")
        with path.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("another invocation owns this campaign") from None
            self.read()
            yield

    def read(self):
        self.state = {"tasks": {}, "status": "FROZEN", "invocation": None, "decisions": {}}
        tip = self.campaign["identity"]
        paths = sorted(self.root.glob("transition-*.json"))
        if len(paths) > 4096:
            raise ValueError("finite campaign transition limit")
        for i, p in enumerate(paths):
            v = load(p)
            if p.name != f"transition-{i:04d}.json" or seal(v) != v or v["previous"] != tip:
                raise ValueError("invalid run transition")
            self.state = v["state"]
            tip = v["identity"]
        self.tip = tip
        self.ordinal = len(paths)
        return self.state

    def append(self, state):
        if self.ordinal >= 4096:
            raise ValueError("transition allowance exhausted")
        state = dict(state, decisions=state.get("decisions", self.state.get("decisions", {})))
        atomic_write(
            self.root / f"transition-{self.ordinal:04d}.json",
            seal({"previous": self.tip, "state": state}),
        )
        return self.read()

    def next_task(self):
        nodes = {n["id"]: n for n in self.campaign["payload"]["graph"]["nodes"]}
        for plan in self.campaign["payload"]["tasks"]:
            status = self.state["tasks"].get(plan["id"])
            if status in {"COMPLETE", "FAILED", "PARTIAL"}:
                continue
            node = nodes[plan["slot"]]
            deps = [
                p for p in self.campaign["payload"]["tasks"] if p["slot"] in node["dependencies"]
            ]
            if set(node["dependencies"]) - {p["slot"] for p in deps}:
                continue
            terminal = {"COMPLETE", "PARTIAL", "FAILED"}
            if any(self.state["tasks"].get(p["id"]) not in terminal for p in deps):
                continue
            if plan["initializer"].get("parent_task") and self.state["tasks"].get(
                plan["initializer"]["parent_task"]
            ) not in {"COMPLETE", "PARTIAL"}:
                continue
            edges = [
                e for e in self.campaign["payload"]["graph"]["edges"] if e["target"] == plan["slot"]
            ]
            certificates = self.state.get("decisions", {})
            if any(
                (
                    any(
                        self.state["tasks"].get(p["id"]) != "COMPLETE"
                        for p in deps
                        if p["slot"] == e["source"]
                    )
                    if e["condition"] == "COMPLETE"
                    else not any(
                        v["source_slot"] == e["source"]
                        and v["target_slot"] == e["target"]
                        and v["verdict"] == e["condition"]
                        and v["action"] == e["action"]
                        for v in certificates.values()
                    )
                )
                for e in edges
            ):
                continue
            return record(
                "task",
                dict(plan, campaign=self.campaign["identity"]),
                created_at=self.campaign["created_at"],
                purpose=self.campaign["purpose"],
                track=self.campaign["track"],
            )
        return None

    def authorize(self, bundle, *, policy, remaining, panels=None):
        from .contracts import validate
        from .decisions import verify
        from .targets import validate_policy

        validate_policy(policy)
        if policy["identity"] != self.campaign["payload"]["target_policy"]:
            raise ValueError("decision target policy is not the frozen campaign policy")
        fields = {
            "version",
            "identity",
            "source_slot",
            "target_slot",
            "decision",
            "rows",
            "roster",
            "seeds",
            "design",
            "kwargs",
        }
        if (
            set(bundle) != fields
            or bundle["version"] != "progressive-decision-bundle/v1"
            or seal(bundle) != bundle
        ):
            raise ValueError("closed producer decision bundle required")
        key = digest([bundle["source_slot"], bundle["target_slot"]])
        prior = self.state.get("decisions", {}).get(key)
        if prior is not None:
            if prior.get("bundle") != bundle["identity"]:
                raise ValueError("conflicting decision edge")
            return prior
        training_seeds = sorted(
            {
                p["seed"]
                for p in self.campaign["payload"]["tasks"]
                if p["slot"] == bundle["source_slot"] and p["kind"] == "training"
            }
        )
        if (
            training_seeds
            and not bundle["kwargs"].get("confirmation")
            and sorted(bundle["seeds"]) != training_seeds
        ):
            raise ValueError("decision seed roster differs from frozen source tasks")
        decision = bundle["decision"]
        if (
            decision["purpose"] != self.campaign["purpose"]
            or decision["track"] != self.campaign["track"]
        ):
            raise ValueError("cross-purpose decision")
        edge = next(
            (
                e
                for e in self.campaign["payload"]["graph"]["edges"]
                if e["source"] == bundle["source_slot"]
                and e["target"] == bundle["target_slot"]
                and e["condition"] == decision["payload"]["evidence_verdict"]
                and e["action"] == decision["payload"]["next_action"]
            ),
            None,
        )
        if edge is None:
            raise ValueError("decision lacks a frozen edge")
        kwargs = bundle["kwargs"]
        if set(kwargs) - {"target_id", "action", "trigger", "costs"}:
            raise ValueError("closed decision arguments required")
        visited = [
            v["failure_class"]
            for v in self.state.get("decisions", {}).values()
            if v.get("failure_class")
        ]
        verify(
            decision,
            bundle["rows"],
            bundle["roster"],
            bundle["seeds"],
            bundle["design"],
            policy=policy,
            remaining=remaining,
            visited=visited,
            **kwargs,
        )
        branch = None
        if edge["action"] == "EXPLORE_RESERVED":
            target = next(t for t in policy["payload"]["targets"] if t["id"] == kwargs["target_id"])
            branch = next(
                d for d in policy["payload"]["diagnostics"] if d["id"] == target["budget_ref"]
            )
            if bundle["target_slot"] not in branch["task_slots"] or branch["signal"] not in {
                "heldout_yield",
                "seed_disagreement",
                "runtime_bottleneck",
                "support_deficit",
            }:
                raise ValueError("diagnostic lacks frozen held-out/runtime task reservation")
            if panels is None or branch["panel"] not in panels:
                raise ValueError("original reserved diagnostic panel required")
            panel = validate("cohort", panels[branch["panel"]])
            if (
                panel["payload"]["role"] in {"assessment", "training"}
                or panel["track"] != self.campaign["track"]
            ):
                raise ValueError("diagnostic cannot consume assessment/training truth")
            if any(
                v.get("panel") == branch["panel"] for v in self.state.get("decisions", {}).values()
            ):
                raise ValueError("diagnostic panel already consumed")
        certificates = dict(self.state.get("decisions", {}))
        cert = {
            "bundle": bundle["identity"],
            "source_slot": bundle["source_slot"],
            "target_slot": bundle["target_slot"],
            "decision": decision["identity"],
            "verdict": decision["payload"]["evidence_verdict"],
            "action": decision["payload"]["next_action"],
            "failure_class": branch["failure_class"] if branch else None,
            "panel": branch["panel"] if branch else None,
        }
        certificates[key] = cert
        self.append(dict(self.state, decisions=certificates))
        return cert

    def pause_path(self):
        return self.root / ("pause-" + digest(self.state["invocation"]).split(":")[1] + ".json")

    def pause(self):
        self.read()
        if not (self.pause_path()).exists():
            atomic_write(
                self.pause_path(),
                seal({"campaign": self.campaign["identity"], "requested": True}),
            )
        self.read()
        return (
            "PAUSED"
            if self.state["status"] in {"PAUSED", "COMPLETE", "FAILED", "PARTIAL"}
            else "PENDING_EXTERNAL_DEADLINE"
        )

    def boundary(self, task, status, invocation):
        validate_task(task, campaign=self.campaign)
        if status not in {"COMPLETE", "PARTIAL", "FAILED", "PAUSED"}:
            raise ValueError("closed task outcome required")
        tasks = dict(self.state["tasks"])
        tasks[task["payload"]["id"]] = status
        return self.append(
            {
                "tasks": tasks,
                "status": "PAUSED" if (self.pause_path()).exists() else status,
                "invocation": invocation,
            }
        )
