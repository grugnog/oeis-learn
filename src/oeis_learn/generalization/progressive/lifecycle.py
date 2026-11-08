"""Finite campaign transitions; worker identity is owned by the existing supervisor."""

from pathlib import Path

from .campaign import validate_campaign, validate_task
from .contracts import atomic_write, load, record, seal


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

    def read(self):
        self.state = {"tasks": {}, "status": "FROZEN", "invocation": None}
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
            if any(self.state["tasks"].get(p["id"]) != "COMPLETE" for p in deps):
                continue
            edges = [
                e for e in self.campaign["payload"]["graph"]["edges"] if e["target"] == plan["slot"]
            ]
            # Scientific/diagnostic edges need separately verified decisions; an
            # ordinary process completion never supplies those claim semantics.
            if any(e["condition"] != "COMPLETE" for e in edges):
                continue
            return record(
                "task",
                dict(plan, campaign=self.campaign["identity"]),
                created_at=self.campaign["created_at"],
                purpose=self.campaign["purpose"],
                track=self.campaign["track"],
            )
        return None

    def pause(self):
        if not (self.root / "pause.json").exists():
            atomic_write(
                self.root / "pause.json",
                seal({"campaign": self.campaign["identity"], "requested": True}),
            )
        self.read()
        return "PAUSED" if self.state["status"] == "PAUSED" else "PENDING_EXTERNAL_DEADLINE"

    def boundary(self, task, status, invocation):
        validate_task(task, campaign=self.campaign)
        if status not in {"COMPLETE", "PARTIAL", "FAILED", "PAUSED"}:
            raise ValueError("closed task outcome required")
        tasks = dict(self.state["tasks"])
        tasks[task["payload"]["id"]] = status
        return self.append(
            {
                "tasks": tasks,
                "status": "PAUSED" if (self.root / "pause.json").exists() else status,
                "invocation": invocation,
            }
        )
