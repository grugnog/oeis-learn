"""Controller-owned durable elapsed budget, independent of model rollback."""

from __future__ import annotations

from pathlib import Path
import time
from oeis_learn.tracking.foundation_metrics import EventLog, replace_json

ACTIVE = frozenset({"bootstrap", "validation", "training", "evaluation", "checkpoint", "recovery"})
INACTIVE = frozenset({"paused", "completed", "failed"})


def clock():
    return dict(
        monotonic_ns=time.monotonic_ns(),
        utc_ns=time.time_ns(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )


def elapsed(before, after):
    if after["utc_ns"] < before["utc_ns"]:
        raise ValueError("backward UTC clock")
    same = before["boot_id"] == after["boot_id"]
    delta = (
        after["monotonic_ns"] - before["monotonic_ns"]
        if same
        else after["utc_ns"] - before["utc_ns"]
    )
    if delta < 0 or (same and abs(delta - (after["utc_ns"] - before["utc_ns"])) > 5_000_000_000):
        raise ValueError("backward/inconsistent clock")
    return delta


class BudgetLedger:
    def __init__(self, root, run_id, budget_ns, *, clock=clock):
        if type(budget_ns) is not int or budget_ns <= 0:
            raise ValueError("positive frozen budget required")
        self.root, self.clock = Path(root), clock
        self.log = EventLog(self.root / "budget-events", quota_bytes=1 << 30)
        if self.log.events:
            previous = None
            for index, (identity, state) in enumerate(self.log.events.items()):
                fields = {
                    "run_id",
                    "budget_ns",
                    "sequence",
                    "charged_budget_ns",
                    "measured_ns",
                    "estimated_ns",
                    "phase_ns",
                    "status",
                    "clock",
                }
                if set(state) != fields or identity != str(index) or state["sequence"] != index:
                    raise ValueError("invalid durable ledger sequence/fields")
                if (
                    state["run_id"] != run_id
                    or state["budget_ns"] != budget_ns
                    or state["status"] not in ACTIVE | INACTIVE
                ):
                    raise ValueError("ledger identity/budget cannot change on resume")
                counts = [
                    state[key] for key in ("charged_budget_ns", "measured_ns", "estimated_ns")
                ]
                counts += list(state["phase_ns"].values())
                if (
                    any(type(n) is not int or n < 0 for n in counts)
                    or set(state["phase_ns"]) != ACTIVE
                ):
                    raise ValueError("invalid ledger accounting")
                if (
                    state["charged_budget_ns"] != state["measured_ns"] + state["estimated_ns"]
                    or sum(state["phase_ns"].values()) != state["charged_budget_ns"]
                ):
                    raise ValueError("inconsistent ledger accounting")
                if previous:
                    delta = elapsed(previous["clock"], state["clock"])
                    expected = previous["charged_budget_ns"] + (
                        delta if previous["status"] in ACTIVE else 0
                    )
                    if state["charged_budget_ns"] != expected or any(
                        state[key] < previous[key] for key in ("measured_ns", "estimated_ns")
                    ):
                        raise ValueError("ledger high-water mark/clock mismatch")
                elif state["charged_budget_ns"] != 0:
                    raise ValueError("invalid initial ledger accounting")
                previous = state
            self.state = list(self.log.events.values())[-1]
            if self.state["run_id"] != run_id or self.state["budget_ns"] != budget_ns:
                raise ValueError("ledger identity/budget cannot change on resume")
            self._record(self.state["status"], recovery=True)
        else:
            self.state = dict(
                run_id=run_id,
                budget_ns=budget_ns,
                sequence=-1,
                charged_budget_ns=0,
                measured_ns=0,
                estimated_ns=0,
                phase_ns={p: 0 for p in sorted(ACTIVE)},
                status="paused",
                clock=clock(),
            )
            self._record("paused")

    def _record(self, status, *, recovery=False):
        if status not in ACTIVE | INACTIVE:
            raise ValueError("unknown budget phase")
        now = self.clock()
        delta = elapsed(self.state["clock"], now)
        previous = self.state
        state = dict(
            previous,
            sequence=previous["sequence"] + 1,
            status=status,
            clock=now,
            phase_ns=dict(previous["phase_ns"]),
        )
        if previous["status"] in ACTIVE:
            state["charged_budget_ns"] += delta
            state["estimated_ns" if recovery else "measured_ns"] += delta
            state["phase_ns"][previous["status"]] += delta
        self.log.append(str(state["sequence"]), state)
        self.state = state
        replace_json(self.root / "ledger.json", self.snapshot())
        return self.snapshot()

    def snapshot(self):
        return {
            key: self.state[key] for key in ("run_id", "sequence", "charged_budget_ns", "status")
        }

    def heartbeat(self):
        return self._record(self.state["status"])

    def transition(self, phase):
        return self._record(phase)

    def remaining_ns(self):
        pending = (
            elapsed(self.state["clock"], self.clock()) if self.state["status"] in ACTIVE else 0
        )
        return max(0, self.state["budget_ns"] - self.state["charged_budget_ns"] - pending)

    def check_checkpoint(self, manifest):
        if (
            manifest["run_id"] != self.state["run_id"]
            or manifest["ledger_sequence"] > self.state["sequence"]
            or manifest["charged_budget_ns"] > self.state["charged_budget_ns"]
        ):
            raise ValueError("checkpoint exceeds durable ledger high-water mark")
