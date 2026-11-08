"""Durable debit-before-work journals; pending work never restores entitlement."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from .contracts import atomic_write, load, digest, seal, _check, schema
from oeis_learn.generalization.budgets import Ledger, integer

KEYS = (
    "tokens",
    "draws",
    "rejections",
    "attempts",
    "native_work",
    "elapsed_us",
    "disk_bytes",
    "archive_bytes",
)


def vector(value):
    _check(schema("common")["$defs"]["caps"], value, schema("common"))
    return value


class BudgetJournal:
    """One campaign journal shared by all tasks/invocations, protected by flock.

    Reservations debit maximum unproven work. Only supervisor monotonic termination
    observations on the original boot can release unused elapsed time. No worker
    timestamp or checkpoint can refund tokens/draws/attempts/discarded updates.
    """

    def __init__(self, root, campaign, caps):
        self.root = Path(root)
        if self.root.is_symlink() or any(p.is_symlink() for p in self.root.parents):
            raise ValueError("symlink journal root")
        self.root.mkdir(parents=True, exist_ok=True)
        if any(
            (self.root / name).is_symlink()
            for name in ("budget.lock", "budget.jsonl", "budget-binding.json")
        ):
            raise ValueError("symlink budget state")
        self.path = self.root / "budget.jsonl"
        self.binding = dict(campaign=campaign, caps=vector(caps))
        with self.lock():
            p = self.root / "budget-binding.json"
            if p.exists():
                if load(p) != seal(self.binding):
                    raise ValueError("campaign budget binding changed")
            else:
                atomic_write(p, seal(self.binding))
            self._read()

    @contextmanager
    def lock(self):
        with (self.root / "budget.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _read(self):
        events = []
        previous = digest(self.binding)
        pending = {}
        finished = set()
        totals = dict.fromkeys(KEYS, 0)
        if self.path.exists():
            if self.path.is_symlink() or self.path.stat().st_size > 64 << 20:
                raise ValueError("bounded regular journal required")
            for line in self.path.read_text().splitlines():
                event = json.loads(line)
                if (
                    set(event)
                    != {
                        "identity",
                        "previous",
                        "index",
                        "operation",
                        "event",
                        "cost",
                        "boot",
                        "start_us",
                        "end_us",
                    }
                    or seal(event) != event
                    or event["previous"] != previous
                    or event["index"] != len(events)
                ):
                    raise ValueError("journal chain broken")
                vector(event["cost"])
                integer(event["index"])
                integer(event["start_us"])
                if (
                    not isinstance(event["boot"], str)
                    or not event["boot"]
                    or len(event["boot"]) > 128
                ):
                    raise ValueError("bounded boot identity")
                key = event["operation"]
                if event["event"] == "RESERVE":
                    if key in pending or key in finished:
                        raise ValueError("duplicate ordinal")
                    pending[key] = event
                    for k in KEYS:
                        totals[k] += event["cost"][k]
                elif event["event"] == "COMMIT":
                    if key not in pending:
                        raise ValueError("unreserved commit")
                    original = pending.pop(key)
                    finished.add(key)
                    if (
                        event["boot"] != original["boot"]
                        or event["start_us"] != original["start_us"]
                    ):
                        raise ValueError("supervisor clock binding changed")
                    if event["end_us"] is not None and (
                        type(event["end_us"]) is not int
                        or event["end_us"] < event["start_us"]
                        or event["cost"]["elapsed_us"] != event["end_us"] - event["start_us"]
                    ):
                        raise ValueError("invalid monotonic elapsed proof")
                    if (
                        event["end_us"] is None
                        and event["cost"]["elapsed_us"] != original["cost"]["elapsed_us"]
                    ):
                        raise ValueError("missing termination proof cannot refund time")
                    if (
                        any(
                            event["cost"][k] != original["cost"][k]
                            for k in KEYS
                            if k != "elapsed_us"
                        )
                        or event["cost"]["elapsed_us"] > original["cost"]["elapsed_us"]
                    ):
                        raise ValueError("invalid debit refund")
                    totals["elapsed_us"] -= (
                        original["cost"]["elapsed_us"] - event["cost"]["elapsed_us"]
                    )
                else:
                    raise ValueError("unknown journal event")
                previous = event["identity"]
                events.append(event)
        if any(totals[k] > self.binding["caps"][k] for k in KEYS):
            raise ValueError("budget exceeded")
        return events, pending, finished, totals, previous

    def _append(self, event):
        with self.path.open("a") as stream:
            stream.write(json.dumps(seal(event), sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(self.root, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def reserve(self, task, operation, ordinal, costs, *, boot, monotonic_us):
        integer(ordinal)
        integer(monotonic_us)
        vector(costs)
        key = f"{task}/{operation}/{ordinal}"
        with self.lock():
            events, pending, finished, totals, previous = self._read()
            prior = next(
                (e for e in events if e["operation"] == key and e["event"] == "RESERVE"), None
            )
            if prior:
                if prior["cost"] != costs:
                    raise ValueError("conflicting duplicate ordinal")
                return dict(key=key, committed=key in finished, reserved=prior)
            for k in KEYS:
                if k == "tokens" and costs[k]:
                    if (
                        not Ledger(self.binding["caps"][k], totals[k])
                        .reserve(key, costs[k])
                        .debited
                        and costs[k]
                    ):
                        raise ValueError("token reservation exhausted")
                elif totals[k] + costs[k] > self.binding["caps"][k]:
                    raise ValueError("reservation exhausted: " + k)
            if not boot:
                raise ValueError("supervisor boot identity required")
            self._append(
                dict(
                    previous=previous,
                    index=len(events),
                    operation=key,
                    event="RESERVE",
                    cost=costs,
                    boot=boot,
                    start_us=monotonic_us,
                    end_us=None,
                )
            )
            return dict(key=key, committed=False)

    def commit(self, key, *, boot=None, monotonic_us=None, termination_proven=False):
        with self.lock():
            events, pending, finished, totals, previous = self._read()
            if key in finished:
                return
            if key not in pending:
                raise ValueError("unreserved commit")
            original = pending[key]
            cost = dict(original["cost"])
            if termination_proven:
                if (
                    boot != original["boot"]
                    or type(monotonic_us) is not int
                    or monotonic_us < original["start_us"]
                ):
                    raise ValueError("unproven cross-boot elapsed")
                elapsed = monotonic_us - original["start_us"]
                if elapsed > cost["elapsed_us"]:
                    raise ValueError("supervisor exceeded reservation")
                cost["elapsed_us"] = elapsed
            self._append(
                dict(
                    previous=previous,
                    index=len(events),
                    operation=key,
                    event="COMMIT",
                    cost=cost,
                    boot=original["boot"],
                    start_us=original["start_us"],
                    end_us=monotonic_us if termination_proven else None,
                )
            )

    def snapshot(self):
        with self.lock():
            events, pending, finished, totals, previous = self._read()
            return dict(
                consumed=totals,
                remaining={k: self.binding["caps"][k] - totals[k] for k in KEYS},
                pending=sorted(pending),
                committed=sorted(finished),
                watermark=previous,
            )

    def reconcile(self, task=None):
        # Without trustworthy termination proof, active time outside the last
        # observed interval is unknowable. Exhaust this task's remaining clock;
        # never infer entitlement from calendar timestamps or a new output path.
        if task is not None:
            remaining = self.task_remaining(task)
            prefix = task["payload"]["id"] + "/"
            if (
                any(key.startswith(prefix) for key in self.snapshot()["pending"])
                and remaining["elapsed_us"]
            ):
                costs = dict.fromkeys(KEYS, 0)
                costs["elapsed_us"] = remaining["elapsed_us"]
                event = self.reserve(
                    task["payload"]["id"],
                    "unknown_elapsed",
                    0,
                    costs,
                    boot="termination-unproven",
                    monotonic_us=0,
                )
                self.commit(event["key"])
        return self.snapshot()

    def task_remaining(self, task):
        from .campaign import validate_task

        p = validate_task(task)["payload"]
        if p["campaign"] != self.binding["campaign"]:
            raise ValueError("task journal campaign differs")
        with self.lock():
            binding = self.root / ("task-binding-" + digest(p["id"]).split(":")[1] + ".json")
            if binding.exists():
                if load(binding) != task:
                    raise ValueError("duplicate identity cannot enlarge task or change output")
            else:
                atomic_write(binding, task)
            events, _, _, _, _ = self._read()
            totals = dict.fromkeys(KEYS, 0)
            for event in events:
                if not event["operation"].startswith(p["id"] + "/"):
                    continue
                if event["event"] == "RESERVE":
                    for key in KEYS:
                        totals[key] += event["cost"][key]
                elif event["event"] == "COMMIT":
                    reservation = next(
                        e
                        for e in events
                        if e["operation"] == event["operation"] and e["event"] == "RESERVE"
                    )
                    totals["elapsed_us"] -= (
                        reservation["cost"]["elapsed_us"] - event["cost"]["elapsed_us"]
                    )
            if any(totals[k] > p["caps"][k] for k in KEYS):
                raise ValueError("task allowance exceeded")
            return {k: p["caps"][k] - totals[k] for k in KEYS}


class TaskLease:
    """One stable lease per task id in the campaign, independent of output path."""

    def __init__(self, root, task):
        from .campaign import validate_task

        p = validate_task(task)["payload"]
        self.path = Path(root) / (
            "task-" + digest([p["campaign"], p["id"]]).split(":")[1] + ".lock"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = None

    def __enter__(self):
        if self.path.is_symlink():
            raise ValueError("symlink task lease")
        self.stream = self.path.open("a")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise ValueError("duplicate active task lease") from None
        return self

    def __exit__(self, *_):
        self.stream.close()
