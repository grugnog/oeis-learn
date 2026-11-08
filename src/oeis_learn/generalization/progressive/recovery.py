"""Task-leased write-ahead updates over the shared cost journal and tensor store.

The immutable recovery event is the commit point. Tensor-store latest pointers
are never recovery authority. Supervisor clock proofs are supplied by the owner,
not inferred from worker timestamps, files, or time spent powered off.
"""

import copy

from oeis_learn.generalization.budgets import Ledger, integer

from .budgets import KEYS, TaskLease
from .campaign import validate_task
from .contracts import atomic_write, digest, load, seal


class Recovery:
    def __init__(self, journal, task, compatibility):
        self.journal = journal
        self.task = validate_task(task)
        self.p = self.task["payload"]
        journal.task_remaining(task)  # bind stable identity/output/caps before I/O
        self.root = journal.root / ("recovery-" + digest(self.p["id"]).split(":")[1])
        if self.root.is_symlink() or (self.root / "checkpoints").is_symlink():
            raise ValueError("symlink recovery storage")
        self.root.mkdir(exist_ok=True)
        self.compatibility = compatibility
        self.lease = TaskLease(journal.root, task)
        self.owned = False
        self.store = None

    def __enter__(self):
        self.lease.__enter__()
        self.owned = True
        try:
            self.read()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *args):
        self.owned = False
        self.lease.__exit__(*args)

    def read(self):
        if not self.owned:
            raise ValueError("exclusive task lease required")
        binding = seal({"task": self.task, "compatibility": self.compatibility})
        path = self.root / "binding.json"
        if path.exists():
            if load(path) != binding:
                raise ValueError("changed recovery lineage")
        else:
            atomic_write(path, binding)
        self.tip = binding["identity"]
        self.state = {
            "checkpoint": None,
            "useful_tokens": 0,
            "pending": None,
            "status": "NEW",
            "last": None,
        }
        paths = sorted(self.root.glob("event-*.json"))
        if len(paths) > 65536:
            raise ValueError("recovery journal exhausted")
        for i, path in enumerate(paths):
            e = load(path, maximum=4 << 20)
            if (
                path.name != f"event-{i:05d}.json"
                or set(e) != {"previous", "state", "identity"}
                or seal(e) != e
                or e["previous"] != self.tip
                or set(e["state"]) != set(self.state)
            ):
                raise ValueError("invalid recovery event chain")
            self.state, self.tip = e["state"], e["identity"]
        self.ordinal = len(paths)
        lineage = self.root / "lineage.json"
        self.base = {"tokens": 0, "updates": 0, "useful": 0, "ancestor_cost_us": 0}
        if lineage.exists():
            value = load(lineage)
            if (
                seal(value) != value
                or value["task"] != self.task["identity"]
                or value["compatibility"] != self.compatibility
            ):
                raise ValueError("successor lineage binding differs")
            self.base = value["base"]
            for n in self.base.values():
                integer(n)
        self.token_cap = self.base["tokens"] + self.p["caps"]["tokens"]
        from oeis_learn.fresh.bigint_state import ProgressiveCheckpointStore

        self.store = ProgressiveCheckpointStore(
            self.root / "checkpoints",
            compatibility=self.compatibility,
            campaign=self.p["campaign"],
            task=self.task["identity"],
            token_cap=self.token_cap,
        )
        return copy.deepcopy(self.state)

    def sequence(self):
        with self.journal.lock():
            return len(self.journal._read()[0])

    def append(self, **changes):
        if self.ordinal >= 65536:
            raise ValueError("recovery journal exhausted")
        s = dict(self.state, **changes)
        atomic_write(
            self.root / f"event-{self.ordinal:05d}.json", seal({"previous": self.tip, "state": s})
        )
        self.read()

    def initialize(self, states, cuts):
        if not self.owned:
            raise ValueError("exclusive task lease required")
        if self.state["checkpoint"] is not None:
            raise ValueError("existing task must restore its committed cut")
        remaining = self.journal.task_remaining(self.task)
        if any(
            self.p["caps"][k] != remaining[k]
            for k in KEYS
            if k != "elapsed_us" or not hasattr(self, "clock")
        ):
            raise ValueError("spent task cannot be initialized again")
        if self.p["initializer"].get("parent_slot") and not (self.root / "lineage.json").exists():
            raise ValueError("successor must resolve its frozen parent slot")
        record = self.save(
            states,
            Ledger(self.token_cap, consumed=self.base["tokens"], updates=self.base["updates"]),
            cuts,
        )
        self.append(
            checkpoint=record,
            useful_tokens=self.base["useful"],
            status="RUNNING" if hasattr(self, "clock") else "PAUSED",
        )
        return record

    def initialize_from(self, parent, *, campaign):
        from .campaign import validate_campaign

        validate_campaign(campaign)
        validate_task(self.task, campaign=campaign, compatibility=self.compatibility)
        validate_task(parent.task, campaign=campaign, compatibility=self.compatibility)
        parent.read()
        if (
            self.state["checkpoint"] is not None
            or self.state["status"] != "NEW"
            or self.p["initializer"].get("parent_slot") != parent.p["slot"]
            or parent.state["status"] not in {"COMPLETE", "PARTIAL"}
            or self.p["arm"] != parent.p["arm"]
            or self.p["seed"] != parent.p["seed"]
        ):
            raise ValueError("reserved terminal paired predecessor required")
        restored = parent.restore()
        remaining = parent.journal.task_remaining(parent.task)
        base = {
            "tokens": restored["ledger"].consumed,
            "updates": restored["ledger"].updates,
            "useful": parent.state["useful_tokens"],
            "ancestor_cost_us": parent.base["ancestor_cost_us"]
            + parent.p["caps"]["elapsed_us"]
            - remaining["elapsed_us"],
        }
        value = seal(
            {
                "task": self.task["identity"],
                "compatibility": self.compatibility,
                "parent": parent.state["checkpoint"]["identity"],
                "base": base,
            }
        )
        path = self.root / "lineage.json"
        if path.exists():
            if load(path) != value:
                raise ValueError("conflicting predecessor cut")
        else:
            atomic_write(path, value)
        self.read()
        return self.initialize(restored["states"], restored["cuts"])

    def save(self, states, ledger, cuts):
        if not self.owned:
            raise ValueError("exclusive task lease required")
        cuts = dict(cuts, cost_journal=self.journal.snapshot()["watermark"])
        record = self.store.save(
            states,
            ledger,
            cuts=cuts,
            created_at=self.task["created_at"],
            purpose=self.task["purpose"],
            track=self.task["track"],
        )
        self.store.load(record)  # verify complete reference/tensor closure before commit
        return record

    def restore(self):
        self.read()
        if self.state["checkpoint"] is None:
            raise ValueError("no committed checkpoint")
        restored = self.store.load(self.state["checkpoint"])
        remaining = self.journal.task_remaining(self.task)
        consumed = self.base["tokens"] + self.p["caps"]["tokens"] - remaining["tokens"]
        if consumed < restored["ledger"].consumed or consumed < self.state["useful_tokens"]:
            raise ValueError("cost journal rolled back behind checkpoint")
        restored["ledger"] = Ledger(
            self.token_cap, consumed=consumed, updates=restored["ledger"].updates
        )
        restored["discarded_tokens"] = consumed - self.state["useful_tokens"]
        restored["pending"] = copy.deepcopy(self.state["pending"])
        restored["cuts"] = copy.deepcopy(self.state["checkpoint"]["payload"]["cuts"])
        return restored

    def begin(self, *, boot, monotonic_us):
        integer(monotonic_us)
        self.read()
        if self.state["status"] in {"PARTIAL", "COMPLETE"}:
            raise ValueError("terminal task requires a separately reserved compatible next task")
        # Any unclosed invocation on any boot has no trustworthy end proof.
        # A proven supervisor close covers orphan zero-time update debits too.
        # Completing these cost records does not refund tokens or other work.
        with self.journal.lock():
            events, pending, _, _, _ = self.journal._read()
        prefix = self.p["id"] + "/"
        invocations = [e for e in events if e["operation"].startswith(prefix + "invocation/")]
        if (
            invocations
            and invocations[-1]["event"] == "COMMIT"
            and invocations[-1]["end_us"] is not None
        ):
            for key, event in pending.items():
                if key.startswith(prefix + "update/") and event["cost"]["elapsed_us"] == 0:
                    self.journal.commit(key)
        self.journal.reconcile(self.task)
        remaining = self.journal.task_remaining(self.task)
        if (
            remaining["elapsed_us"] <= self.p["startup_us"] + self.p["finalization_us"]
            or not remaining["tokens"]
        ):
            self.append(status="PARTIAL")
            raise ValueError("task allowance exhausted; no automatic continuation")
        cost = dict.fromkeys(KEYS, 0)
        cost["elapsed_us"] = remaining["elapsed_us"]
        operation = self.journal.reserve(
            self.p["id"],
            "invocation",
            self.sequence(),
            cost,
            boot=boot,
            monotonic_us=monotonic_us,
        )
        self.clock = {
            "key": operation["key"],
            "boot": boot,
            "start": monotonic_us,
            "end": monotonic_us + remaining["elapsed_us"],
        }
        self.append(status="RUNNING")
        return self.restore() if self.state["checkpoint"] is not None else None

    def attach(self, authority):
        """Adopt a reservation made by the external supervisor, without refund rights."""
        self.read()
        if self.state["status"] in {"COMPLETE", "PARTIAL"}:
            raise ValueError("terminal task cannot restart")
        with self.journal.lock():
            _, pending, _, _, _ = self.journal._read()
        key = authority["reservation_key"]
        if key not in pending or not key.startswith(self.p["id"] + "/train/"):
            raise ValueError("original supervisor training reservation required")
        e = pending[key]
        if (
            authority["boot"] != e["boot"]
            or authority["start_us"] != e["start_us"]
            or authority["end_us"] != e["start_us"] + e["cost"]["elapsed_us"]
        ):
            raise ValueError("supervisor clock differs")
        self.clock = {
            "key": key,
            "boot": e["boot"],
            "start": e["start_us"],
            "end": authority["end_us"],
            "external": True,
        }
        self.append(status="RUNNING")
        return self.restore() if self.state["checkpoint"] is not None else None

    def finish_external(self, status):
        if (
            not self.owned
            or not getattr(self, "clock", {}).get("external")
            or status not in {"PAUSED", "COMPLETE", "PARTIAL", "FAILED"}
        ):
            raise ValueError("owned external invocation outcome required")
        if status == "COMPLETE" and self.state["pending"] is not None:
            raise ValueError("pending update cannot complete")
        # Only the supervisor commits elapsed after proving process termination.
        self.append(status=status)
        del self.clock

    def completed(self, update, batch):
        integer(update, 1)
        self.read()
        for path in sorted(self.root.glob("event-*.json")):
            state = load(path, maximum=4 << 20)["state"]
            if state["last"] is not None and state["last"]["update"] == update:
                if state["last"]["batch"] != batch:
                    raise ValueError("conflicting committed update replay")
                return state["checkpoint"]
        if update != self.store.load(self.state["checkpoint"])["ledger"].updates + 1:
            raise ValueError("ordered logical update required")
        return None

    def reserve(self, batch, tokens, *, boot, monotonic_us, costs=None):
        if not self.owned:
            raise ValueError("exclusive task lease required")
        integer(tokens, 1)
        integer(monotonic_us)
        if (
            self.state["status"] != "RUNNING"
            or not hasattr(self, "clock")
            or boot != self.clock["boot"]
        ):
            raise ValueError("owned current-boot invocation required")
        if not self.clock["start"] <= monotonic_us < self.clock["end"] - self.p["finalization_us"]:
            raise ValueError("protected finalization or elapsed allowance exhausted")
        if self.state["pending"] is not None and self.state["pending"]["batch"] != batch:
            raise ValueError("retry must consume the exact pending batch and data cut")
        charge = dict.fromkeys(KEYS, 0) if costs is None else dict(costs)
        if charge["elapsed_us"] or charge["tokens"] not in (0, tokens):
            raise ValueError("update time is enclosed by invocation; tokens must match")
        charge["tokens"] = tokens
        remaining = self.journal.task_remaining(self.task)
        if any(charge[k] > remaining[k] for k in KEYS):
            raise ValueError("full update cannot fit remaining task reservation")
        ordinal = self.sequence()
        event = self.journal.reserve(
            self.p["id"], "update", ordinal, charge, boot=boot, monotonic_us=monotonic_us
        )
        # This commits the COST, never the model. Even a crash before the recovery
        # event becomes visible retains this irreversible debit.
        self.journal.commit(event["key"])
        pending = {"batch": copy.deepcopy(batch), "tokens": tokens, "operation": event["key"]}
        self.append(pending=pending)
        restored = self.restore()
        return Ledger(
            restored["ledger"].cap,
            consumed=restored["ledger"].consumed,
            updates=restored["ledger"].updates,
            pending=(event["key"], tokens),
            debited=True,
        )

    def commit(self, states, ledger, cuts):
        if not self.owned:
            raise ValueError("exclusive task lease required")
        if self.state["status"] != "RUNNING" or self.state["pending"] is None:
            raise ValueError("reserved update required")
        old = self.store.load(self.state["checkpoint"])["ledger"]
        spent = (
            self.base["tokens"]
            + self.p["caps"]["tokens"]
            - self.journal.task_remaining(self.task)["tokens"]
        )
        if (
            ledger.pending is not None
            or ledger.updates != old.updates + 1
            or ledger.consumed != spent
        ):
            raise ValueError("complete matching optimizer and cost cut required")
        record = self.save(states, ledger, cuts)
        useful = self.state["useful_tokens"] + self.state["pending"]["tokens"]
        self.append(
            checkpoint=record,
            useful_tokens=useful,
            pending=None,
            last={"update": ledger.updates, "batch": self.state["pending"]["batch"]},
        )
        return record

    def close(self, *, boot, monotonic_us, complete=False):
        """Supervisor calls only after owned worker termination/publication proof."""
        if not self.owned or not hasattr(self, "clock") or self.clock.get("external"):
            raise ValueError("no invocation owned by this supervisor")
        if complete and self.state["pending"] is not None:
            raise ValueError("unfinished update cannot complete task")
        self.journal.commit(
            self.clock["key"], boot=boot, monotonic_us=monotonic_us, termination_proven=True
        )
        remaining = self.journal.task_remaining(self.task)
        terminal = not remaining["tokens"] or remaining["elapsed_us"] <= self.p["finalization_us"]
        self.append(status="COMPLETE" if complete else "PARTIAL" if terminal else "PAUSED")
        del self.clock
