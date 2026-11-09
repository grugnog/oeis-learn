"""Bounded durable trial state machine; sampler feedback is development-only."""

import copy
import math
import re
from contextlib import contextmanager
from pathlib import Path
from statistics import median

from oeis_learn.pipeline_validation.identity import seal

from .contracts import atomic_write, digest, load, record

TERMINAL = {"COMPLETE", "PRUNED", "FAILED", "PARTIAL", "INFEASIBLE"}


def integer(n, minimum=0):
    if type(n) is not int or not minimum <= n <= 2**63 - 1:
        raise ValueError("bounded integer required")
    return n


def scalar(n):
    if type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1:
        raise ValueError("finite capability fraction required")
    return n


def should_prune(*, value, coordinate, references, control=False):
    scalar(value)
    integer(coordinate)
    if type(control) is not bool:
        raise ValueError("explicit control flag required")
    eligible = [
        scalar(r["value"])
        for r in references
        if r["coordinate"] == coordinate
        and r["status"] == "COMPLETE"
        and r["role"] == "development"
    ]
    return (
        not control
        and coordinate in (1000000, 2000000)
        and len(eligible) >= 5
        and value < median(eligible)
    )


class TrialJournal:
    def __init__(self, root, *, study, token_cap=4000000, elapsed_cap_us=7200000000):
        if not isinstance(study, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", study) is None:
            raise ValueError("sealed study identity required")
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError("journal symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = seal(
            {
                "version": "progressive-trial-journal/v1",
                "study": study,
                "token_cap": integer(token_cap, 1),
                "elapsed_cap_us": integer(elapsed_cap_us, 1),
            }
        )
        with self.lock():
            p = self.root / "binding.json"
            if p.exists():
                if load(p) != self.binding:
                    raise ValueError("trial allowance cannot change")
            else:
                atomic_write(p, self.binding)
            self.read()

    @contextmanager
    def lock(self):
        import fcntl

        p = self.root / "journal.lock"
        if p.is_symlink():
            raise ValueError("lock alias")
        with p.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def read(self):
        events = sorted(self.root.glob("event-*.json"))
        if len(events) > 4096:
            raise ValueError("trial journal exhausted")
        self.trials = {}
        self.tip = self.binding["identity"]
        for i, p in enumerate(events):
            e = load(p)
            if (
                p.name != f"event-{i:04d}.json"
                or set(e) != {"previous", "ordinal", "state", "identity"}
                or seal(e) != e
                or e["previous"] != self.tip
            ):
                raise ValueError("trial event chain differs")
            record(
                "trial",
                e["state"],
                created_at="2026-11-08T00:00:00Z",
                purpose="acceptance",
                track="synthetic_strict",
            )
            if (
                e["state"]["proposal"]["study"] != self.binding["study"]
                or str(e["state"]["proposal"]["ordinal"]) != e["ordinal"]
            ):
                raise ValueError("trial study or ordinal differs")
            if e["state"]["status"] == "COMPLETE":
                from .trial_evidence import eligible
                if eligible(e["state"]["result"],root=self.root) is None:
                    raise ValueError("journal contains ineligible complete feedback")
            self.tip = e["identity"]
            self.trials[e["ordinal"]] = e["state"]
        self.count = len(events)

    def write(self, ordinal, state):
        if self.count >= 4096:
            raise ValueError("bounded trial events exhausted")
        record(
            "trial",
            state,
            created_at="2026-11-08T00:00:00Z",
            purpose="acceptance",
            track="synthetic_strict",
        )
        e = seal({"previous": self.tip, "ordinal": str(ordinal), "state": state})
        atomic_write(self.root / f"event-{self.count:04d}.json", e)
        self.read()
        return copy.deepcopy(state)

    def proposal(self, ordinal):
        integer(ordinal)
        self.read()
        return copy.deepcopy(self.trials[str(ordinal)]["proposal"])

    def propose(self, ordinal, *, parameters, seed):
        integer(ordinal)
        integer(seed)
        if ordinal >= 16 or not isinstance(parameters, dict) or not parameters:
            raise ValueError("bounded proposal required")
        digest(parameters)  # reject nonfinite/non-JSON settings
        proposal = seal(
            {
                "study": self.binding["study"],
                "ordinal": ordinal,
                "parameters": parameters,
                "seed": seed,
            }
        )
        with self.lock():
            self.read()
            if str(ordinal) in self.trials:
                if self.trials[str(ordinal)]["proposal"] != proposal:
                    raise ValueError("conflicting duplicate proposal")
                return proposal
            if ordinal != len(self.trials):
                raise ValueError("ordered proposals required")
            self.write(
                ordinal,
                {
                    "proposal": proposal,
                    "status": "PROPOSED",
                    "recovery_pending": False,
                    "tokens": 0,
                    "elapsed_us": 0,
                    "invocation": 0,
                    "result": None,
                },
            )
        return proposal

    def reserve(self, ordinal):
        with self.lock():
            self.read()
            s = copy.deepcopy(self.trials[str(ordinal)])
            if s["status"] != "PROPOSED":
                raise ValueError("proposal already reserved")
            s["status"] = "RESERVED"
            return self.write(ordinal, s)

    def dispatch(self, ordinal):
        with self.lock():
            self.read()
            if str(ordinal) not in self.trials:
                raise ValueError("proposal must be durable before dispatch")
            s = copy.deepcopy(self.trials[str(ordinal)])
            if s["status"] not in {"PROPOSED", "RESERVED"} and not (
                s["status"] == "RUNNING" and s["recovery_pending"]
            ):
                raise ValueError("trial cannot restart or dispatch twice")
            if (
                s["tokens"] >= self.binding["token_cap"]
                or s["elapsed_us"] >= self.binding["elapsed_cap_us"]
            ):
                raise ValueError("trial allowance exhausted")
            s.update(status="RUNNING", recovery_pending=False, invocation=s["invocation"] + 1)
            return self.write(ordinal, s)

    def interrupt(self, ordinal, *, tokens, elapsed_us):
        with self.lock():
            self.read()
            s = copy.deepcopy(self.trials[str(ordinal)])
            self.cost(s, tokens, elapsed_us)
            if s["status"] != "RUNNING" or s["recovery_pending"]:
                raise ValueError("no active invocation")
            s.update(tokens=tokens, elapsed_us=elapsed_us, recovery_pending=True)
            if tokens >= self.binding["token_cap"] or elapsed_us >= self.binding["elapsed_cap_us"]:
                s.update(status="PARTIAL", recovery_pending=False)
            return self.write(ordinal, s)

    def cost(self, s, tokens, elapsed_us):
        integer(tokens)
        integer(elapsed_us)
        if (
            tokens < s["tokens"]
            or elapsed_us < s["elapsed_us"]
            or tokens > self.binding["token_cap"]
            or elapsed_us > self.binding["elapsed_cap_us"]
        ):
            raise ValueError("trial costs must be cumulative within reservation")

    def tell(self, ordinal, result):
        if (
            set(result) - {"evidence"} != {"status", "tokens", "elapsed_us", "value", "role"}
            or result["status"] not in TERMINAL
            or result["role"] != "development"
        ):
            raise ValueError("closed development-only terminal result required")
        if result["value"] is not None:
            scalar(result["value"])
        if result["status"] == "COMPLETE" and result["value"] is None:
            raise ValueError("complete trial needs measured capability")
        from .trial_evidence import eligible

        if result["status"] == "COMPLETE" and eligible(result, root=self.root) is None:
            raise ValueError("complete trial lacks eligible bound development evidence")
        with self.lock():
            self.read()
            s = copy.deepcopy(self.trials[str(ordinal)])
            if s["result"] is not None:
                if digest(s["result"]) != digest(result):
                    raise ValueError("conflicting duplicate tell")
                return s["result"]
            if s["status"] != "RUNNING" or s["recovery_pending"]:
                raise ValueError("terminal trial or inactive invocation")
            self.cost(s, result["tokens"], result["elapsed_us"])
            s.update(
                status=result["status"],
                tokens=result["tokens"],
                elapsed_us=result["elapsed_us"],
                result=result,
            )
            self.write(ordinal, s)
            return copy.deepcopy(result)

    def equal_token_eligible(self, *, target):
        integer(target, 1)
        self.read()
        from .trial_evidence import eligible

        return [
            int(k)
            for k, s in self.trials.items()
            if s["status"] == "COMPLETE"
            and s["tokens"] >= target
            and eligible(s["result"], root=self.root) is not None
        ]
