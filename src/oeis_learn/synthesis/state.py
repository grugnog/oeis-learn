"""Pure immutable contracts and conservative transitions; persistence is Phase 3."""

from dataclasses import dataclass, asdict
import math
import re
from .contracts import identity


def require_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("invalid content identity")


def positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("positive finite budget required")


@dataclass(frozen=True)
class ExperimentContract:
    data: str
    split: str
    model: str
    inputs: str
    decoder: str
    solver: str
    runtime: str
    resources: str
    policy: str
    track: str
    foundation: str | None
    seed: int
    budget: float

    def __post_init__(self):
        for name in (
            "data",
            "split",
            "model",
            "inputs",
            "decoder",
            "solver",
            "runtime",
            "resources",
            "policy",
        ):
            require_hash(getattr(self, name))
        if self.foundation is not None:
            require_hash(self.foundation)
        if (
            self.track not in {"generic_only", "assisted"}
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("invalid track or seed")
        positive(self.budget)

    @property
    def sha256(self):
        return identity({"version": "synthesis-experiment/v1", **asdict(self)})


class BudgetLedger:
    """Debit the whole reservation before launch; missing completion never refunds it."""

    def __init__(self, budget):
        positive(budget)
        self.budget = budget
        self.records = {}

    @property
    def remaining(self):
        return self.budget - sum(r["debit"] for r in self.records.values())

    def start(self, key, debit):
        positive(debit)
        if not isinstance(key, str) or not key or key in self.records or debit > self.remaining:
            raise ValueError("duplicate attempt or exhausted budget")
        self.records[key] = {"debit": debit, "status": "active"}
        # Caller durably saves this snapshot before executing any work.
        return self.snapshot()

    def finish(self, key, status):
        if (
            status not in {"completed", "interrupted", "cancelled", "failed"}
            or self.records.get(key, {}).get("status") != "active"
        ):
            raise ValueError("invalid completion")
        self.records[key]["status"] = status

    def snapshot(self):
        return {"budget": self.budget, "records": {k: dict(v) for k, v in self.records.items()}}

    @classmethod
    def restore(cls, record):
        if set(record) != {"budget", "records"} or not isinstance(record["records"], dict):
            raise ValueError("invalid ledger")
        ledger = cls(record["budget"])
        for key, value in record["records"].items():
            if set(value) != {"debit", "status"}:
                raise ValueError("invalid debit record")
            ledger.start(key, value["debit"])
            if value["status"] != "active":
                ledger.finish(key, value["status"])
        return ledger


def validate_checkpoint(record, contract):
    states = {"weights", "optimizer", "rng", "data_order", "selection", "curriculum", "replay"}
    if set(record) != states | {"contract", "ledger", "state"}:
        raise ValueError("complete checkpoint required; weights-only is analysis")
    if record["contract"] != contract.sha256 or record["state"] not in {
        "active",
        "interrupted",
        "cancelled",
    }:
        raise ValueError("resume requires eligible identical contract")
    for key in states:
        require_hash(record[key])
    ledger = BudgetLedger.restore(record["ledger"])
    if ledger.budget != contract.budget:
        raise ValueError("changed cumulative allowance")
    return ledger


def analysis_identity(parent_contract, checkpoint, analysis_contract):
    for h in (parent_contract, checkpoint, analysis_contract):
        require_hash(h)
    return identity(
        dict(
            kind="immutable-analysis-child/v1",
            parent=parent_contract,
            checkpoint=checkpoint,
            analysis=analysis_contract,
        )
    )


@dataclass(frozen=True)
class FinalPlan:
    # arm tuple: name, seed, checkpoint, recipe, portfolio, runtime, budget
    cohort: str
    decisions: tuple
    arms: tuple
    denominator: int
    controls: tuple
    budget: float
    DECISIONS = ("representation", "curriculum", "replay", "assistance", "encoder")

    def __post_init__(self):
        require_hash(self.cohort)
        positive(self.budget)
        if (
            type(self.decisions) is not tuple
            or len(self.decisions) != len(self.DECISIONS)
            or set(dict(self.decisions)) != set(self.DECISIONS)
        ):
            raise ValueError("all study decisions must be complete")
        for _, h in self.decisions:
            require_hash(h)
        if type(self.denominator) is not int or self.denominator <= 0:
            raise ValueError("fixed denominator required")
        if self.controls != ("correct", "shuffled", "absent", "template"):
            raise ValueError("complete fixed controls required")
        if type(self.arms) is not tuple or not self.arms or len(self.arms) > 128:
            raise ValueError("invalid arm roster")
        names = set()
        total = 0
        for arm in self.arms:
            if type(arm) is not tuple or len(arm) != 7:
                raise ValueError("invalid arm")
            name, seed, checkpoint, recipe, portfolio, runtime, budget = arm
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or type(seed) is not int
                or not 0 <= seed < 2**32
            ):
                raise ValueError("invalid duplicate arm/seed")
            names.add(name)
            for h in (checkpoint, recipe, portfolio, runtime):
                require_hash(h)
            positive(budget)
            total += budget
        if total > self.budget:
            raise ValueError("aggregate budget cannot cover registered arms")

    @property
    def sha256(self):
        return identity({"version": "final-evaluation/v1", **asdict(self)})


class FinalStudy:
    def __init__(self, cohort):
        require_hash(cohort)
        self.cohort = cohort
        self.state = "sealed"
        self.plan = None
        self.ledger = None
        self.status = {}
        self.arm_spent = {}
        self.active = {}

    def authorize(self, plan):
        if plan.cohort != self.cohort or (self.plan and plan.sha256 != self.plan.sha256):
            raise ValueError("changed final plan/cohort; exposed cohort cannot become fresh")
        if self.plan:
            return
        if self.state != "sealed":
            raise ValueError("fresh plan requires sealed cohort")
        self.plan = plan
        self.state = "authorized"
        self.ledger = BudgetLedger(plan.budget)
        self.status = {arm[0]: "unstarted" for arm in plan.arms}
        self.arm_spent = {name: 0 for name in self.status}

    def start(self, arm, checkpoint, debit, *, operation):
        positive(debit)
        if operation != "evaluate_final" or not self.plan or arm not in self.status:
            raise ValueError("only explicitly registered final evaluation may expose cohort")
        row = next(a for a in self.plan.arms if a[0] == arm)
        if (
            row[2] != checkpoint
            or self.status[arm] in {"active", "completed"}
            or self.arm_spent[arm] + debit > row[6]
        ):
            raise ValueError("changed checkpoint or exhausted/ineligible arm")
        key = f"{arm}:{len(self.ledger.records)}"
        self.ledger.start(key, debit)
        self.arm_spent[arm] += debit
        self.active[arm] = key
        self.status[arm] = "active"
        self.state = "exposed"
        return self.snapshot()  # Must be persisted before first prompt, including on retries.

    def finish(self, arm, status):
        if arm not in self.active:
            raise ValueError("no active arm")
        self.ledger.finish(self.active[arm], status)
        del self.active[arm]
        self.status[arm] = status

    def snapshot(self):
        return dict(
            cohort=self.cohort,
            state=self.state,
            plan=self.plan.sha256 if self.plan else None,
            ledger=self.ledger.snapshot() if self.ledger else None,
            status=dict(self.status),
            arm_spent=dict(self.arm_spent),
            active=dict(self.active),
        )

    @classmethod
    def restore(cls, record, plan):
        if (
            set(record) != {"cohort", "state", "plan", "ledger", "status", "arm_spent", "active"}
            or record["plan"] != plan.sha256
        ):
            raise ValueError("invalid or changed final study")
        study = cls(record["cohort"])
        study.authorize(plan)
        study.ledger = BudgetLedger.restore(record["ledger"])
        if study.ledger.budget != plan.budget:
            raise ValueError("changed final budget")
        if set(record["status"]) != set(study.status) or set(record["arm_spent"]) != set(
            study.status
        ):
            raise ValueError("missing registered arm")
        spent = {a: 0 for a in study.status}
        active = {}
        status = {a: "unstarted" for a in study.status}
        for key, r in study.ledger.records.items():
            arm = key.rsplit(":", 1)[0]
            if arm not in spent or status[arm] in {"active", "completed"}:
                raise ValueError("invalid arm history")
            spent[arm] += r["debit"]
            status[arm] = r["status"]
            if r["status"] == "active":
                active[arm] = key
        expected_state = "exposed" if study.ledger.records else "authorized"
        if (
            record["state"] != expected_state
            or record["arm_spent"] != spent
            or record["active"] != active
            or record["status"] != status
            or any(spent[a[0]] > a[6] for a in plan.arms)
        ):
            raise ValueError("final freshness/budget history mismatch")
        study.state = expected_state
        study.arm_spent = spent
        study.status = status
        study.active = active
        return study


class CandidateState:
    def __init__(self, candidate_id, generation, contract):
        require_hash(contract)
        if type(generation) is not int or generation < 0:
            raise ValueError("invalid generation")
        self.candidate_id = candidate_id
        self.generation = generation
        self.contract = contract
        self.state = "decoded"

    def response(self, generation, contract, outcome):
        if generation != self.generation or contract != self.contract:
            return False
        if self.state != "decoded" or outcome not in {
            "assignment_found",
            "unknown",
            "unsupported",
            "unsat_in_scope",
            "resource_exhausted",
            "verification_failure",
        }:
            raise ValueError("invalid candidate transition")
        self.state = "provisional" if outcome == "assignment_found" else outcome
        return True

    def admit(self, verification):
        from .admission import Admission

        if (
            self.state != "provisional"
            or not isinstance(verification, Admission)
            or verification.outcome != "verified_solution"
            or not verification.verification
            or verification.verification.get("admitted") is not True
        ):
            raise ValueError("independent final admission required")
        self.state = "admitted"


class BatchState:
    """Partial completion never refunds a batch debit or starts unstarted work."""

    def __init__(self, candidate_ids, generation, contract, ledger, *, batch_id, debit):
        if (
            not candidate_ids
            or len(candidate_ids) > 32
            or len(set(candidate_ids)) != len(candidate_ids)
        ):
            raise ValueError("bounded unique candidate batch required")
        self.candidates = {key: CandidateState(key, generation, contract) for key in candidate_ids}
        self.ledger = ledger
        self.batch_id = batch_id
        self.started = set()
        self.closed = False
        ledger.start(batch_id, debit)

    def start_candidate(self, key):
        if self.closed or key not in self.candidates or key in self.started:
            raise ValueError("invalid candidate start")
        self.started.add(key)

    def response(self, key, generation, contract, outcome):
        if self.closed or key not in self.started:
            return False
        return self.candidates[key].response(generation, contract, outcome)

    def close(self, status):
        if self.closed:
            raise ValueError("batch already finalized")
        self.ledger.finish(self.batch_id, status)
        self.closed = True
        return {
            key: (candidate.state if key in self.started else "unstarted")
            for key, candidate in self.candidates.items()
        }
