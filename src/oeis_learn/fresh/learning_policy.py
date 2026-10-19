"""Full-denominator visible selection, predeclared panels and curriculum."""

from collections import defaultdict
import math
from oeis_learn.synthesis.contracts import identity


def score_panel(ids, results):
    if not ids or len(set(ids)) != len(ids) or set(results) - set(ids):
        raise ValueError("fixed unique panel required")
    complete = success = 0
    for row in results.values():
        if set(row) != {"complete", "verified"} or any(type(v) is not bool for v in row.values()):
            raise ValueError("visible evidence required")
        if row["verified"] and not row["complete"]:
            raise ValueError("partial evidence cannot be verified")
        complete += row["complete"]
        success += row["verified"]
    return dict(
        metric="visible_fit_accuracy_20",
        denominator=len(ids),
        complete=complete,
        success=success,
        accuracy=success / len(ids),
        coverage=complete / len(ids),
        panel_sha256=identity(list(ids)),
    )


def checkpoint_key(report, *, loss, loss_complete, update):
    if report["metric"] != "visible_fit_accuracy_20" or type(update) is not int or update < 0:
        raise ValueError("visible-only checkpoint criterion required")
    if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
        raise ValueError("finite teacher-forced loss required")
    # Incomplete evaluation cannot outrank a completely evaluated easier subset:
    # the numerator always divides by the immutable full panel denominator.
    return (
        -report["success"] / report["denominator"],
        loss if loss_complete else float("inf"),
        update,
    )


def fixed_panel(rows):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate panel source")
    stages = defaultdict(list)
    for row in sorted(rows, key=lambda r: identity(["diagnostic-panel/v1", r["id"]])):
        stages[row["stage"]].append(row["id"])
    keys = sorted(stages, key=str)
    ordered = []
    for i in range(max(map(len, stages.values()), default=0)):
        ordered.extend(stages[k][i] for k in keys if i < len(stages[k]))
    if len(ordered) < 32:
        raise ValueError("diagnostic panel shortfall")
    return tuple(ordered[:32])


class Curriculum:
    def __init__(self, validation_every=500):
        if type(validation_every) is not int or validation_every <= 0:
            raise ValueError("frozen validation cadence required")
        self.stage = 0
        self.streak = 0
        self.last_update = -1
        self.validation_every = validation_every

    def observe(self, report, *, update):
        if (
            report["metric"] != "visible_fit_accuracy_20"
            or type(update) is not int
            or update <= self.last_update
        ):
            raise ValueError("scheduled distinct visible validation required")
        if update % self.validation_every:
            raise ValueError("unscheduled curriculum validation")
        if self.last_update >= 0 and update - self.last_update != self.validation_every:
            self.streak = 0
        self.last_update = update
        enough = (
            report["complete"] * 100 >= 95 * report["denominator"]
            and report["success"] * 5 >= 4 * report["denominator"]
        )
        self.streak = self.streak + 1 if enough else 0
        if self.streak >= 2 and self.stage < 3:
            self.stage += 1
            self.streak = 0
            return True
        return False

    def earlier_count(self, batch):
        if type(batch) is not int or batch <= 0:
            raise ValueError("invalid batch")
        return (batch + 3) // 4 if self.stage else 0

    def snapshot(self):
        return dict(
            stage=self.stage,
            streak=self.streak,
            last_update=self.last_update,
            validation_every=self.validation_every,
        )

    @classmethod
    def restore(cls, state):
        if (
            set(state) != {"stage", "streak", "last_update", "validation_every"}
            or any(type(v) is not int for v in state.values())
            or not 0 <= state["stage"] <= 3
            or state["streak"] < 0
            or state["last_update"] < -1
        ):
            raise ValueError("invalid curriculum state")
        result = cls(state["validation_every"])
        result.__dict__.update(state)
        return result


class Stopping:
    def __init__(self):
        self.best = -1
        self.stale = 0

    def observe(self, success):
        if type(success) is not int or success < 0:
            raise ValueError("full-panel success count required")
        if success >= self.best + 1:
            self.best = success
            self.stale = 0
        else:
            self.stale += 1
        return self.stale >= 8


def choose_checkpoint(evaluations, *, panel_sha256):
    """Reject easier/changed panels before comparing full-denominator competence."""
    if not evaluations:
        raise ValueError("no checkpoint evidence")
    for item in evaluations:
        if (
            set(item) != {"checkpoint", "visible", "loss", "loss_complete", "update"}
            or item["visible"]["panel_sha256"] != panel_sha256
        ):
            raise ValueError("checkpoint panel/contract changed")
    return min(
        evaluations,
        key=lambda item: checkpoint_key(
            item["visible"],
            loss=item["loss"],
            loss_complete=item["loss_complete"],
            update=item["update"],
        ),
    )["checkpoint"]
