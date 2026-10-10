"""Frozen non-LODA trigger priority and exploratory paired-effect rules.

Unknown diagnostics stay inconclusive; known repairs are independent of these decisions.
"""

from math import isfinite
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.experiments.qualified_models import seal_record

PRIORITY = ("encoder", "self_training", "grpo", "search", "position", "capacity")


def frozen_protocol():
    return dict(
        version="bounded-decisions/v1",
        priority=list(PRIORITY),
        seeds=[7001, 7002, 7003],
        engineering_seconds=7200,
        training_seconds=21600,
        diagnostic_seconds=1800,
        diagnostic_updates=1000,
        arm_seconds=3300,
        evaluation_minutes=[15, 30, 45, 55],
        engineering_prompts=128,
        warmups=10,
        repeats=30,
        workers=[4, 8],
        fuel_tiers=[100000, 250000, 1000000],
        fuel_aggregate=50000000,
        useful_effect=0.05,
        zero_baseline_added_groups=2,
    )


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def select_trigger(diagnostic):
    allowed = {
        "encoder_collisions",
        "new_structures",
        "plateau_improvements",
        "valid_fraction",
        "reward_variance_fraction",
        "duplicate_or_prefix_failure_fraction",
        "eligible_archive",
        "short_accuracy",
        "long_accuracy",
        "matched_structural_coverage",
        "binding_scope_cause",
        "train_loss",
        "validation_loss",
        "updates",
        "diverse_programs",
    }
    if not isinstance(diagnostic, dict) or not set(diagnostic) <= allowed:
        raise ValueError("unknown diagnostic fields; final truth is forbidden")
    counts = {
        "encoder_collisions",
        "new_structures",
        "eligible_archive",
        "updates",
        "diverse_programs",
    }
    flags = {"matched_structural_coverage", "binding_scope_cause"}
    for key, value in diagnostic.items():
        if key in counts:
            valid = type(value) is int and value >= 0
        elif key in flags:
            valid = type(value) is bool
        elif key == "plateau_improvements":
            valid = isinstance(value, list) and all(_number(x) for x in value)
        else:
            valid = _number(value) and value >= 0
            if key.endswith("fraction") or key.endswith("accuracy"):
                valid = valid and value <= 1
        if not valid:
            raise ValueError(f"invalid diagnostic {key}")

    def condition(keys, fn):
        return fn() if all(k in diagnostic for k in keys) else None

    d = diagnostic
    windows = d.get("plateau_improvements")
    plateau = all(x < 0.01 for x in windows) if windows is not None and len(windows) == 3 else None
    triggers = {
        "encoder": condition(["encoder_collisions"], lambda: d["encoder_collisions"] > 0),
        "self_training": condition(
            ["new_structures"], lambda: d["new_structures"] >= 100 and plateau
        )
        if plateau is not None
        else None,
        "grpo": condition(
            ["valid_fraction", "reward_variance_fraction"],
            lambda: (
                d["valid_fraction"] >= 0.25 and d["reward_variance_fraction"] >= 0.25 and plateau
            ),
        )
        if plateau is not None
        else None,
        "search": condition(
            ["duplicate_or_prefix_failure_fraction", "eligible_archive"],
            lambda: (
                d["duplicate_or_prefix_failure_fraction"] >= 0.5 and d["eligible_archive"] >= 100
            ),
        ),
        "position": condition(
            ["short_accuracy", "long_accuracy", "matched_structural_coverage"],
            lambda: (
                d["matched_structural_coverage"]
                and d["short_accuracy"] - d["long_accuracy"] >= 0.20
            ),
        ),
        "capacity": condition(
            ["train_loss", "validation_loss", "updates", "diverse_programs"],
            lambda: (
                d["train_loss"] > 0.5
                and abs(d["validation_loss"] - d["train_loss"]) <= 0.1 * d["train_loss"]
                and d["updates"] >= 1000
                and d["diverse_programs"] >= 10000
            ),
        ),
    }
    selected = next((k for k in PRIORITY if triggers[k] is True), None)
    states = {
        k: "inconclusive"
        if triggers[k] is None
        else "not_triggered"
        if not triggers[k]
        else "selected"
        if k == selected
        else "baseline_retained"
        for k in PRIORITY
    }
    return selected, states


def useful_effect(baseline, challenger):
    if any(type(v) is not int or v < 0 for v in (baseline, challenger)):
        raise ValueError("unique solved-group counts must be nonnegative exact integers")
    return (
        challenger - baseline >= 2
        if baseline == 0
        else 100 * (challenger - baseline) >= 5 * baseline
    )


def paired_disposition(pairs):
    if not isinstance(pairs, list) or len(pairs) > 3:
        raise ValueError("at most three paired seeds")
    effects = [useful_effect(*pair) for pair in pairs]
    if not effects:
        return "inconclusive"
    if not effects[0]:
        if len(pairs) > 1:
            raise ValueError("confirmation arms cannot run after an unsuccessful first pair")
        return "baseline_retained"
    if len(pairs) < 3:
        return "inconclusive"
    return (
        "adopted"
        if sum(effects) >= 2 and sum(c for b, c in pairs) >= sum(b for b, c in pairs)
        else "baseline_retained"
    )


def decision_records(identities, diagnostic=None, pairs=(), *, charged_ns=0, failure=None):
    selected, states = select_trigger(diagnostic or {})
    diagnostic_hash = digest(diagnostic) if diagnostic is not None else None
    records = []
    for key in PRIORITY:
        state = states[key]
        reason = "required diagnostic unavailable"
        if state == "selected":
            state = paired_disposition(list(pairs))
            reason = "paired useful-effect and aggregate nonregression rule"
        elif state == "not_triggered":
            reason = "measured trigger condition false"
        elif state == "baseline_retained":
            reason = f"selected higher priority: {selected}"
        if failure:
            state, reason = "failed_gate", failure
        records.append(
            seal_record(
                "experiment_decision",
                question=key,
                hypothesis="bounded contrast: " + key,
                baseline_sha256=identities["model"],
                challenger_sha256=digest([identities["model"], key]),
                code_sha256=identities["code"],
                runtime_sha256=identities["runtime"],
                data_sha256=identities["data"],
                seeds=[7001, 7002, 7003],
                budget_ns=21600 * 10**9,
                primary_metric="unique_development_groups_per_charged_arm_hour",
                minimum_useful_effect=0.05,
                correctness_gates=[identities["code"]],
                selection_rule="first triggered in frozen priority; confirm useful effect in two of three seeds; no aggregate regression",
                stop_rule="30 minute diagnostic plus 55 minute arms; six hour cumulative hard cap; no second contrast",
                status=state,
                active_ns=charged_ns,
                charged_ns=charged_ns,
                paired_results=[
                    dict(seed=7001 + i, baseline=b, challenger=c) for i, (b, c) in enumerate(pairs)
                ]
                if key == selected
                else [],
                stop_reason=reason,
                trigger=dict(
                    diagnostic_sha256=diagnostic_hash,
                    observed=0
                    if states[key] == "not_triggered"
                    else 1
                    if states[key] in ("selected", "baseline_retained")
                    else None,
                    threshold=1,
                    disposition=states[key],
                ),
                selection_split="development",
            )
        )
    return records


class PilotSchedule:
    """Durable single-contrast schedule with interruption charging before resumption.

    The device controller must enforce the issued deadline outside its learner.
    This object controls issuance and scoring order; it never runs user commands.
    """

    def __init__(self, root, manifest_identity, *, clock=None):
        import time
        from oeis_learn.tracking.foundation_metrics import EventLog

        self.clock = clock or time.time_ns
        self.log = EventLog(root)
        self.binding = manifest_identity
        if not self.log.events:
            self.log.append("binding", dict(identity=manifest_identity))
        if self.log.events["binding"] != {"identity": manifest_identity}:
            raise ValueError("pilot manifest cannot change on resume")
        self.charged = 0
        self.selected = None
        self.selection_recorded = False
        self.diagnostic_done = False
        self.pairs = []
        self.pending = None
        for name, event in self.log.events.items():
            if name == "binding":
                continue
            if event["type"] == "issue":
                if self.pending:
                    raise ValueError("overlapping pilot operations")
                self.pending = event
            elif event["type"] == "complete":
                if (
                    not self.pending
                    or event["issued_id"] != self.pending["issued_id"]
                    or event["elapsed_ns"] < 0
                ):
                    raise ValueError("invalid pilot completion")
                self.charged += event["elapsed_ns"]
                if self.pending["kind"] == "diagnostic":
                    self.diagnostic_done = True
                elif event.get("solved_groups") is not None:
                    arm = self.pending["arm"]
                    if arm == "baseline":
                        self.pairs.append([event["solved_groups"], None])
                    else:
                        self.pairs[-1][1] = event["solved_groups"]
                self.pending = None
            elif event["type"] == "select":
                if self.selection_recorded or not self.diagnostic_done:
                    raise ValueError("invalid repeated/pre-diagnostic selection")
                self.selected = event["question"]
                self.selection_recorded = True
            elif event["type"] == "interrupted":
                if not self.pending:
                    raise ValueError("interruption without issued work")
                self.charged += event["elapsed_ns"]
                self.pending = None
            else:
                raise ValueError("unknown pilot event")
        self.interrupted = any(e.get("type") == "interrupted" for e in self.log.events.values())
        if self.pending:
            elapsed = self.clock() - self.pending["start_ns"]
            if elapsed < 0:
                raise ValueError("backward pilot clock")
            self.log.append(str(len(self.log.events)), dict(type="interrupted", elapsed_ns=elapsed))
            self.charged += elapsed
            self.pending = None
            self.interrupted = True

    def select(self, diagnostic):
        if self.selection_recorded or self.pending or not self.diagnostic_done:
            raise ValueError("one post-diagnostic selection permitted")
        selected, states = select_trigger(diagnostic)
        self.log.append(
            str(len(self.log.events)),
            dict(type="select", question=selected, diagnostic_sha256=digest(diagnostic)),
        )
        self.selected = selected
        self.selection_recorded = True
        return selected, states

    def issue(self):
        if self.pending or self.interrupted or self.charged >= 21600 * 10**9:
            raise ValueError("pending/interrupted/exhausted pilot cannot issue work")
        if not self.diagnostic_done:
            task = dict(
                kind="diagnostic",
                seed=7001,
                arm=None,
                seconds=1800,
                update_cap=1000,
                evaluation_minutes=[],
            )
        else:
            if not self.selected:
                raise ValueError("no triggered contrast")
            completed = [p for p in self.pairs if p[1] is not None]
            if completed and (not useful_effect(*completed[0]) or len(completed) == 3):
                raise ValueError("paired stop rule prohibits additional arms")
            arm = "challenger" if self.pairs and self.pairs[-1][1] is None else "baseline"
            index = len(self.pairs) - 1 if arm == "challenger" else len(self.pairs)
            task = dict(
                kind="arm",
                seed=7001 + index,
                arm=arm,
                seconds=3300,
                update_cap=None,
                evaluation_minutes=[15, 30, 45, 55],
            )
        if self.charged + task["seconds"] * 10**9 > 21600 * 10**9:
            raise ValueError("arm cannot fit remaining cumulative cap")
        task.update(type="issue", issued_id=str(len(self.log.events)), start_ns=self.clock())
        self.log.append(task["issued_id"], task)
        self.pending = task
        return dict(task, deadline_ns=task["start_ns"] + task["seconds"] * 10**9)

    def complete(self, *, solved_groups=None):
        if not self.pending:
            raise ValueError("no issued operation")
        elapsed = self.clock() - self.pending["start_ns"]
        if elapsed < 0:
            raise ValueError("backward pilot clock")
        if self.pending["kind"] == "arm" and (type(solved_groups) is not int or solved_groups < 0):
            raise ValueError("exact unique group count required for completed arm")
        if elapsed > self.pending["seconds"] * 10**9:
            self.log.append(str(len(self.log.events)), dict(type="interrupted", elapsed_ns=elapsed))
            self.charged += elapsed
            self.pending = None
            self.interrupted = True
            raise TimeoutError("issued hard deadline exceeded; work remains charged")
        event = dict(
            type="complete",
            issued_id=self.pending["issued_id"],
            elapsed_ns=elapsed,
            solved_groups=solved_groups,
        )
        self.log.append(str(len(self.log.events)), event)
        self.charged += elapsed
        if self.pending["kind"] == "diagnostic":
            self.diagnostic_done = True
        elif self.pending["arm"] == "baseline":
            self.pairs.append([solved_groups, None])
        else:
            self.pairs[-1][1] = solved_groups
        self.pending = None
