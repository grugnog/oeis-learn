"""Target classes and bounded actions remain independent of observed verdicts."""

import json
from importlib.resources import files

from .contracts import validate

NO_WORK = {"RETAIN_BASELINE", "DEFER_BRANCH", "STOP_AFFECTED"}
VERDICTS = (
    "INVALID",
    "INCOMPLETE",
    "REGRESSION",
    "CONFIRMED_GAIN",
    "CONFIRMED_EFFICIENCY",
    "NO_MATERIAL_GAIN",
    "INCONCLUSIVE",
)


def defaults():
    return validate_policy(
        json.loads(
            files("oeis_learn.generalization")
            .joinpath("policies", "progressive-targets.json")
            .read_text()
        )
    )


def validate_policy(policy, *, reservations=None):
    p = validate("target-policy", policy)["payload"]
    if reservations is not None:
        reservations = dict(reservations)
        for branch in p["diagnostics"]:
            if any(slot not in reservations for slot in branch["task_slots"]):
                raise ValueError("unreserved diagnostic slots")
            if branch["id"] not in reservations:
                reservations[branch["id"]] = branch["reservation"]
    if p["verdict_precedence"] != list(VERDICTS):
        raise ValueError("frozen verdict precedence required")
    ids = [t["id"] for t in p["targets"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate target")
    for t in p["targets"]:
        if t["class"] == "integrity_invariant" and (
            t["on_miss"] != "STOP_AFFECTED" or t["allowed_actions"] != ["STOP_AFFECTED"]
        ):
            raise ValueError("invariants cannot be softened")
        if any(a not in NO_WORK for a in t["allowed_actions"]) and not t["budget_ref"]:
            raise ValueError("action needs frozen reservation")
        if (
            reservations is not None
            and t["budget_ref"] is not None
            and t["budget_ref"] not in reservations
        ):
            raise ValueError("unresolved target reservation")
    failures = set()
    slots = set()
    for d in p["diagnostics"]:
        if (d['signal']=='runtime_bottleneck') != ('runtime_rule' in d):
            raise ValueError('runtime diagnostic requires its frozen measured rule')
        if not d["reservation"]["elapsed_us"]:
            raise ValueError("positive diagnostic reservation required")
        if d["failure_class"] in failures or slots.intersection(d["task_slots"]):
            raise ValueError("recursive or duplicate diagnostic episode")
        failures.add(d["failure_class"])
        slots.update(d["task_slots"])
        if reservations is not None:
            if any(slot not in reservations for slot in d["task_slots"]):
                raise ValueError("unreserved diagnostic")
            if any(
                d["reservation"][k] > sum(reservations[s][k] for s in d["task_slots"])
                for k in d["reservation"]
            ):
                raise ValueError("diagnostic exceeds slots")
    return policy


def validate_action(policy, target_id, *, verdict, action, remaining, trigger=None, visited=()):
    validate_policy(policy)
    if verdict not in VERDICTS:
        raise ValueError("unknown evidence verdict")
    t = next((t for t in policy["payload"]["targets"] if t["id"] == target_id), None)
    if t is None or action not in t["allowed_actions"]:
        raise ValueError("unregistered action")
    if verdict == "INVALID" and action != "STOP_AFFECTED":
        raise ValueError("invalid integrity stops affected operation")
    if action in NO_WORK:
        return action
    if not policy["payload"]["selected"]:
        raise ValueError("proposed defaults are non-executable")
    if action == "ADVANCE_REGISTERED" and verdict not in {"CONFIRMED_GAIN", "CONFIRMED_EFFICIENCY"}:
        raise ValueError("failed gate cannot advance")
    if t["budget_ref"] not in remaining:
        raise ValueError("unreserved action")
    if not remaining[t["budget_ref"]].get("elapsed_us", 0):
        raise ValueError("reserved action has no remaining active time")
    if action == "EXPLORE_RESERVED":
        branch = next(
            (
                d
                for d in policy["payload"]["diagnostics"]
                if d["trigger"] == trigger and d["id"] == t["budget_ref"]
            ),
            None,
        )
        if branch is None or branch["failure_class"] in visited:
            raise ValueError("diagnostic trigger absent or already consumed")
        if any(
            branch["reservation"][k] > remaining[t["budget_ref"]][k] for k in branch["reservation"]
        ):
            raise ValueError("diagnostic exhausted")
    return action
