"""Stdlib reconciliation of frozen policy, remaining slots and producer decisions."""

import json
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    record,
    verify_refs,
)


def frozen_evidence(root, campaign):
    policies, panels, designs = [], {}, {}
    for ref in campaign["payload"]["references"]:
        if ref["kind"] not in {"policy", "groups"}:
            continue
        value = json.loads(verify_refs(root, [ref["artifact"]])[0])
        if value.get("identity") == campaign["payload"]["target_policy"]:
            policies.append(value)
        if value.get("version") == "discovery-cohort/v1":
            panels[value["identity"]] = value
            panels[ref["name"]] = value
        designs[digest(value)] = value
    if len(policies) != 1:
        raise ValueError("unique original frozen target policy required")
    return policies[0], panels, designs


def remaining_slots(journal, campaign, policy):
    graph = campaign["payload"]["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    remaining = {n: dict.fromkeys(journal.binding["caps"], 0) for n in nodes}
    spent = {n: dict.fromkeys(journal.binding["caps"], 0) for n in nodes}
    for plan in campaign["payload"]["tasks"]:
        task = record(
            "task",
            dict(plan, campaign=campaign["identity"]),
            created_at=campaign["created_at"],
            purpose=campaign["purpose"],
            track=campaign["track"],
        )
        left = journal.plan_remaining(plan)
        for k in left:
            remaining[plan["slot"]][k] += left[k]
            spent[plan["slot"]][k] += plan["caps"][k] - left[k]
    pool = {
        k: max(0, cap - sum(spent[n][k] for n in nodes if nodes[n]["kind"] == "diagnostic"))
        for k, cap in graph["diagnostic_pool"].items()
    }
    for branch in policy["payload"]["diagnostics"]:
        remaining[branch["id"]] = {
            k: min(
                pool[k],
                max(0, cap - sum(spent[n][k] for n in branch["task_slots"])),
                sum(remaining[n][k] for n in branch["task_slots"]),
            )
            for k, cap in branch["reservation"].items()
        }
    return remaining


def reconcile_decision(root, role_output, result, store, journal, task):
    ref = result.get("metrics", {}).get("decision_bundle")
    if ref is None:
        return None
    bundle = json.loads(verify_refs(Path(role_output), [ref])[0])
    if bundle["source_slot"] != task["payload"]["slot"]:
        raise ValueError("producer cannot authorize another source slot")
    policy, panels, designs = frozen_evidence(root, store.campaign)
    from oeis_learn.generalization.progressive.contracts import seal
    with journal.lock():
        budget_events = journal._read()[0]
    audit = seal(dict(bundle=bundle['identity'], kwargs=bundle['kwargs'],
                      budget_events=budget_events, prior_decisions=list(store.state.get('decisions',{}).values())))
    audit_path = Path(role_output)/'decision-audit.json'
    certificate = store.authorize(
        bundle,
        policy=policy,
        remaining=remaining_slots(journal, store.campaign, policy),
        panels=panels,
        designs=designs,
        evidence_root=root,
    )
    if audit_path.exists():
        from oeis_learn.generalization.progressive.contracts import load
        if load(audit_path) != audit:
            # Resuming a committed decision preserves its original reservation cut.
            audit = load(audit_path)
    else:
        from oeis_learn.generalization.progressive.storage import register_invocation, owned_guard
        parent=Path(role_output).parent if Path(role_output).name=='role' else Path(role_output)
        proof=register_invocation(root,store.campaign,task,journal.root,parent)
        state=Path(root)/'run-state/discovery-learning'/store.campaign['identity'][7:]/task['identity'][7:]
        with owned_guard(dict(campaign=store.campaign,task=task,storage=proof),output=parent,state_root=state).activate():
            atomic_write(audit_path, audit)
    from oeis_learn.generalization.progressive.milestones import gain_cost

    source = bundle["source_slot"]
    nodes = {n["id"]: n for n in store.campaign["payload"]["graph"]["nodes"]}
    ancestors = set()

    def walk(slot):
        for parent in nodes[slot]["dependencies"]:
            if parent not in ancestors:
                ancestors.add(parent)
                walk(parent)

    walk(source)
    costs, prior = {}, {}
    for plan in store.campaign["payload"]["tasks"]:
        t = record(
            "task",
            dict(plan, campaign=store.campaign["identity"]),
            created_at=store.campaign["created_at"],
            purpose=store.campaign["purpose"],
            track=store.campaign["track"],
        )
        elapsed = plan["caps"]["elapsed_us"] - journal.plan_remaining(plan)["elapsed_us"]
        if plan["slot"] == source:
            costs[plan["id"]] = elapsed
        elif plan["slot"] in ancestors:
            prior[plan["id"]] = elapsed
    atomic_write(
        Path(role_output) / "gain-cost.json",
        gain_cost(
            bundle["decision"],
            bundle["rows"],
            bundle["roster"],
            bundle["seeds"],
            bundle["design"],
            costs=costs,
            ancestor_costs=prior,
        ),
    )
    return certificate
