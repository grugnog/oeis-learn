import copy

import pytest

from oeis_learn.generalization.progressive.decisions import publish
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.pipeline_validation.identity import seal
from tests.helpers.progressive_fixture import caps
from tests.unit.test_progressive_decisions import fixture
from tests.unit.test_progressive_freeze import campaign as base_campaign


def configured(tmp_path):
    data, roster, design, policy = fixture()
    target = next(t for t in policy["payload"]["targets"] if t["id"] == "gain")
    target["allowed_actions"].append("ADVANCE_REGISTERED")
    target["budget_ref"] = "diversity"
    policy = seal(policy)
    c = base_campaign()
    plans = c["payload"]["tasks"]
    child = plans[0]
    parents = []
    for seed in (1, 2):
        p = copy.deepcopy(child)
        p.update(
            id=f"parent-{seed}",
            slot="capability",
            stage="capability",
            seed=seed,
            output=f"parent-{seed}",
        )
        parents.append(p)
    c["payload"]["tasks"] = parents + [child]
    c["payload"]["graph"]["nodes"][0]["caps"] = {k: 2 * v for k, v in caps().items()}
    c["payload"]["graph"]["edges"][0]["condition"] = "CONFIRMED_GAIN"
    c["payload"]["target_policy"] = policy["identity"]
    c = seal(c)
    store = RunStore(tmp_path, c)
    store.append(
        {
            "tasks": {"parent-1": "COMPLETE", "parent-2": "COMPLETE"},
            "status": "COMPLETE",
            "invocation": "parents",
        }
    )
    remaining = {"diversity": caps()}
    decision = publish(
        data,
        roster,
        [1, 2],
        design,
        policy=policy,
        target_id="gain",
        action="ADVANCE_REGISTERED",
        remaining=remaining,
        created_at=policy["created_at"],
        purpose="acceptance",
        track="synthetic_strict",
    )
    bundle = seal(
        {
            "version": "progressive-decision-bundle/v1",
            "source_slot": "capability",
            "target_slot": "diversity",
            "decision": decision,
            "rows": data,
            "roster": roster,
            "seeds": [1, 2],
            "design": design,
            "kwargs": {"target_id": "gain", "action": "ADVANCE_REGISTERED"},
        }
    )
    return store, bundle, policy, remaining


def test_only_recomputed_frozen_decision_advances_scientific_edge(tmp_path):
    store, bundle, policy, remaining = configured(tmp_path)
    assert store.next_task() is None
    certificate = store.authorize(bundle, policy=policy, remaining=remaining)
    assert store.next_task()["payload"]["slot"] == "diversity"
    assert (
        store.authorize(bundle, policy=policy, remaining={}) == certificate
    )  # replay, no new action
    forged = copy.deepcopy(bundle)
    forged["rows"][0]["outcome"] = (
        "success" if forged["rows"][0]["outcome"] == "failure" else "failure"
    )
    with pytest.raises(ValueError):
        store.authorize(seal(forged), policy=policy, remaining=remaining)


def test_changed_seed_or_policy_cannot_buy_an_advance(tmp_path):
    store, bundle, policy, remaining = configured(tmp_path)
    changed = copy.deepcopy(bundle)
    changed["seeds"] = [1, 3]
    with pytest.raises(ValueError, match="seed"):
        store.authorize(seal(changed), policy=policy, remaining=remaining)
    altered = seal(dict(policy, payload=dict(policy["payload"], selected=False)))
    with pytest.raises(ValueError):
        store.authorize(bundle, policy=altered, remaining=remaining)
