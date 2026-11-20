import copy

import pytest

from oeis_learn.generalization.progressive.decisions import publish
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.pipeline_validation.identity import seal
from tests.helpers.progressive_fixture import caps
from tests.unit.test_progressive_decisions import fixture
from tests.unit.test_progressive_freeze import campaign as base_campaign


def configured(tmp_path, *, bound=True, producer_change=None):
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
    from oeis_learn.generalization.progressive.contracts import atomic_write, reference

    atomic_write(tmp_path / "policy.json", policy)
    c["payload"]["references"][0]["artifact"] = reference(tmp_path, "policy.json")
    if bound:
        bind_producers(tmp_path, c, data, roster, design, producer_change=producer_change)
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


def test_ordinary_gate_requires_original_roster_and_producers(tmp_path):
    store, bundle, policy, remaining = configured(tmp_path, bound=False)
    with pytest.raises(ValueError, match="original.*registration"):
        store.authorize(bundle, policy=policy, remaining=remaining, panels={})


def bind_producers(root, campaign, data, roster, design, *, producer_change=None):
    """Explicit synthetic acceptance producers, never a research receipt."""
    from tests.unit.test_progressive_cohorts import fixture as panel_fixture
    from oeis_learn.generalization.progressive.contracts import atomic_write, reference, digest

    panel, _ = panel_fixture(root / "panels")
    template = panel["payload"]["rows"][0]
    panel["payload"]["strata"] = ["composition", "unseen_structure", "retention"]
    panel["payload"]["rows"] = [
        dict(template, id=r["task"], stratum=r["stratum"], groups=[r["group"]]) for r in roster
    ]
    panel = seal(panel)
    atomic_write(root / "panel.json", panel)
    producers = []
    for seed in (1, 2):
        for arm in ("baseline", "candidate"):
            name = f"{arm}-{seed}"
            atomic_write(root / (name + "-weights.json"), {"diagnostic_weights": name})
            checkpoint = reference(root, name + "-weights.json")
            model = digest(name)
            candidates = seal(
                dict(
                    version="discovery-candidates/v1",
                    checkpoint=checkpoint["sha256"],
                    model=model,
                    records={r["task"]: {} for r in roster},
                )
            )
            if producer_change == "wrong-model" and name == "candidate-1":
                candidates = seal(dict(candidates, checkpoint=digest("other weights")))
            atomic_write(root / (name + "-generated.json"), candidates)
            gen_ref = reference(root, name + "-generated.json")
            outcomes = {
                r["task"]: {
                    "equal_budget": dict(
                        success=r["outcome"] == "success",
                        complete=r["outcome"] != "unresolved",
                        outcome=r["outcome"],
                    )
                }
                for r in data
                if r["seed"] == seed and r["arm"] == arm
            }
            if producer_change == "missing-row" and name == "candidate-1":
                del outcomes[next(iter(outcomes))]
            scores = seal(
                dict(
                    version="discovery-scoring/v1",
                    candidates=gen_ref["sha256"],
                    outcomes=outcomes,
                    denominator=len(roster),
                )
            )
            atomic_write(root / (name + "-scores.json"), scores)
            producers.append(
                dict(
                    arm=arm,
                    seed=seed,
                    checkpoint=checkpoint,
                    model=model,
                    candidates=gen_ref,
                    scores=reference(root, name + "-scores.json"),
                )
            )
    registration = seal(
        dict(
            version="progressive-gate-registration/v1",
            source_slot="capability",
            target_slot="diversity",
            panels=[panel["identity"]],
            seeds=[1, 2],
            design=digest(design),
            producers=producers,
        )
    )
    atomic_write(root / "gate.json", registration)
    campaign["payload"]["references"].extend(
        [
            dict(
                name="panel",
                kind="groups",
                role="custodian",
                artifact=reference(root, "panel.json"),
            ),
            dict(
                name="gate", kind="policy", role="verifier", artifact=reference(root, "gate.json")
            ),
        ]
    )


@pytest.mark.parametrize("change", ["rename", "filter", "outcome"])
def test_recomputed_replacement_tables_cannot_advance(tmp_path, change):
    store, bundle, policy, remaining = configured(tmp_path)
    if change == "rename":
        for r in bundle["rows"] + bundle["roster"]:
            r["task"] = "UNREGISTERED/" + r["task"]
    elif change == "filter":
        member = bundle["roster"].pop()["task"]
        bundle["rows"] = [r for r in bundle["rows"] if r["task"] != member]
    else:
        bundle["rows"][0]["outcome"] = "success"
    bundle["decision"] = publish(
        bundle["rows"],
        bundle["roster"],
        bundle["seeds"],
        bundle["design"],
        policy=policy,
        remaining=remaining,
        created_at=policy["created_at"],
        purpose="acceptance",
        track="synthetic_strict",
        **bundle["kwargs"],
    )
    with pytest.raises(ValueError, match="roster|scorer producers"):
        store.authorize(seal(bundle), policy=policy, remaining=remaining)
    assert not store.state["decisions"]


def test_wrong_checkpoint_producer_cannot_advance(tmp_path):
    store, bundle, policy, remaining = configured(tmp_path, producer_change="wrong-model")
    with pytest.raises(ValueError, match="provenance"):
        store.authorize(bundle, policy=policy, remaining=remaining)


def test_missing_score_retains_original_denominator(tmp_path):
    from oeis_learn.generalization.progressive.decisions import verify_gate_producers

    store, bundle, policy, remaining = configured(tmp_path, producer_change="missing-row")
    with pytest.raises(ValueError, match="scorer producers"):
        store.authorize(bundle, policy=policy, remaining=remaining)
    missing = next(r for r in bundle["rows"] if r["arm"] == "candidate" and r["seed"] == 1)
    bundle["rows"].remove(missing)
    verify_gate_producers(tmp_path, store.campaign, bundle)
    from oeis_learn.generalization.progressive.metrics import _paired

    aligned, _, _ = _paired(bundle["rows"], bundle["roster"], bundle["seeds"])
    assert len(aligned) == 4 * len(bundle["roster"])
    assert sum(r["outcome"] == "unresolved" for r in aligned) == 1


@pytest.mark.parametrize("wrong_model", [False, True])
def test_installed_publication_and_host_reconciliation_bind_producers(tmp_path, wrong_model):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.contracts import atomic_write, record
    from oeis_learn.pipeline_validation.progressive_role import execute
    from oeis_learn.pipeline_validation.progressive_decision_control import reconcile_decision

    store, bundle, policy, remaining = configured(
        tmp_path, producer_change="wrong-model" if wrong_model else None
    )
    request = dict(
        version="progressive-decision-input/v1",
        **{
            k: bundle[k]
            for k in ("source_slot", "target_slot", "rows", "roster", "seeds", "design", "kwargs")
        },
        policy=policy,
        remaining=remaining,
    )
    atomic_write(
        tmp_path / "inputs/public" / ("a" * 64 + ".json"),
        seal(
            dict(
                version="discovery-channel/v1",
                role="verifier",
                kind="public_evidence",
                data=request,
            )
        ),
    )
    output = tmp_path / "published"
    result = execute("publish", inputs=tmp_path / "inputs", output=output)
    journal = BudgetJournal(
        tmp_path / "budgets",
        campaign=store.campaign["identity"],
        caps=store.campaign["payload"]["caps"],
    )
    task = record(
        "task",
        dict(store.campaign["payload"]["tasks"][0], campaign=store.campaign["identity"]),
        created_at=store.campaign["created_at"],
        purpose=store.campaign["purpose"],
        track=store.campaign["track"],
    )
    if wrong_model:
        with pytest.raises(ValueError, match="provenance"):
            reconcile_decision(tmp_path, output, result, store, journal, task)
        assert not store.state["decisions"]
    else:
        cert = reconcile_decision(tmp_path, output, result, store, journal, task)
        assert cert["action"] == "ADVANCE_REGISTERED"
        assert (output / "gain-cost.json").is_file()
