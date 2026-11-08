"""Negative action contracts; later third-seed routing stays an explicit specification."""

import pytest

from oeis_learn.generalization.progressive.targets import (
    VERDICTS,
    defaults,
    validate_action,
    validate_policy,
)
from oeis_learn.pipeline_validation.identity import seal


@pytest.mark.parametrize("verdict", VERDICTS)
def test_evidence_never_grants_unreserved_action(verdict):
    p = defaults()
    t = p["payload"]["targets"][5]
    with pytest.raises(ValueError):
        validate_action(p, t["id"], verdict=verdict, action="ADVANCE_REGISTERED", remaining={})


def test_no_integrity_waiver_or_recursive_diagnostic():
    p = defaults()
    p["payload"]["targets"][0]["allowed_actions"] = ["RETAIN_BASELINE"]
    with pytest.raises(ValueError):
        validate_policy(seal(p))


def test_fifth_pool_task_and_recursive_diagnostic_are_rejected():
    from oeis_learn.generalization.progressive.graph import validate_graph
    from tests.helpers.progressive_fixture import caps, graph

    g = graph()
    ceiling = caps(**{k: 200 * v for k, v in caps().items()})
    g["diagnostic_pool"] = caps(**{k: 10 * v for k, v in caps().items()})
    for i in range(4):
        g["nodes"].append(
            {
                "id": f"d{i}",
                "stage": "diversity",
                "kind": "diagnostic",
                "dependencies": ["capability"],
                "maximum_visits": 1,
                "caps": caps(),
            }
        )
        g["edges"].append(
            {
                "source": "capability",
                "target": f"d{i}",
                "condition": "INCONCLUSIVE",
                "action": "EXPLORE_RESERVED",
            }
        )
    validate_graph(g, caps=ceiling)
    g["nodes"].append(dict(g["nodes"][-1], id="d4"))
    g["edges"].append(dict(g["edges"][-1], target="d4"))
    with pytest.raises(ValueError, match="diagnostic task cap"):
        validate_graph(g, caps=ceiling)
    g["nodes"].pop()
    g["edges"].pop()
    g["nodes"][-1]["dependencies"] = ["capability", "d0"]
    g["edges"][-1]["source"] = "d0"
    with pytest.raises(ValueError, match="recursive"):
        validate_graph(g, caps=ceiling)


def third_seed_fixture():
    import copy

    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.generalization.progressive.decisions import publish
    from tests.helpers.progressive_fixture import caps
    from tests.unit.test_progressive_decisions import fixture

    data, roster, design, policy = fixture()
    for row in data:
        if row["seed"] == 2 and row["arm"] == "candidate" and row["stratum"] != "retention":
            row["outcome"] = "failure"
    design = design | {"comparisons": 2}
    extra = dict(design)
    registration = {
        "version": "progressive-third-seed-design/v1",
        "branch": "d",
        "original_seeds": [1, 2],
        "third_seed": 3,
        "original_design": digest(design),
        "confirmation_design": digest(extra),
    }
    gain = next(t for t in policy["payload"]["targets"] if t["id"] == "gain")
    gain["evidence"].append(digest(design))
    gain.update(allowed_actions=["RETAIN_BASELINE", "EXPLORE_RESERVED"], budget_ref="d")
    target = copy.deepcopy(gain)
    target.update(
        id="all-three",
        allowed_actions=["RETAIN_BASELINE", "ADVANCE_REGISTERED"],
        budget_ref="next",
        evidence=[digest(extra)],
    )
    policy["payload"]["targets"].append(target)
    policy["payload"]["diagnostics"] = [
        {
            "id": "d",
            "failure_class": "seed-disagreement",
            "trigger": "paired_seed_disagreement",
            "signal": "seed_disagreement",
            "design": digest(registration),
            "task_slots": ["diagnostic"],
            "panel": "heldout",
            "reservation": caps(),
            "maximum_visits": 1,
            "terminal_action": "RETAIN_BASELINE",
            "assessment": False,
        }
    ]
    policy = seal(policy)
    kwargs = {
        "target_id": "gain",
        "action": "EXPLORE_RESERVED",
        "trigger": "paired_seed_disagreement",
    }
    remaining = {"d": caps(), "next": caps()}
    original = publish(
        data,
        roster,
        [1, 2],
        design,
        policy=policy,
        remaining=remaining,
        created_at=policy["created_at"],
        purpose=policy["purpose"],
        track=policy["track"],
        **kwargs,
    )
    assert original["payload"]["evidence_verdict"] == "INCONCLUSIVE"
    confirmation = {
        "registration": registration,
        "original": {
            "decision": original,
            "rows": copy.deepcopy(data),
            "roster": roster,
            "seeds": [1, 2],
            "design": design,
            "kwargs": kwargs,
        },
    }
    data += [dict(r, seed=3) for r in data if r["seed"] == 1]
    return data, roster, extra, policy, remaining, confirmation


def test_third_seed_authority_recomputes_all_outcomes_and_preserves_failed_gate():
    import copy

    from oeis_learn.generalization.progressive.decisions import publish, verify

    data, roster, design, policy, remaining, confirmation = third_seed_fixture()
    kwargs = {
        "policy": policy,
        "target_id": "all-three",
        "action": "ADVANCE_REGISTERED",
        "remaining": remaining,
        "confirmation": confirmation,
    }
    decision = publish(
        data,
        roster,
        [1, 2, 3],
        design,
        created_at=policy["created_at"],
        purpose=policy["purpose"],
        track=policy["track"],
        **kwargs,
    )
    assert decision["payload"]["evidence_verdict"] == "CONFIRMED_GAIN"
    assert decision["payload"]["seeds"] == [1, 2, 3]
    assert confirmation["original"]["decision"]["payload"]["evidence_verdict"] == "INCONCLUSIVE"
    assert verify(decision, data, roster, [1, 2, 3], design, **kwargs) == decision
    for changed_seeds in ([1, 3], [1, 2, 4]):
        with pytest.raises(ValueError):
            publish(
                data,
                roster,
                changed_seeds,
                design,
                created_at=policy["created_at"],
                purpose=policy["purpose"],
                track=policy["track"],
                **kwargs,
            )
    altered = copy.deepcopy(data)
    altered[0]["outcome"] = "success" if altered[0]["outcome"] == "failure" else "failure"
    with pytest.raises(ValueError, match="original outcomes"):
        publish(
            altered,
            roster,
            [1, 2, 3],
            design,
            created_at=policy["created_at"],
            purpose=policy["purpose"],
            track=policy["track"],
            **kwargs,
        )
    incomplete = [r for r in data if not (r["seed"] == 3 and r["arm"] == "candidate")]
    result = publish(
        incomplete,
        roster,
        [1, 2, 3],
        design,
        created_at=policy["created_at"],
        purpose=policy["purpose"],
        track=policy["track"],
        **(kwargs | {"action": "RETAIN_BASELINE"}),
    )
    assert result["payload"]["evidence_verdict"] == "INCOMPLETE"
