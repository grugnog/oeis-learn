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
