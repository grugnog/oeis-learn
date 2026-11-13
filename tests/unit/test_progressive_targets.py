import pytest
from oeis_learn.generalization.progressive.targets import defaults, validate_policy, validate_action
from oeis_learn.pipeline_validation.identity import seal


def test_defaults_are_proposed_and_invariants_cannot_be_softened():
    policy = defaults()
    assert policy["payload"]["selected"] is False
    p = policy["payload"]
    p["targets"][0]["on_miss"] = "RETAIN_BASELINE"
    with pytest.raises(ValueError):
        validate_policy(seal(policy))


def test_unreserved_diagnostics_and_invalid_evidence_cannot_advance():
    p = defaults()
    t = p["payload"]["targets"][0]
    with pytest.raises(ValueError):
        validate_action(p, t["id"], verdict="INVALID", action="RETAIN_BASELINE", remaining={})
    t = p["payload"]["targets"][5]
    assert (
        validate_action(p, t["id"], verdict="INCOMPLETE", action="RETAIN_BASELINE", remaining={})
        == "RETAIN_BASELINE"
    )
    t["allowed_actions"].append("EXPLORE_RESERVED")
    with pytest.raises(ValueError):
        validate_policy(seal(p))
