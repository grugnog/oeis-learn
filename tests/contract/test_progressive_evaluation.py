"""Frozen evaluation constraints and identified checkpoint boundary."""

import pytest
from oeis_learn.generalization.progressive.baselines import (
    Portfolio,
    checkpoint_identity,
    conditioning,
    observed_union,
    service_outcome,
)
from oeis_learn.generalization.progressive.cohorts import DEFAULT_PROFILE, validate_profile
from oeis_learn.generalization.progressive.contracts import reference


def prompts():
    return [
        dict(id=str(j), indices=[str(i) for i in range(20)], values=[str(i + j) for i in range(20)])
        for j in range(3)
    ]


def test_base_profile_is_fixed_and_portfolio_cannot_multiply_allowance():
    assert DEFAULT_PROFILE == dict(
        id="bigint-20-100/v1",
        visible=20,
        total=100,
        attempts=8,
        portfolio_us=60000000,
        fit_us=4000000,
        token_cap=1024,
        schedule="drain",
    )
    for key, value in [
        ("visible", 4),
        ("total", 9),
        ("attempts", 9),
        ("schedule", "overlap"),
        ("token_cap", 2048),
    ]:
        with pytest.raises(ValueError):
            validate_profile(DEFAULT_PROFILE | {key: value})
    portfolio = Portfolio(started=0.0)
    for i in range(8):
        assert portfolio.reserve("templates" if i % 2 else "retained_D", now=1.0) == 5.0
    with pytest.raises(TimeoutError):
        portfolio.reserve("templates", now=1.0)
    with pytest.raises(TimeoutError):
        Portfolio(started=0.0).reserve("retained_D", now=60.0)


def test_actual_artifact_identity_and_tampering(tmp_path):
    (tmp_path / "weights.pt").write_bytes(b"identified payload")
    ref = reference(tmp_path, "weights.pt")
    assert checkpoint_identity(tmp_path, ref) == ref["sha256"]
    (tmp_path / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        checkpoint_identity(tmp_path, ref)


def test_shuffled_and_absent_never_receive_witnesses():
    p = prompts()
    shuffled = conditioning(p, "shuffled", seed=13)
    assert all(a["values"] != b["values"] for a, b in zip(p, shuffled))
    assert shuffled == conditioning(p, "shuffled", seed=13)
    assert all(a["values"] is None for a in conditioning(p, "absent", seed=13))
    with pytest.raises(ValueError):
        conditioning([p[0] | {"witness": "private"}], "normal", seed=1)


def test_expired_and_unsupported_results_remain_unresolved():
    assert service_outcome({"success": True}, observed_at=5.0, deadline=4.0) == dict(
        success=False, complete=False, reason="expired_service_result"
    )
    assert (
        service_outcome({"success": True}, observed_at=3.0, deadline=4.0, fitting_supported=False)[
            "complete"
        ]
        is False
    )


def test_observed_success_union_has_no_execution_authority():
    union = observed_union({"learned": {"a"}, "templates": {"b"}})
    assert union["successes"] == ["a", "b"] and union["executable"] is False
