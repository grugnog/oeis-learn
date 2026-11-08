"""Negative action contracts; later third-seed routing stays an explicit specification."""

import importlib.util

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


@pytest.mark.xfail(
    importlib.util.find_spec("oeis_learn.generalization.progressive.contingencies") is None,
    strict=True,
    reason="T053 third-seed prospective routing owner",
)
def test_third_seed_reports_all_seeds_and_cannot_erase_original_gate():
    from oeis_learn.generalization.progressive.contingencies import third_seed

    assert (
        third_seed(
            effects=[0.04, -0.01, 0.04],
            completion=[1, 1, 1],
            retention=[0, 0, 0],
            lower=0.001,
            original_verdict="INCONCLUSIVE",
            registered=True,
            comparisons=2,
        )["original_verdict"]
        == "INCONCLUSIVE"
    )
    with pytest.raises(ValueError):
        third_seed(
            effects=[0.05, -0.03, 0.05],
            completion=[1, 1, 1],
            retention=[0, 0, 0],
            lower=0.001,
            original_verdict="INCONCLUSIVE",
            registered=False,
            comparisons=1,
        )
