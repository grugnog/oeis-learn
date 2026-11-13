"""Prospective T053 third-seed routing counterexample; no execution authority."""

import importlib.util

import pytest


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
