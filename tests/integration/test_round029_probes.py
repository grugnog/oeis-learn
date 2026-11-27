import pytest
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.graph import velocity_look
from oeis_learn.pipeline_validation.progressive_evaluation import velocity_outcome


def test_profiles_and_final_selected_outcome_have_separate_identities():
    cells = [dict(task=str(i), group=str(i // 2), stratum="oeis") for i in range(6)]
    retain = [dict(task="ret", group="retention", stratum="retention")]
    args = dict(
        registration=digest("study"), method="00", seed=1, development=cells, retention=retain
    )
    cheap = [velocity_look(**args, look=k, profile="common") for k in (0, 1000000, 4000000)]
    assert all(r["members"] == cells[::2] and r["attempts"] == 1 for r in cheap)
    full = velocity_look(**args, look=4000000, profile="full")
    assert full["attempts"] == 8 and full["members"] == cells + retain
    assert full["identity"] != cheap[-1]["identity"]
    with pytest.raises(ValueError):
        velocity_look(**args, look=1000000, profile="full")
    evidence = dict(case="0", outcome_ref=digest("score"), choice_ref=digest("selected"),
                    scorer_ref=digest("independent reference"), truth_ref=digest("exact indexed truth"))
    observation = velocity_outcome(selected_exact=True, closed=False, complete=False, evidence=evidence)
    assert observation["status"] == "solve" and not observation["closed"]
    receipt = observation["receipt"]
    assert all(receipt[k] == v for k, v in evidence.items()) and receipt["status"] == observation["status"]
    assert velocity_outcome(selected_exact=None, closed=True, complete=True)["status"] == "miss"
    assert (
        velocity_outcome(selected_exact=None, closed=False, complete=False)["status"]
        == "unresolved"
    )
