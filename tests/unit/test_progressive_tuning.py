import pytest

from oeis_learn.generalization.progressive.selection import confirmation, rank
from oeis_learn.generalization.progressive.trials import TrialJournal
from oeis_learn.generalization.progressive.tuning import BASELINE, preflight, propose, pruning


def test_serial_actual_optuna_replay_controls_and_all_outcomes(tmp_path):
    j = TrialJournal(tmp_path, study="sha256:" + "a" * 64)
    for i, status in enumerate(
        ["COMPLETE", "PRUNED", "FAILED", "PARTIAL", "INFEASIBLE", "COMPLETE"]
    ):
        p = propose(j, i)
        assert propose(TrialJournal(tmp_path, study=j.binding["study"]), i) == p
        if i == 0:
            assert p["parameters"] == BASELINE
        if i == 1:
            assert p["parameters"]["lr_multiplier"] == 0.3
        j.dispatch(i)
        j.tell(
            i,
            {
                "status": status,
                "tokens": 4000000 if status == "COMPLETE" else 1,
                "elapsed_us": 100,
                "value": 0.3 if status == "COMPLETE" else None,
                "role": "development",
            },
        )
    with pytest.raises(ValueError):
        TrialJournal(tmp_path, study=j.binding["study"], token_cap=5000000)


def test_recovery_keeps_remaining_cost_and_terminal_cap(tmp_path):
    j = TrialJournal(tmp_path, study="sha256:" + "a" * 64, token_cap=10, elapsed_cap_us=20)
    j.propose(0, parameters={"lr_multiplier": 1}, seed=13)
    j.reserve(0)
    j.dispatch(0)
    j.interrupt(0, tokens=5, elapsed_us=10)
    assert j.dispatch(0)["tokens"] == 5
    with pytest.raises(ValueError):
        j.interrupt(0, tokens=4, elapsed_us=10)
    assert j.interrupt(0, tokens=10, elapsed_us=20)["status"] == "PARTIAL"
    with pytest.raises(ValueError):
        j.dispatch(0)


def test_measured_feasibility_and_unpruned_sizing():
    assert (
        preflight(
            startup_us=1, evaluation_us=[1, 2, 3], training_us=4, finalization_us=1, cap_us=12
        )
        == 12
    )
    with pytest.raises(ValueError):
        preflight(
            startup_us=1, evaluation_us=[1, 2, 3], training_us=5, finalization_us=1, cap_us=12
        )
    refs = [{"coordinate": 1000000, "value": 0.8, "status": "COMPLETE", "role": "development"}] * 5
    assert pruning(ordinal=2, value=0.1, coordinate=1000000, references=refs)
    assert not pruning(ordinal=12, value=0.1, coordinate=1000000, references=refs)


def test_independent_ranking_cost_ties_and_fresh_panels():
    rows = [
        {
            "id": k,
            "value": v,
            "elapsed_us": c,
            "tokens": 4000000,
            "status": "COMPLETE",
            "role": "development",
        }
        for k, v, c in [("baseline", 0.2, 30), ("b", 0.3, 20), ("a", 0.3, 20)]
    ]
    assert rank(rows)["ordered"] == ["a", "b", "baseline"]
    rows.append(dict(rows[0], id="partial", status="PARTIAL", value=1))
    panels = [
        {"identity": str(i), "role": "confirmation", "exposed": False, "groups": [str(i)]}
        for i in range(3)
    ]
    locked = confirmation(rows, baseline="baseline", panels=panels, seeds=[13, 14])
    assert locked["selected"] == ["baseline", "a", "b"] and locked["sampler_feedback"] is False
    panels[0]["exposed"] = True
    with pytest.raises(ValueError):
        confirmation(rows, baseline="baseline", panels=panels, seeds=[13, 14])
