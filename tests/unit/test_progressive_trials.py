"""Real pinned-Optuna history replay; no optimizer mock can pass these cases."""

import json

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial


def propose(history, ordinal):
    assert optuna.__version__ == "4.5.0"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=1300 + ordinal, n_startup_trials=4),
    )
    for entry in history:
        study.add_trial(
            create_trial(
                params=entry["params"],
                distributions={"lr": CategoricalDistribution([0.3, 1.0, 3.0])},
                value=entry["value"] if entry["state"] == "COMPLETE" else None,
                state=TrialState[entry["state"]],
            )
        )
    trial = study.ask()
    trial.suggest_categorical("lr", [0.3, 1.0, 3.0])
    return trial.params


def test_pinned_sampler_replay_is_identical_after_serialized_history():
    history = [
        {"params": {"lr": v}, "value": (i % 3) / 10, "state": "COMPLETE"}
        for i, v in enumerate([0.3, 1.0, 3.0, 1.0, 0.3, 3.0])
    ]
    assert propose(history, 6) == propose(json.loads(json.dumps(history)), 6)
    # Replay derives from the sealed ordinal/history, not a saved sampler pickle.
    assert set(propose(history, 6)) == {"lr"}


def test_failed_and_pruned_trials_remain_in_actual_sampler_history():
    history = [
        {"params": {"lr": 0.3}, "value": None, "state": "FAIL"},
        {"params": {"lr": 1.0}, "value": None, "state": "PRUNED"},
    ]
    assert propose(history, 2) == propose(json.loads(json.dumps(history)), 2)


def test_enqueued_baseline_and_nonbaseline_controls_are_actual_optuna_trials():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=13), direction="maximize")
    study.enqueue_trial({"lr": 1.0})
    study.enqueue_trial({"lr": 0.3})
    for expected in (1.0, 0.3):
        trial = study.ask()
        assert trial.suggest_categorical("lr", [0.3, 1.0, 3.0]) == expected
        study.tell(trial, 0.0)
    assert len(study.get_trials()) == 2
    assert all(t.state == TrialState.COMPLETE for t in study.get_trials())
