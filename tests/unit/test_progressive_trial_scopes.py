import copy
import pytest
from oeis_learn.generalization.progressive.trial_scopes import resolve
from oeis_learn.generalization.progressive.tuning import BASELINE
from oeis_learn.generalization.progressive.contracts import digest, seal, record
from oeis_learn.generalization.progressive.campaign import validate_task
from oeis_learn.generalization.progressive.lifecycle import RunStore
from tests.unit.test_progressive_continuation import campaign


def test_finite_choice_changes_scope_without_changing_task_allowance(tmp_path):
    c = campaign()
    p = c["payload"]["tasks"][0]
    p["trial"] = dict(study=digest("study"), ordinal=0, phase="training", coordinate=0)
    p["trial_choices"] = [
        dict(
            parameters=BASELINE, compatibility=p["initializer"]["compatibility"], policy=p["policy"]
        ),
        dict(
            parameters=dict(BASELINE, dropout=0.1),
            compatibility=digest("other-compatible-profile"),
            policy=digest("other-request"),
        ),
    ]
    c = seal(c)
    store = RunStore(tmp_path, c)
    assert store.resolved_plan(p) is None
    store.choose_trial(
        seal(
            dict(
                study=digest("study"),
                ordinal=0,
                seed=p["seed"],
                parameters=dict(BASELINE, dropout=0.1),
            )
        )
    )
    chosen = store.resolved_plan(p)
    assert chosen["caps"] == p["caps"] and chosen["token_target"] == p["token_target"]
    task = record(
        "task",
        dict(chosen, campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    validate_task(task, campaign=c)
    bad = copy.deepcopy(task)
    bad["payload"]["caps"]["tokens"] += 1
    with pytest.raises(ValueError):
        validate_task(seal(bad), campaign=c)
    with pytest.raises(ValueError, match="cannot change"):
        store.choose_trial(
            seal(dict(study=digest("study"), ordinal=0, seed=p["seed"], parameters=BASELINE))
        )
