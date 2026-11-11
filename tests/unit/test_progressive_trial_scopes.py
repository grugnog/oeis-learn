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

@pytest.mark.parametrize('schedule', ['constant', 'warmup_cosine'])
def test_confirmation_policy_survives_scaling_and_rejects_changed_trajectory(tmp_path, schedule):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy
    from oeis_learn.pipeline_validation.progressive_learner import compatibility, validate_input
    from oeis_learn.generalization.progressive.contracts import load

    _, tasks, _, _ = fixture(tmp_path, interrupt=False)
    package = load(tmp_path/'training-input.json')
    package['update_policy'] = resolve_update_policy(package['profile'], schedule_horizon_tokens=400000000,
        schedule=schedule, warmup_tokens=1250000 if schedule == 'warmup_cosine' else 0)
    package = seal(package)
    task = copy.deepcopy(tasks[0])
    task['purpose'] = 'research'
    task['payload']['stage'] = 'scaling'
    task['payload']['initializer']['compatibility'] = compatibility(package)
    # Only input validation is exercised; these diagnostic rows grant no research authority.
    package = seal(dict(package, scope='NATIVE_VERIFIED'))
    package['shards'] = [[seal(dict(r, native_verified=True)) for r in shard] for shard in package['shards']]
    package = seal(package)
    task['payload']['initializer']['compatibility'] = compatibility(package)
    validate_input(package, task)
    bad = seal(dict(package, update_policy=resolve_update_policy(package['profile'],
        schedule_horizon_tokens=25000000, schedule=schedule)))
    with pytest.raises(ValueError, match='400M'):
        validate_input(bad, task)


def test_study_freeze_uses_same_finite_materialization_as_calibration_and_execution(tmp_path,monkeypatch):
    from tests.helpers.progressive_study import build
    from oeis_learn.pipeline_validation.progressive_study import validate_frozen
    from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy
    from oeis_learn.pipeline_validation.progressive_learner import compatibility
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.trial_scopes import materialize
    from oeis_learn.pipeline_validation.progressive_calibration import selected_inputs

    campaign,_,_,_=build(tmp_path,monkeypatch)
    task=next(p for p in campaign['payload']['tasks'] if p['id']=='train')
    original=load(tmp_path/'study-training.json')
    parameters=dict(BASELINE,lr_multiplier=3)
    other=seal(dict(original,update_policy=resolve_update_policy(original['profile'],
        schedule_horizon_tokens=4000000,lr_multiplier=3)))
    task['trial_choices'].append(dict(parameters=parameters,compatibility=compatibility(other),policy=task['policy']))
    campaign=seal(campaign)
    validate_frozen(tmp_path,campaign)
    cases,_=selected_inputs(tmp_path,dict(purpose='acceptance',track=campaign['track'],payload=dict(
        task_plans=campaign['payload']['tasks'],references=campaign['payload']['references'])))
    assert len(cases)==2
    assert materialize(original,resolve(task,1))==other
    assert load(tmp_path/'study-training.json')==original
    bad=copy.deepcopy(campaign)
    next(p for p in bad['payload']['tasks'] if p['id']=='train')['trial_choices'][1]['compatibility']=digest('foreign')
    with pytest.raises(ValueError,match='compatibility'):
        validate_frozen(tmp_path,seal(bad))
