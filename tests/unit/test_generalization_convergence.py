"""Independent counterexamples from the reconstruction requirements audit."""
import copy
import pytest
from oeis_learn.generalization.selection import select
from oeis_learn.generalization.targets import pair


def test_selection_cannot_rank_different_frozen_denominators():
    a=dict(id='a',tokens=1,role='selection',structural=({'s':'f'},{'s':True}),parameter=({'p':'g'},{'p':True}))
    b=copy.deepcopy(a);b['id']='b';b['structural']=({'other':'f'},{'other':True})
    with pytest.raises(ValueError,match='same frozen'):
        select([a,b])


def test_selection_cannot_change_family_weighting_between_candidates():
    a=dict(id='a',tokens=1,role='selection',structural=({'s':'f','t':'f'},{'s':True,'t':False}),parameter=({'p':'g'},{'p':True}))
    b=copy.deepcopy(a);b['id']='b';b['structural'][0]['t']='other'
    with pytest.raises(ValueError,match='same frozen'):
        select([a,b])


def test_unsupported_control_target_stays_literal_and_counted():
    from oeis_learn.generalization.targets import learning_target
    literal='(program (return (select (lt $index (int 2)) (int 1) (int 0))))'
    parameter='(program (return (select (lt $index (param.get p0)) (int 1) (int 0))))'
    proof=pair(literal,parameter,{'p0':'2'},indices=tuple(map(str,range(100))),sharing='explicit')
    assert proof['disposition']=='unsupported_control_dependency'
    assert learning_target(proof)==(literal,proof['literal_tokens'])


def test_supported_parameter_target_is_preserved():
    from oeis_learn.generalization.targets import learning_target
    literal='(program (return (add $index (int 2))))'
    parameter='(program (return (add $index (param.get p0))))'
    proof=pair(literal,parameter,{'p0':'2'},indices=tuple(map(str,range(100))),sharing='explicit')
    assert learning_target(proof)==(parameter,proof['parameter_tokens'])


def execution_fixture():
    import json
    from pathlib import Path
    from oeis_learn.pipeline_validation.identity import seal,digest
    s=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())['study']
    s.update(version='generalization-study/v3')
    s['data']['sharing_policy']='explicit'
    s['budgets'].update(target_tokens=100,seconds=30)
    s['checkpoints']['milestones']=[100]
    r=dict(version='generalization-runtime/v2',purpose='acceptance',recipe=s['source_closure'],images=['sha256:'+'1'*64,'sha256:'+'2'*64],
        preparation=None,training={'static':dict(cells=['static'],data_mode='static',batch_size=1),
        'refreshed':dict(cells=['static','fresh'],data_mode='refresh',batch_size=1)},evaluation=None,replay=None,assessment=None,feature=None,
        storage=dict(study_bytes=64<<20,checkpoint_bytes=32<<20,shard_bytes=8<<20,raw_bytes=4<<20,sharing_bytes=1<<20))
    from oeis_learn.generalization.execution import bind
    return bind(s,r)


def test_full_runtime_recipe_and_choices_are_bound_to_study():
    from oeis_learn.generalization.execution import verify_binding
    from oeis_learn.pipeline_validation.identity import seal,digest
    s,r=execution_fixture();verify_binding(s,r)
    for key in ('recipe','training','purpose'):
        changed=copy.deepcopy(r)
        if key=='recipe':changed[key]['sha256']=digest('changed optimizer or learning rate')
        elif key=='training':changed[key]['static']['batch_size']=2
        else:changed[key]='research'
        with pytest.raises(ValueError,match='execution binding'):
            verify_binding(s,seal(changed))


def test_acceptance_case_cannot_substitute_a_different_experiment():
    from oeis_learn.generalization.execution import validate_policy
    s,r=execution_fixture()
    validate_policy(s,r,case='refresh')
    with pytest.raises(ValueError,match='case'):
        validate_policy(s,r,case='parameter-targets')
    r['training']['refreshed']['batch_size']=2
    with pytest.raises(ValueError,match='paired batch'):
        validate_policy(s,r,case='refresh')


def test_e0_has_no_training_or_fictitious_checkpoint_milestones():
    from oeis_learn.generalization.execution import bind,validate_policy
    s,r=execution_fixture();s['progression']['stage']='E0';s['recipe']='E0-evaluation-extension/v1'
    s['arms']=['retained'];s['changing_factors']=['evaluation'];s['budgets']['target_tokens']=0;s['checkpoints']['milestones']=[]
    r['training']={};r['evaluation']=dict(role='selection',cells=['selection'],checkpoints=[s['source_closure']],seconds=60,bank_seeds=None)
    s,r=bind(s,r);validate_policy(s,r,case='retained-eval')
    r['training']['retained']=dict(cells=['training'],data_mode='static',batch_size=1)
    with pytest.raises(ValueError,match='evaluation-only'):
        validate_policy(s,r)


def test_refresh_cannot_change_family_mixture_or_reuse_duplicate_sources():
    from oeis_learn.generalization.execution import validate_refresh
    static=[dict(id='a',family='f',source='a'),dict(id='b',family='g',source='b')]
    fresh=static+[dict(id='c',family='f',source='c')]
    with pytest.raises(ValueError,match='mixture'):
        validate_refresh(static,fresh)
    fresh.append(dict(id='d',family='g',source='d'));validate_refresh(static,fresh)
    fresh[-1]['source']='b'
    with pytest.raises(ValueError,match='duplicate'):
        validate_refresh(static,fresh)
