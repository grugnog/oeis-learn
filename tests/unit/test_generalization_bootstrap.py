"""Independent acceptance dependency and legacy initializer counterexamples."""
import copy,json
import pytest
from oeis_learn.pipeline_validation.identity import seal,digest,write


def pending_pair():
    from tests.unit.test_generalization_convergence import execution_fixture
    from oeis_learn.generalization.bootstrap import pending
    from oeis_learn.generalization.execution import bind
    s,r=execution_fixture();s.update(recipe='P4-replay/v1',arms=['continued_sft','verified_replay'],changing_factors=['verified_replay'])
    s['progression']['stage']='P4';s['model']['initialization']=pending(digest('frozen-e0'))
    r['training']={a:dict(cells=['static'],data_mode='static',batch_size=4) for a in s['arms']}
    r['replay']=dict(cell='static',seconds=30);r['evaluation']=dict(role='selection',cells=['selection'],checkpoints=[],seconds=30,bank_seeds=None)
    return bind(s,r)


def test_pending_is_acceptance_only_and_never_a_learner_initializer(tmp_path):
    from oeis_learn.generalization.execution import validate_policy
    from oeis_learn.generalization.initializers import verify_initializer
    from oeis_learn.generalization.contracts import validate
    s,r=pending_pair();validate_policy(s,r);validate('study',s)
    with pytest.raises(ValueError,match='unresolved'):verify_initializer(tmp_path,s)
    assert verify_initializer(tmp_path,s,allow_pending=True) is None
    with pytest.raises(ValueError,match='acceptance'):
        validate_policy(s,dict(r,purpose='research'))


def test_dependency_transition_changes_only_initializer_and_binding():
    from oeis_learn.generalization.bootstrap import verify_transition
    from oeis_learn.generalization.execution import bind
    s,r=pending_pair();resolved=copy.deepcopy(s)
    resolved['model']['initialization']=dict(kind='retained',artifact=s['source_closure'])
    resolved,rr=bind(resolved,r)
    verify_transition(s,r,resolved,rr)
    changed=copy.deepcopy(resolved);changed['budgets']['target_tokens']+=1;changed,cr=bind(changed,rr)
    with pytest.raises(ValueError,match='choices'):verify_transition(s,r,changed,cr)
    with pytest.raises(ValueError,match='unresolved'):verify_transition(s,r,s,r)


def legacy_authority(root):
    from tests.integration.test_generalization_inputs import legacy_inputs,import_uniform
    from tests.unit.test_generalization_initializer import fixture
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    legacy_inputs(root);import_uniform(root,bundle='run.zip',metadata='metadata.zip',weights='train',output='imported')
    base=root/'imported';target,authority,old,put=fixture(base/'new')
    checkpoint=json.loads((base/'best-checkpoint.json').read_text())
    # Keep all original relative legacy references and bytes together.
    import shutil
    shutil.copytree(base/'provenance',base/'new/provenance');shutil.copytree(base/'private',base/'new/private')
    shutil.copyfile(base/'best-provenance.json',base/'new/best-provenance.json')
    source=json.loads((base/'new/study.json').read_text());source.update(version='generalization-study/v3',recipe='E0-evaluation-extension/v1',execution=digest('execution'))
    source['arms']=['retained','controls','shared_absent_bank'];source['progression']['stage']='E0';source['budgets']['target_tokens']=0;source['checkpoints']['milestones']=[]
    recipe=json.loads((base/'new/provenance/recipe.json').read_text());source['model'].update(profile=recipe.get('version','legacy-test/v1'),architecture=digest({k:v for k,v in recipe['model'].items() if k!='seed'}))
    source=seal(source);authority['study']=put('study.json',source)
    selection=json.loads((base/'new/selection.json').read_text());selection.update(selected=checkpoint['identity'],eligible=[checkpoint['identity']],best_loss=checkpoint['identity'])
    selection['candidates'][0]['checkpoint']=checkpoint;selection['measurements'][0]['checkpoint']=checkpoint['identity'];selection['measurements'][0]['behavior']['id']=checkpoint['identity']
    authority.update(version='generalization-initializer/v2',checkpoint=put('checkpoint.json',checkpoint),selection=put('selection.json',seal(selection)),legacy=checkpoint['provenance'])
    producer=json.loads((base/'new/receipt.json').read_text());producer['binding'].update(study=source['identity'],checkpoint=checkpoint['identity'])
    producer['files']={str(p.relative_to(base/'new')):reference(p,base/'new')['sha256'] for p in (base/'new').rglob('*') if p.is_file() and p.name not in {'receipt.json','initializer.json'}}
    authority['producer']=put('receipt.json',seal(producer));target['model'].update(profile=source['model']['profile'],architecture=source['model']['architecture'],initialization=dict(kind='retained',artifact=put('initializer.json',seal(authority))))
    return base/'new',seal(target),authority,put


def test_new_selection_can_initialize_from_verified_legacy_but_not_resume(tmp_path):
    from oeis_learn.generalization.initializers import verify_initializer
    root,s,_,_=legacy_authority(tmp_path)
    result=verify_initializer(root,s)
    assert result['weights']['path']=='private/best-weights.pt'
    original=json.loads((root/'checkpoint.json').read_text())
    assert original['kind']=='weights_only' and original['state'] is None


@pytest.mark.parametrize('change',['history','selection','model','producer'])
def test_legacy_initialization_rejects_changed_evidence(tmp_path,change):
    from oeis_learn.generalization.initializers import verify_initializer
    root,s,a,put=legacy_authority(tmp_path)
    if change=='history':(root/'provenance/history.json').write_text('[]')
    elif change=='selection':
        d=json.loads((root/'selection.json').read_text());d['selected']=digest('invented');a['selection']=put('selection.json',seal(d))
    elif change=='model':s['model']['architecture']=digest('other')
    else:
        d=json.loads((root/'receipt.json').read_text());d['state']='REUSED';a['producer']=put('receipt.json',seal(d))
    s['model']['initialization']['artifact']=put('initializer.json',seal(a))
    with pytest.raises(ValueError):verify_initializer(root,seal(s))
