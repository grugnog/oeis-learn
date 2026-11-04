"""Independent retained-initializer authority and lineage counterexamples."""
import copy
import json
from pathlib import Path
import pytest
from oeis_learn.generalization.contracts import freeze
from oeis_learn.pipeline_validation.identity import seal,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference


def fixture(root):
    root.mkdir(exist_ok=True)
    def put(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else json.dumps(value,sort_keys=True).encode())
        return reference(path,root)
    source=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())['study']
    source=freeze('study',source).record()
    weights=put('weights.pt',b'exact frozen initializer')
    checkpoint=seal(dict(version='generalization-checkpoint/v1',kind='full_state',weights=weights,
        state=dict(path='not-needed-private-state.json',sha256=digest('state'),bytes=100),
        initialization=digest('initial random weights'),parent=None,tokens=10,updates=1,pins=['selected'],compatibility=digest('compatibility')))
    behavior=dict(id=checkpoint['identity'],tokens=10,role='selection',
        structural=[{'s':'family-s'},{'s':True}],parameter=[{'p':'family-p'},{'p':True}])
    candidate=dict(checkpoint=checkpoint,selections=put('candidates.json',{}))
    selection=seal(dict(version='generalization-checkpoint-selection/v1',status='PASS',scientific_status='UNMEASURED',
        targets=dict(path='selection-truth-not-mounted.json',sha256=digest('truth'),bytes=10),candidates=[candidate],
        measurements=[dict(checkpoint=checkpoint['identity'],loss=1.,behavior=behavior)],best_loss=checkpoint['identity'],
        selected=checkpoint['identity'],eligible=[checkpoint['identity']],reason='selected'))
    refs=dict(study=put('study.json',source),checkpoint=put('checkpoint.json',checkpoint),selection=put('selection.json',selection))
    ancestry=['mechanical','training','random_initialization']
    producer=seal(dict(version='pipeline-stage/v1',stage='generalization.initializer',attempt='original',state='PASS',reason=None,
        binding=dict(study=source['identity'],checkpoint=checkpoint['identity'],ancestry=ancestry,parent=None),
        files={r['path']:r['sha256'] for r in [*refs.values(),weights]}))
    authority=seal(dict(version='generalization-initializer/v1',**refs,producer=put('receipt.json',producer),ancestry=ancestry,parent=None))
    pointer=put('initializer.json',authority)
    target=copy.deepcopy(source);target.update(version='generalization-study/v2',recipe='P4-replay/v1',arms=['continued_sft','verified_replay'],changing_factors=['verified_replay'])
    target['progression']['stage']='P4';target['model']['initialization']=dict(kind='retained',artifact=pointer)
    target['budgets'].update(target_tokens=4_000_000,seconds=7200);target['checkpoints']['milestones']=[1_000_000,2_000_000,4_000_000]
    return target,authority,checkpoint,put


def test_retained_p4_freezes_and_verifies_exact_weights(tmp_path):
    from oeis_learn.generalization.initializers import verify_initializer
    target,authority,checkpoint,put=fixture(tmp_path)
    study=freeze('study',target).record()
    resolved=verify_initializer(tmp_path,study)
    assert resolved['weights']==checkpoint['weights'] and resolved['authority']==authority['identity']
    assert resolved['checkpoint']==checkpoint['identity']
    (tmp_path/'weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):verify_initializer(tmp_path,study)


@pytest.mark.parametrize('mutation',['architecture','precision','parameterized','imported','reused','wrong_selected','random_parent'])
def test_retained_initializer_rejects_incompatible_or_untrusted_lineage(tmp_path,mutation):
    from oeis_learn.generalization.initializers import verify_initializer
    target,authority,checkpoint,put=fixture(tmp_path)
    if mutation=='architecture':target['model']['architecture']=digest('other model')
    elif mutation=='precision':target['precision']='fp16'
    elif mutation=='parameterized':target['data']['target_format']='parameterized'
    elif mutation=='imported':authority['ancestry'].append('oeis')
    elif mutation=='reused':
        producer=json.loads((tmp_path/'receipt.json').read_text());producer['state']='REUSED'
        authority['producer']=put('receipt.json',seal(producer))
    elif mutation=='wrong_selected':
        authority['selection']=put('selection.json',seal(dict(version='generalization-checkpoint-selection/v1',status='PASS',selected=digest('wrong'))))
    elif mutation=='random_parent':
        checkpoint['parent']=digest('unknown imported initializer')
        authority['checkpoint']=put('checkpoint.json',seal(checkpoint))
    target['model']['initialization']['artifact']=put('initializer.json',seal(authority))
    with pytest.raises(ValueError):verify_initializer(tmp_path,seal(target))


def test_retained_initialization_never_changes_v1_meaning(tmp_path):
    target,_,_,_=fixture(tmp_path);target['version']='generalization-study/v1'
    with pytest.raises(ValueError):freeze('study',target)


def test_selection_claim_without_complete_measurements_is_not_initializer_authority(tmp_path):
    from oeis_learn.generalization.initializers import verify_initializer
    target,authority,checkpoint,put=fixture(tmp_path)
    selection=json.loads((tmp_path/'selection.json').read_text())
    selection['measurements'][0]['behavior']['structural'][1]['s']=None
    authority['selection']=put('selection.json',seal(selection))
    producer=json.loads((tmp_path/'receipt.json').read_text())
    producer['files']['selection.json']=authority['selection']['sha256']
    authority['producer']=put('receipt.json',seal(producer))
    target['model']['initialization']['artifact']=put('initializer.json',seal(authority))
    with pytest.raises(ValueError,match='contradicts'):
        verify_initializer(tmp_path,seal(target))


def test_original_parent_chain_is_required_and_reverified(tmp_path):
    from oeis_learn.generalization.initializers import verify_initializer
    parent_target,parent_authority,parent_checkpoint,_=fixture(tmp_path/'parent')
    target,authority,checkpoint,put=fixture(tmp_path)
    parent_ref=reference(tmp_path/'parent/initializer.json',tmp_path)
    source=json.loads((tmp_path/'study.json').read_text())
    source['version']='generalization-study/v2'
    source['model']['initialization']=dict(kind='retained',artifact=parent_ref)
    source=seal(source);authority['study']=put('study.json',source)
    checkpoint['parent']=parent_checkpoint['weights']['sha256'];checkpoint=seal(checkpoint)
    authority['checkpoint']=put('checkpoint.json',checkpoint);authority['parent']=parent_ref
    selection=json.loads((tmp_path/'selection.json').read_text())
    selection.update(selected=checkpoint['identity'],eligible=[checkpoint['identity']],best_loss=checkpoint['identity'])
    selection['candidates'][0]['checkpoint']=checkpoint
    selection['measurements'][0]['checkpoint']=checkpoint['identity']
    selection['measurements'][0]['behavior']['id']=checkpoint['identity']
    authority['selection']=put('selection.json',seal(selection))
    producer=json.loads((tmp_path/'receipt.json').read_text())
    producer['binding'].update(study=source['identity'],checkpoint=checkpoint['identity'],parent=parent_authority['identity'])
    producer['files'].update({authority[k]['path']:authority[k]['sha256'] for k in ('study','checkpoint','selection')})
    authority['producer']=put('receipt.json',seal(producer))
    target['model']['initialization']['artifact']=put('initializer.json',seal(authority))
    result=verify_initializer(tmp_path,seal(target))
    assert len(result['lineage'])==2
    (tmp_path/'parent/weights.pt').write_bytes(b'tampered ancestor')
    with pytest.raises(ValueError):verify_initializer(tmp_path,seal(target))


def test_preparation_v2_binds_retained_reference_and_preserves_v1(tmp_path):
    target,_,_,_=fixture(tmp_path)
    preparation=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())['preparation']
    preparation['version']='generalization-preparation/v2'
    preparation['study_settings'].update(model=target['model'],arms=target['arms'],recipe=target['recipe'])
    assert freeze('preparation',preparation).record()['study_settings']['model']==target['model']
    preparation['version']='generalization-preparation/v1'
    with pytest.raises(ValueError):freeze('preparation',preparation)
