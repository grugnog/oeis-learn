import json
import pytest
from oeis_learn.pipeline_validation.generalization_preparation import prepare_cell
from oeis_learn.pipeline_validation.generalization_roles import mounts
from oeis_learn.pipeline_validation.identity import digest
from oeis_learn.data.bigint_programs import canonical_family


def source(i):
    return dict(id=str(i),stage='A',source=f'(program (return (add $index (int {i}))))',parameter_source=f'(program (return (add $index (int {i}))))',family=canonical_family(f'(program (return (add $index (int {i}))))'),variant=str(i),group='group'+str(i),origin='generic_random')

def execute(src,indices):
    i=int(src.split('(int ')[1].split(')')[0]);return dict(values=[str(int(n)+i) for n in indices],outcome='ok')

def test_separate_visible_and_scorer_files_and_shortfall(tmp_path):
    result=prepare_cell([source(1),source(2)],role='selection',holdout='parameter',count=2,quotas={},structural=False,output=tmp_path/'good',execute=execute)
    assert result['status']=='PASS'
    visible=json.loads((tmp_path/'good/prompts.json').read_text())
    assert all(set(r)=={'id','indices','values'} and len(r['values'])==20 for r in visible)
    assert len(json.loads((tmp_path/'good/private/targets.json').read_text())[0]['values'])==100
    failed=prepare_cell([source(1)],role='assessment',holdout='structural',count=2,quotas={'loop_active':2},structural=True,output=tmp_path/'short',execute=execute)
    assert failed['status']=='BLOCKED' and not (tmp_path/'short/prompts.json').exists()
    assert not (tmp_path/'short/study.json').exists()
    with pytest.raises(ValueError): prepare_cell([],role='training',holdout='parameter',count=1,quotas={},structural=False,output=tmp_path/'good',execute=execute)


def test_native_disagreement_does_not_publish(tmp_path):
    result=prepare_cell([source(1)],role='training',holdout='parameter',count=1,quotas={},structural=False,output=tmp_path/'bad',execute=lambda s,i:dict(values=['999']*100,outcome='ok'))
    assert result['status']=='BLOCKED'


def test_mounts_are_separate_and_read_only(tmp_path):
    for name in ('prompts','weights','truth'): (tmp_path/name).mkdir()
    result=mounts(tmp_path,'generator',{'visible_prompts':'prompts','frozen_weights':'weights'},output='generated')
    assert all(m['read_only'] for m in result[:-1]) and not result[-1]['read_only']
    with pytest.raises(ValueError): mounts(tmp_path,'generator',{'visible_prompts':'prompts','frozen_weights':'weights','assessment_truth':'truth'},output='generated')


def test_role_launch_mounts_only_declared_inputs_and_reclaims(tmp_path):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.generalization_roles import invoke
    for name in ('prompts','weights','hidden'):(tmp_path/name).mkdir()
    calls=[]
    class Engine:
        def command(self,**kw):calls.append(kw);return ['fake-container']
        def remove(self,name,timeout):calls.append(('remove',name))
    host=SimpleNamespace(engine=Engine(),policy=dict(memory_bytes=1024,cpus=2))
    def launch(argv,output,**kw):
        assert argv==['fake-container'];calls.append(('launch',kw))
        (tmp_path/'result/result.json').write_text('{"status":"PASS"}')
    result=invoke(host,'sha256:'+'1'*64,tmp_path,'generator',
        {'visible_prompts':'prompts','frozen_weights':'weights'},output='result',
        operation='evaluate',settings={'fixture':True},seconds=30,gpu=False,launcher=launch)
    assert result['status']=='PASS'
    args=calls[0]
    assert args['network'] is False and args['gpu'] is False
    assert args['mounts']==[(str(tmp_path/'prompts'),'/inputs/prompts',False),
                            (str(tmp_path/'weights'),'/inputs/weights',False),
                            (str(tmp_path/'result'),'/output/generator',True)]
    assert calls[-1][0]=='remove'
    with pytest.raises(ValueError):invoke(host,'sha256:'+'1'*64,tmp_path,'generator',
        {'visible_prompts':'prompts','frozen_weights':'weights'},output='result',operation='evaluate',settings={},seconds=30)


@pytest.mark.parametrize('stage',['P1-screen','P3'])
def test_complete_preparation_publishes_new_study_and_cross_role_isolation(tmp_path,monkeypatch,stage):
    from pathlib import Path
    from oeis_learn.generalization.contracts import freeze,validate,verify_artifact
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_study,reference
    from oeis_learn.synthesis.reference import evaluate
    raw=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())
    incoming=tmp_path/'incoming';incoming.mkdir()
    def artifact(name,value):
        (incoming/name).write_text(json.dumps(value));return reference(incoming/name,incoming)
    empty=artifact('empty.json',[])
    plans=[]
    for i,role in enumerate(('training','selection','assessment')):
        row=source(i);row['source']=f'(program (return (mul $index (int {i+2}))))'
        row['parameter_source']=row['source'];row['family']=canonical_family(row['source'])
        if stage=='P3' and role=='training':
            row['parameter_source']='(program (return (mul $index (param.get p0))))'
            row['assignment']={'p0':str(i+2)};row['family']=canonical_family(row['parameter_source'])
        plans.append(dict(name=role,role=role,holdout='parameter',count=1,quotas={},structural=False,
                          candidates=artifact(role+'.json',[row])))
    generator=artifact('generator.json',dict(version='generalization-materialization-plan/v1',cells=plans))
    prep=raw['preparation'];prep['budgets']['target_tokens']=10;prep.update(generator=generator,descriptors=empty,exclusions=empty,quotas={})
    if stage=='P3':prep['study_settings']['arms']=['literal','parameterized']
    study=raw['study'];study.update(prep['study_settings']);study.update(seeds=prep['seeds'],budgets=prep['budgets'],source_closure=empty)
    study['progression']['stage']=stage
    if stage=='P3':study['data']['sharing_policy']='distinct-occurrences'
    study['data'].update(generator=generator,descriptors=empty,exclusions=empty)
    study['assessment'].update(authorization=empty,exposure_ledger=empty)
    def native(src,indices):
        exact=evaluate(src,indices);return dict(outcome=exact.outcome,values=list(exact.values))
    result=prepare_study(incoming,freeze('preparation',prep),study,output=tmp_path/'prepared',execute=native)
    assert result['status']=='PASS'
    resolved=json.loads((tmp_path/'prepared/study.json').read_text());validate('study',resolved)
    assert len(resolved['data']['manifests'])==3
    training=json.loads((tmp_path/'prepared/cells/training/private/training.json').read_text())
    assert training[0]['role']=='training' and 'family' not in training[0]
    for ref in resolved['data']['manifests']:verify_artifact(tmp_path/'prepared',ref)
    if stage=='P3':
        proof=json.loads((tmp_path/'prepared/cells/training/private/target-pairs.json').read_text())[0]
        assert proof['assignment']=={'p0':'2'} and proof['literal_tokens']!=proof['parameter_tokens']
    # A second run uses the same authority and produces the same resolved identity.
    second=prepare_study(incoming,freeze('preparation',prep),study,output=tmp_path/'again',execute=native)
    assert result['study']==second['study']
    from oeis_learn.pipeline_validation.generalization_role import preparation
    from oeis_learn.synthesis import client as native_module
    import time
    class Native:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,src,indices,*,deadline):
            assert deadline>time.monotonic()
            return native(src,indices)
    monkeypatch.setattr(native_module,'SynthesisClient',Native)
    (tmp_path/'custodian').mkdir()
    role_result=preparation({'mechanical_sources':incoming,'isolation_truth':tmp_path/'custodian'},
        tmp_path/'role',dict(preparation=freeze('preparation',prep).record(),proposed=study),seconds=60)
    assert role_result['study']==result['study']
    assert (tmp_path/'role/private/prepared/study.json').is_file()

    plans[2]['candidates']=plans[0]['candidates']
    changed=artifact('collision.json',dict(version='generalization-materialization-plan/v1',cells=plans))
    prep['generator']=changed;study['data']['generator']=changed
    bad=prepare_study(incoming,freeze('preparation',prep),study,output=tmp_path/'bad-study',execute=native)
    assert bad['status']=='BLOCKED' and not (tmp_path/'bad-study/study.json').exists()


def test_projection_copies_only_named_verified_files(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_roles as roles
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    (tmp_path/'prompts.json').write_text('[]');(tmp_path/'private-target.json').write_text('secret')
    (tmp_path/'weights.json').write_text('{}')
    def invoke(host,image,root,role,channels,**kwargs):
        prompt_root=root/channels['visible_prompts']
        assert list(p.name for p in prompt_root.rglob('*') if p.is_file())==['prompts.json']
        assert not (prompt_root/'private-target.json').exists()
        return dict(status='PASS')
    monkeypatch.setattr(roles,'invoke',invoke)
    result=roles.project_and_invoke(None,'image',tmp_path,'generator',
        {'visible_prompts':[reference(tmp_path/'prompts.json',tmp_path)]*2,'frozen_weights':[reference(tmp_path/'weights.json',tmp_path)]},
        output='new',operation='evaluate',settings={},seconds=30)
    assert result['status']=='PASS' and not (tmp_path/'new/private/input').exists()


@pytest.mark.parametrize('role',['preparation','learner','generator','fitter','selection','assessment','report'])
def test_all_roles_reject_undeclared_and_overlapping_channels(tmp_path,role):
    from oeis_learn.generalization.role_contracts import ROLE_INPUTS,Role
    channels={m.channel:m.channel for m in ROLE_INPUTS[Role(role)]}
    for name in channels.values():(tmp_path/name).mkdir()
    actual=mounts(tmp_path,role,channels,output='result')
    assert len(actual)==len(channels)+1 and all(m['read_only'] for m in actual[:-1])
    with pytest.raises(ValueError):mounts(tmp_path,role,dict(channels,arbitrary='secret'),output='result')
    with pytest.raises(ValueError):mounts(tmp_path,role,channels,output=next(iter(channels.values()))+'/child')
    if len(channels)>1:
        overlap=dict.fromkeys(channels,next(iter(channels.values())))
        with pytest.raises(ValueError):mounts(tmp_path,role,overlap,output='result')


def test_runtime_publication_resolves_frozen_cell_views_and_rejects_cross_role(tmp_path):
    from oeis_learn.pipeline_validation.generalization_runtime import publish_runtime,preparation_views
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.identity import seal,write
    source_root=tmp_path/'source';source_root.mkdir();prepared=tmp_path/'prepared';prepared.mkdir()
    def put(root,name,value):
        write(root/name,value);return reference(root/name,root)
    plan=dict(cells=[dict(name=n,role=n) for n in ('training','selection','assessment')])
    prep=dict(generator=put(source_root,'generator.json',plan))
    study=dict(identity=digest('old'),arms=['literal','parameterized'],progression=dict(stage='P3'))
    put(prepared,'study.json',dict(study,identity=digest('new')))
    for name in ('literal','parameterized'):put(prepared,'cells/training/private/training-'+name+'.json',[name])
    put(prepared,'cells/selection/prompts.json',[dict(id='opaque')]);put(prepared,'cells/selection/private/targets.json',[])
    runtime=seal(dict(study=study['identity'],preparation=put(source_root,'preparation.json',prep),recipe=put(source_root,'recipe.json',{}),
        training={a:dict(cells=['training'],data_mode='static',batch_size=1) for a in study['arms']},
        evaluation=dict(role='selection',cells=['selection'],checkpoints=[],seconds=30,bank_seeds=None),replay=None,assessment=None))
    resolved,actual=publish_runtime(source_root,prepared,study,runtime)
    assert actual['study']==resolved['identity'] and actual['preparation'] is None
    assert actual['training']['literal']['shards'][0]['path'].endswith('training-literal.json')
    assert actual['training']['parameterized']['shards'][0]['path'].endswith('training-parameterized.json')
    assert (prepared/'recipe.json').is_file()
    runtime['training']['literal']['cells']=['assessment']
    with pytest.raises(ValueError,match='its role'):preparation_views(source_root,study,runtime,prep)
