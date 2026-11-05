import copy
import json
from pathlib import Path
import pytest
from oeis_learn.generalization.contracts import freeze
from oeis_learn.pipeline_validation.identity import seal,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.pipeline_validation.generalization_runtime import load,validate_execution,learner_config


def fixture(root):
    raw=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())
    def put(name,value):
        (root/name).write_text(json.dumps(value));return reference(root/name,root)
    recipe=dict(version='fixture-model/v1',model=dict(seed=7,width=8),learning=dict(learning_rate=.001))
    study=raw['study'];study['model'].update(profile=recipe['version'],architecture=digest({'width':8}))
    empty=put('empty.json',[]);study['source_closure']=empty
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.data.bigint_programs import canonical_family,materialize
    from oeis_learn.generalization.refresh import RefreshPlan
    parameter='(program (return (param.get p0)))'
    descriptor=seal(dict(version='generalization-descriptor/v1',generator=digest('mechanical'),family=canonical_family(parameter),stage='A',
        parameter_source=parameter,numeric_law='independent-uniform-integers[-16,16]/v1',parameter_names=['p0']))
    study['data']['descriptors']=put('descriptors.json',[descriptor])
    study['data']['exclusions']=empty;study['data']['sharing_policy']='explicit'
    study['data']['generator']=put('generator.json',dict(version='generalization-materialization-plan/v2',cells=[dict(name='training',role='training',holdout='parameter',count=4,quotas={},structural=False,refresh=dict(seed=7,capacity=4,cursor=0,draws=4))]))
    generated,_=RefreshPlan([descriptor],seed=7,capacity=4).take(0,4)
    rows=[]
    for source in generated:
        actual=materialize(source)
        rows.append(dict(id=digest(['training',actual['id'],actual['source']]),role='training',source=actual['source'],tokens=encode_body(actual['source']),indices=actual['indices'],values=actual['values']))
    shard=put('training.json',rows)
    static=put('static.json',rows[:1])
    cohort=raw['cohort'];cohort.update(role='training',private_payloads=[shard,static],public_payloads=[],exclusions=empty,preparation_ledger=empty)
    study['data']['manifests']=[put('cohort.json',freeze('cohort',cohort).record())]
    study['budgets'].update(target_tokens=sum(len(row['tokens']) for row in rows[:2]),seconds=30)
    study['checkpoints']['milestones']=[len(rows[0]['tokens'])]
    study=freeze('study',study).record();put('study.json',study)
    runtime=seal(dict(version='generalization-runtime/v1',study=study['identity'],purpose='research',recipe=put('recipe.json',recipe),images=['sha256:'+'1'*64,'sha256:'+'2'*64],
        preparation=None,training={a:dict(shards=[shard if a=='refreshed' else static],data_mode='refresh' if a=='refreshed' else 'static',batch_size=1) for a in study['arms']},
        evaluation=None,replay=None,assessment=None,feature=None,
        storage=dict(study_bytes=64<<20,checkpoint_bytes=32<<20,shard_bytes=8<<20,raw_bytes=4<<20,sharing_bytes=1<<20)))
    from oeis_learn.generalization.execution import bind
    study,runtime=bind(study,runtime);put('study.json',study);put('runtime.json',runtime)
    return study,runtime,put


def test_runtime_binds_recipe_cohort_views_and_declared_pairs(tmp_path):
    study,runtime,_=fixture(tmp_path)
    root,loaded,sidecar=load(tmp_path,'study.json')
    assert loaded==study and sidecar==runtime
    plan=validate_execution(root,loaded,sidecar)
    config=learner_config(study,runtime,plan['recipe'],'refreshed',study['seeds'][0])
    assert config['data_mode']=='refresh' and config['device']=='cuda'
    with pytest.raises(ValueError,match='undeclared'):learner_config(study,runtime,plan['recipe'],'unknown',1)
    bad=copy.deepcopy(runtime);bad['training']['static']['shards'][0]['path']='hidden.json'
    with pytest.raises(ValueError,match='execution binding'):validate_execution(root,study,bad)


def test_runtime_rejects_changed_model_and_hidden_training_targets(tmp_path):
    study,runtime,put=fixture(tmp_path)
    recipe=json.loads((tmp_path/'recipe.json').read_text());recipe['model']['width']=9
    runtime['recipe']=put('changed.json',recipe)
    with pytest.raises(ValueError,match='execution binding'):validate_execution(tmp_path,study,runtime)
    changed=copy.deepcopy(runtime);changed['study']=digest('different study');put('runtime.json',seal(changed))
    with pytest.raises(ValueError,match='execution binding'):load(tmp_path,'study.json')


def test_canonical_train_invokes_validation_then_real_cpu_tensor_worker(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import torch
    from oeis_learn.pipeline_validation import generalization_commands as command,generalization_qualification_io as qualification
    from oeis_learn.pipeline_validation.generalization_cli import main
    from oeis_learn.pipeline_validation.generalization_training import run_torch
    from oeis_learn.fresh import bigint_worker
    study,runtime,_=fixture(tmp_path)
    producer=tmp_path/'qualified';producer.mkdir();(producer/'empty.json').write_text('[]')
    # This test replaces the container/qualification boundary, not the learner's
    # persistence or token accounting. It does not establish native/HIP evidence.
    calls=[]
    def consume(*args,**kwargs):
        calls.append('qualification')
        return dict(original=digest('original fixture'),authority=dict(source_closure=reference(producer/'empty.json',producer),images=runtime['images']))
    monkeypatch.setattr(qualification,'consume_original',consume)
    def models(recipe,device):
        assert device=='cpu';torch.manual_seed(recipe['model']['seed'])
        return torch.nn.Linear(1,1),torch.nn.Linear(1,1),{}
    def update(encoder,decoder,batch,device,**kwargs):
        optimizer=kwargs['optimizer'];optimizer.zero_grad()
        loss=decoder(encoder(torch.ones(len(batch),1))).square().mean()
        loss.backward();optimizer.step()
    monkeypatch.setattr(bigint_worker,'models',models);monkeypatch.setattr(bigint_worker,'_batch_loss',update)
    def invoke(host,image,workspace,role,artifacts,**kwargs):
        settings=kwargs['settings'];calls.append(role)
        if role=='preparation':
            assert set(artifacts)=={'mechanical_sources','isolation_truth'}
            return dict(status='PASS',plan=validate_execution(kwargs['source_roots']['mechanical_sources'],settings['study'],settings['runtime']))
        assert role=='learner' and set(artifacts)=={'training_targets','verified_training_replay'}
        assert settings['config']['device']=='cuda' and kwargs['gpu'] is True
        out=workspace/kwargs['output']/'role/private/learner'
        return run_torch(kwargs['source_roots']['training_targets'],out,dict(settings['config'],device='cpu'))
    monkeypatch.setattr(command,'project_and_invoke',invoke)
    args=SimpleNamespace(operation='train',manifest='study.json',runtime=None,output='trained',image=None,
        qualification='qualified',arms=['static'],seeds=study['seeds'],resume=None,reuse_run=None)
    assert main(args,tmp_path,lambda args:object())==0
    result=json.loads((tmp_path/'trained/result.json').read_text())
    width=len(json.loads((tmp_path/'static.json').read_text())[0]['tokens'])
    assert result['result']['runs'][0]['result']['updates']==study['budgets']['target_tokens']//width
    assert calls==['qualification','preparation','learner']
    with pytest.raises(ValueError,match='fresh'):main(args,tmp_path,lambda args:pytest.fail('engine reused terminal root'))
    args.output='unknown';args.arms=['not-declared']
    with pytest.raises(ValueError,match='undeclared'):main(args,tmp_path,lambda args:pytest.fail('engine touched before invalid subset rejection'))


def test_evaluation_projects_original_weights_and_selection_without_truth_leak(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_workflow as workflow,generalization_scoring as scoring,generalization_evaluation as generation
    from oeis_learn.fresh import pilot_metrics
    from oeis_learn.generalization.contracts import verify_artifact
    study,runtime,put=fixture(tmp_path)
    rows=[dict(id=name,role='selection',source='(program (return (int 0)))',family=digest(name),stratum=name,
        indices=list(map(str,range(100))),values=['0']*100) for name in ('structural','parameter')]
    prompts=put('prompts.json',[{k:r[k] for k in ('id','indices','values')} for r in [dict(row,indices=row['indices'][:20],values=row['values'][:20]) for row in rows]])
    targets=put('targets.json',rows);weights=put('weights.pt',{'fixture':True})
    checkpoint=seal(dict(version='generalization-checkpoint/v1',kind='full_state',weights=weights,state=dict(path='not-mounted.json',sha256=digest('state'),bytes=1),initialization=digest('init'),parent=None,tokens=1,updates=1,pins=['latest'],compatibility=digest('contract')))
    descriptor=put('checkpoint.json',checkpoint)
    runtime['evaluation']=dict(role='selection',prompts=[prompts],targets=[targets],checkpoints=[descriptor],seconds=60,bank_seeds=None)
    calls=[]
    def invoke(host,image,workspace,role,artifacts,**kwargs):
        settings=kwargs['settings'];roots=kwargs['source_roots'];calls.append(role)
        if role=='generator':
            assert set(artifacts)=={'visible_prompts','frozen_weights'}
            visible=json.loads(verify_artifact(roots['visible_prompts'],settings['prompts']))
            assert all(set(r)=={'id','indices','values'} and len(r['values'])==20 for r in visible)
            folder=workspace/kwargs['output']/'role/generation';folder.mkdir(parents=True)
            (folder/'selections.json').write_text(json.dumps({r['id']:{'selections':{}} for r in visible}))
            return dict(status='PASS')
        assert role=='selection'
        assert all(c['checkpoint']==checkpoint for c in settings['candidates'])
        return scoring.select_checkpoints(roots['selection_supervision'],settings['targets'],roots['selection_candidates'],settings['candidates'],
            recipe=settings['recipe'],client=None,output=workspace/kwargs['output']/'role/selection',seconds=kwargs['seconds'],directories=settings['directories'])
    monkeypatch.setattr(workflow,'project_and_invoke',invoke)
    monkeypatch.setattr(scoring,'score_frozen',lambda *a,**kw:dict(outcomes={r['id']:{'equal_budget':dict(success=True,complete=True)} for r in rows}))
    monkeypatch.setattr(generation,'load_retained',lambda *a,**kw:(None,None,{}))
    monkeypatch.setattr(pilot_metrics,'measure',lambda *a,**kw:dict(complete=True,loss=1.))
    result=workflow.evaluate(None,tmp_path,tmp_path/'evaluated',tmp_path,study,runtime,dict(recipe={}),checkpoints=[checkpoint['identity']])
    assert result['status']=='PASS' and result['selection']['selected']==checkpoint['identity']
    assert calls==['generator','selection']
    (tmp_path/'weights.pt').unlink();calls.clear()
    result=workflow.evaluate(None,tmp_path,tmp_path/'missing',tmp_path,study,runtime,dict(recipe={}),checkpoints=[checkpoint['identity']])
    assert result['status']=='PARTIAL' and result['runs'][0]['reason']=='MISSING_ARTIFACT' and not calls
    put('weights.pt',{'fixture':True})
    monkeypatch.setattr(workflow,'project_and_invoke',lambda *a,**kw:dict(status='FAIL',reason='injected generation failure'))
    failed=workflow.evaluate(None,tmp_path,tmp_path/'failed-generation',tmp_path,study,runtime,dict(recipe={}),checkpoints=[checkpoint['identity']])
    assert failed['status']=='FAIL' and failed['runs'][0]['result']['reason']=='injected generation failure'


def test_command_export_failure_retains_terminal_work_result(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.generalization_commands import _finish
    from oeis_learn.pipeline_validation import bundle
    study,runtime,_=fixture(tmp_path);output=tmp_path/'failed';output.mkdir()
    def failed(*a,**kw):raise OSError('injected export failure')
    monkeypatch.setattr(bundle,'pack',failed)
    assert _finish(output,study,'train',dict(status='PASS',updates=2),runtime)==1
    result=json.loads((output/'result.json').read_text())
    assert result['status']=='FAIL' and result['result']['reason']=='PACKAGE_FAILED'
    assert result['result']['work_result']==dict(status='PASS',updates=2)
    assert json.loads((output/'summary.json').read_text())['selected_scope']=='FAIL'


def test_projection_deadline_charges_io_and_never_launches_late(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_roles as roles
    from oeis_learn.generalization import contracts
    source=tmp_path/'source';source.mkdir();(source/'x').write_text('x')
    ref=reference(source/'x',source);clock=[0.];calls=[]
    original=contracts.verify_artifact
    def verify(*a,**kw):clock[0]+=2;return original(*a,**kw)
    monkeypatch.setattr(contracts,'verify_artifact',verify)
    monkeypatch.setattr(roles.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(roles,'invoke',lambda *a,**kw:calls.append(kw) or {'status':'PASS'})
    for seconds,expected in [(1,None),(5,3)]:
        kwargs=dict(output='projection'+str(seconds),operation='fixture',settings={},seconds=seconds,source_roots={'public_evidence':source})
        if expected is None:
            with pytest.raises(TimeoutError):roles.project_and_invoke(None,'image',tmp_path,'report',{'public_evidence':[ref]},**kwargs)
            assert not calls
        else:
            roles.project_and_invoke(None,'image',tmp_path,'report',{'public_evidence':[ref]},**kwargs)
            assert calls[-1]['seconds']==expected
        assert not (tmp_path/kwargs['output']/'private/input').exists()


def test_direct_projections_have_one_copy_and_preserve_originals(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_roles as roles
    from oeis_learn.generalization.contracts import verify_artifact
    sources={};refs=[]
    for name in ('first','second'):
        root=tmp_path/name;root.mkdir();(root/'state.pt').write_bytes(name.encode())
        original=reference(root/'state.pt',root);projected=dict(original,path=name+'/state.pt')
        refs.append(projected);sources[projected['path']]=(root,original)
    calls=[]
    def invoke(host,image,root,role,channels,**kw):
        target=root/channels['verified_training_replay']
        assert [verify_artifact(target,r) for r in refs]==[b'first',b'second']
        assert len(list(root.rglob('state.pt')))==4  # two originals + one projection each
        calls.append(channels);return {'status':'PASS'}
    monkeypatch.setattr(roles,'invoke',invoke)
    roles.project_and_invoke(None,'image',tmp_path,'learner',{'training_targets':[],'verified_training_replay':refs},
        artifact_sources={'verified_training_replay':sources},output='projection',operation='check-training',settings={},seconds=10)
    assert calls and len(list(tmp_path.rglob('state.pt')))==2
    assert not (tmp_path/'projection/private/input').exists()
    bad=dict(refs[0],sha256='sha256:'+'0'*64)
    with pytest.raises(ValueError,match='identity differs'):
        roles.project_and_invoke(None,'image',tmp_path,'learner',{'training_targets':[],'verified_training_replay':[bad]},
            artifact_sources={'verified_training_replay':sources},output='bad',operation='check-training',settings={},seconds=10)


def pac_config(tmp_path,arm='C'):
    from tests.integration.test_generalization_curriculum import prepared
    from oeis_learn.fresh.bigint_config import resolve
    batches,bundle=prepared(tmp_path,arm)
    return dict(study=digest('pac-installed-fixture'),arm=arm,seed=7,recipe=resolve(dict(version='bigint-bootstrap/v1')),
        shards=bundle['parameterized'],data_mode='static',batch_size=2,target_tokens=240,milestones=[],seconds=300,device='cpu',checkpoint_policy='hourly/v1',
        pac=dict(bundle=bundle,purpose='acceptance',source=digest('source'),environment=digest('cpu'),initializer=digest('random'),
            observation_seconds=120,publication_seconds=10))


def assert_tensor_tree(a,b):
    import torch
    import numpy as np
    if isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:assert_tensor_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tensor_tree(x,y)
    else:assert a==b


@pytest.mark.parametrize('boundary',[3,8])
def test_pac_real_installed_pause_resume_tensor_policy_equality(tmp_path,monkeypatch,boundary):
    from oeis_learn.pipeline_validation.generalization_training import run_torch
    from oeis_learn.pipeline_validation import generalization_curriculum as module
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    # Inject only unavailable native semantic measurement, never model/update/loss.
    monkeypatch.setattr(module,'semantic_probe',lambda *a,**kw:None)
    config=pac_config(tmp_path)
    full=run_torch(tmp_path/'data',tmp_path/'full',config)
    part=run_torch(tmp_path/'data',tmp_path/'part',config,stop_after_updates=boundary)
    resumed=run_torch(tmp_path/'data',tmp_path/'resumed',config,resume=(tmp_path/'part/checkpoints',part['checkpoint']))
    def load(name,result):return GeneralizationCheckpointStore(tmp_path/name/'checkpoints',result['compatibility'],config['target_tokens']).load(result['checkpoint'])
    a=load('full',full);b=load('resumed',resumed);p=load('part',part)
    assert p['states']['scheduler']['pac']['policy']['block']['phase']==('post' if boundary==8 else 'updates')
    assert a['ledger']==b['ledger']
    for key in ('weights','optimizer','rng','data'):assert_tensor_tree(a['states'][key],b['states'][key])
    assert a['states']['scheduler']['pac']['policy']==b['states']['scheduler']['pac']['policy']
    assert a['states']['scheduler']['pac']['observer']==b['states']['scheduler']['pac']['observer']
    assert full['initial_weights']==resumed['initial_weights']
    assert full['timing']['checkpoint_calls']<=3
    changed=dict(config,pac=dict(config['pac'],environment=digest('changed')))
    with pytest.raises(ValueError,match='incompatible'):
        run_torch(tmp_path/'data',tmp_path/'wrong',changed,resume=(tmp_path/'part/checkpoints',part['checkpoint']))


def test_pac_publication_failure_keeps_previous_real_generation(tmp_path,monkeypatch):
    import errno
    from oeis_learn.pipeline_validation.generalization_training import run_torch
    from oeis_learn.pipeline_validation import generalization_curriculum as module
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    monkeypatch.setattr(module,'semantic_probe',lambda *a,**kw:None)
    config=pac_config(tmp_path)
    part=run_torch(tmp_path/'data',tmp_path/'part',config,stop_after_updates=3)
    store=GeneralizationCheckpointStore(tmp_path/'part/checkpoints',part['compatibility'],config['target_tokens'])
    old=store.load(part['checkpoint'])
    def full_disk(value):raise OSError(errno.ENOSPC,'injected full disk')
    with pytest.raises(OSError):store.save(old['states'],old['ledger'],encode=full_disk)
    assert json.loads((store.root/'generalization-latest.json').read_text())==part['checkpoint']
    assert_tensor_tree(store.load(part['checkpoint'])['states']['weights'],old['states']['weights'])


def test_pac_real_crash_discards_unsaved_work_without_policy_replay(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_training as training
    from oeis_learn.pipeline_validation import generalization_curriculum as module
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    monkeypatch.setattr(module,'semantic_probe',lambda *a,**kw:None)
    config=pac_config(tmp_path);original=training.torch_update
    def injected(*args,**kwargs):
        update=original(*args,**kwargs);count=0
        def step(batch):
            nonlocal count
            result=update(batch);count+=1
            if count==3:raise RuntimeError('crash after actual optimizer update')
            return result
        return step
    monkeypatch.setattr(training,'torch_update',injected)
    with pytest.raises(RuntimeError,match='crash'):
        training.run_torch(tmp_path/'data',tmp_path/'crashed',config)
    root=tmp_path/'crashed/checkpoints';pointer=json.loads((root/'generalization-latest.json').read_text())
    manifest=json.loads((root/pointer['path']).read_text())
    old=GeneralizationCheckpointStore(root,manifest['contract'],config['target_tokens']).load(pointer)
    assert old['ledger'].updates==0
    monkeypatch.setattr(training,'torch_update',original)
    result=training.run_torch(tmp_path/'data',tmp_path/'recovered',config,resume=(root,pointer),resume_elapsed=30)
    assert result['discarded_tokens']>0 and result['charged_tokens']<=config['target_tokens']
    assert result['reason']=='RECOVERED_WITH_DISCARDED_EXPOSURE'
    restored=GeneralizationCheckpointStore(tmp_path/'recovered/checkpoints',result['compatibility'],config['target_tokens']).load(result['checkpoint'])
    policy=restored['states']['scheduler']['pac']['policy']
    assert policy['discarded_tokens']==result['discarded_tokens']
    assert policy['segment']=='recovery-1' and policy['kernel']['valid_updates']==0
