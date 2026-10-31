"""Host assembly stays unqualified until the canonical custodian checks inputs."""
import copy,json,subprocess,sys,zipfile
from pathlib import Path
import pytest
from oeis_learn.pipeline_validation.identity import write,seal,file_hash,digest
from oeis_learn.pipeline_validation.generalization_inputs import import_uniform,assemble_study
from oeis_learn.pipeline_validation.generalization_preparation import reference


def legacy_inputs(root):
    from tests.unit.test_generalization_legacy import records
    from oeis_learn.pipeline_validation.bundle import pack
    t,p,r,h,s=records()
    weights=root/'train/private';weights.mkdir(parents=True)
    for label in ('best','latest'):
        path=weights/(label+'-weights.pt');path.write_bytes(label.encode());t[label+'_weights']=file_hash(path)
    t=seal(t)
    meta=root/'meta';meta.mkdir();write(meta/'recipe.json',r)
    p=seal(dict(p,files={'recipe.json':{'sha256':file_hash(meta/'recipe.json'),'bytes':(meta/'recipe.json').stat().st_size}}))
    t=seal(dict(t,work=p['identity']))
    s['observations']['result']['result']['work']=p['identity']
    write(meta/'prepared.json',p)
    run=root/'run';write(run/'summary.json',dict(selected_scope='PASS'))
    write(run/'stages/study.report/study.json',seal(dict(version='pipeline-followup-study-result/v1',arms=[t])))
    history=run/'stages/followup.uniform/train/validation-history.json';write(history,h)
    s=seal(dict(s,files={'train/validation-history.json':file_hash(history)}));write(run/'stages/followup.uniform/receipt.json',s)
    pack(run,mandatory=[str(x.relative_to(run)) for x in run.rglob('*.json')])
    with zipfile.ZipFile(root/'metadata.zip','w') as z:
        for name in ('recipe.json','prepared.json'):z.write(meta/name,name)


def test_uniform_import_preserves_real_hashes_and_never_claims_resume(tmp_path):
    legacy_inputs(tmp_path)
    result=import_uniform(tmp_path,bundle='run.zip',metadata='metadata.zip',weights='train',output='imported')
    assert not result['resume_available'] and not result['initializer_available']
    best=json.loads((tmp_path/'imported/best-checkpoint.json').read_text())
    assert best['kind']=='weights_only' and best['state'] is None and best['tokens']==10
    from oeis_learn.generalization.contracts import validate
    validate('checkpoint',best)
    from oeis_learn.pipeline_validation.generalization_workflow import _checkpoints
    assert _checkpoints(tmp_path/'imported',[reference(tmp_path/'imported/best-checkpoint.json',tmp_path/'imported')])[0][0]==best
    (tmp_path/'train/private/best-weights.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='weights mismatch'):
        import_uniform(tmp_path,bundle='run.zip',metadata='metadata.zip',weights='train',output='bad')
    assert not (tmp_path/'bad').exists()


def blueprint(root):
    from oeis_learn.data.bigint_programs import canonical_family
    def put(name,data):write(root/name,data);return name
    descriptor=seal(dict(version='generalization-descriptor/v1',generator=digest('mechanical'),family=canonical_family('(program (return (param.get p0)))'),stage='A',parameter_source='(program (return (param.get p0)))',numeric_law='independent-uniform-integers[-16,16]/v1',parameter_names=['p0']))
    cells=[dict(name='static',role='training',holdout='parameter',count=1,quotas={},structural=False,refresh=dict(seed=7,capacity=4,cursor=0,draws=1)),
           dict(name='fresh',role='training',holdout='parameter',count=3,quotas={},structural=False,refresh=dict(seed=7,capacity=4,cursor=1,draws=3))]
    for ordinal,role in enumerate(('selection','assessment')):
        source=f'(program (return (mul $index (int {ordinal+2}))))'
        row=dict(id=role,stage='A',source=source,parameter_source=source,family=canonical_family(source),variant=role,group=role,origin='generic_random')
        cells.append(dict(name=role,role=role,holdout='parameter',count=1,quotas={},structural=False,candidates=put(role+'.json',[row])))
    return dict(version='generalization-input-blueprint/v1',stage='P1-screen',purpose='acceptance',hypothesis='reduced mechanical refresh check',
        recipe=put('recipe.json',dict(version='fixture-model/v1',model=dict(seed=7,width=8),learning=dict(learning_rate=.001))),source_closure=put('source.json',{}),
        images=['sha256:'+'1'*64,'sha256:'+'2'*64],initializer=None,descriptors=put('descriptors.json',[descriptor]),exclusions=put('exclusions.json',[]),cells=cells,
        training={'static':dict(cells=['static'],data_mode='static',batch_size=1),'refreshed':dict(cells=['static','fresh'],data_mode='refresh',batch_size=1)},evaluation=None,replay=None,
        seeds=[20261016],budgets=dict(attempts=8,draws=100,log_bytes=1024,memory_bytes=1<<20,seconds=30,storage_bytes=8<<20,target_tokens=27),milestones=[14,27],
        storage=dict(study_bytes=64<<20,checkpoint_bytes=32<<20,shard_bytes=8<<20,raw_bytes=4<<20,sharing_bytes=1<<20),sharing_policy='explicit')


def test_assembled_study_prepares_and_validates_through_canonical_policies(tmp_path):
    from oeis_learn.pipeline_validation.generalization_runtime import load,publish_runtime,validate_execution
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_study
    from oeis_learn.generalization.contracts import validate
    from oeis_learn.synthesis.reference import evaluate
    cfg=blueprint(tmp_path);result=assemble_study(tmp_path,cfg,'inputs')
    base,s,r=load(tmp_path,result['manifest']);prep=json.loads((base/'preparation.json').read_text())
    validate('study',s);validate('preparation',prep);validate_execution(base,s,r)
    def native(source,indices):
        x=evaluate(source,indices);return dict(outcome=x.outcome,values=list(x.values))
    assert prepare_study(base,validate('preparation',prep),s,output=tmp_path/'prepared',execute=native)['status']=='PASS'
    actual,sidecar=publish_runtime(base,tmp_path/'prepared',s,r)
    assert actual['identity']!=s['identity'];validate_execution(tmp_path/'prepared',actual,sidecar)
    changed=copy.deepcopy(s);changed['data']['sharing_policy']='distinct-occurrences';changed=seal(changed)
    with pytest.raises(ValueError,match='changed proposed'):
        prepare_study(base,validate('preparation',prep),changed,output=tmp_path/'tampered',execute=native)
    assert not (tmp_path/'tampered').exists()


def test_stdlib_cli_has_no_host_package_dependencies(tmp_path):
    cfg=blueprint(tmp_path);write(tmp_path/'blueprint.json',cfg)
    code='from oeis_learn.pipeline_validation.generalization_inputs import main; raise SystemExit(main(["study","--blueprint","blueprint.json","--output","new"],workspace='+repr(str(tmp_path))+'))'
    import os
    env=dict(os.environ,PYTHONPATH=str(Path('src').resolve()))
    result=subprocess.run([sys.executable,'-S','-c',code],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['status']=='ASSEMBLED'


def test_missing_p4_initializer_blocks_without_creating_output(tmp_path):
    cfg=blueprint(tmp_path);cfg['stage']='P4'
    with pytest.raises(ValueError,match='selection-designated initializer'):
        assemble_study(tmp_path,cfg,'missing')
    assert not (tmp_path/'missing').exists()


def test_evaluation_only_preflight_does_not_allocate_a_learner(monkeypatch):
    from oeis_learn.pipeline_validation.generalization_training import storage_preflight
    from oeis_learn.fresh import bigint_worker
    monkeypatch.setattr(bigint_worker,'models',lambda *a:pytest.fail('evaluation-only preflight constructed a learner'))
    assert storage_preflight({}, {}, {'training':{}})['required_checkpoint_bytes']==0


def test_session_assembly_rejects_wrong_case_before_launch(tmp_path):
    from oeis_learn.pipeline_validation.generalization_inputs import assemble_session
    cfg=blueprint(tmp_path);result=assemble_study(tmp_path,cfg,'refresh')
    cases={name:result['manifest'] for name in ('retained-eval','refresh','parameter-targets','replay-resume')}
    with pytest.raises(ValueError,match='case'):
        assemble_session(tmp_path,cases,'acceptance.json')
    assert not (tmp_path/'acceptance.json').exists()


def test_parameter_acceptance_rejects_all_unsupported_before_training(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.generalization_acceptance import run_case
    from oeis_learn.pipeline_validation import generalization_commands
    from oeis_learn.generalization.execution import bind
    cfg=blueprint(tmp_path);cfg.update(stage='P3',training={a:dict(cells=['static'],data_mode='static',batch_size=1) for a in ('literal','parameterized')})
    result=assemble_study(tmp_path,cfg,'parameters')
    from oeis_learn.pipeline_validation.generalization_runtime import load
    base,study,runtime=load(tmp_path,result['manifest'])
    monkeypatch.setattr(generalization_commands,'train',lambda *a,**kw:pytest.fail('invalid acceptance launched learner'))
    proof=run_case('parameter-targets',None,tmp_path,tmp_path/'case',base,study,runtime,dict(target_pairs_verified=2,target_pairs_parameterized=0),qualification=digest('fixture'))
    assert proof['status']=='FAIL' and 'actual supported parameterized' in proof['reason']


def test_replay_acceptance_requires_finite_updates_in_both_arms(tmp_path,monkeypatch):
    from tests.unit.test_generalization_convergence import execution_fixture
    from oeis_learn.pipeline_validation.generalization_acceptance import run_case
    from oeis_learn.pipeline_validation import generalization_workflow
    s,r=execution_fixture();s.update(recipe='P4-replay/v1',arms=['continued_sft','verified_replay'],changing_factors=['verified_replay'])
    s['progression']['stage']='P4';s['model']['initialization']={'kind':'retained','artifact':s['source_closure']}
    r['training']={a:dict(cells=['static'],batch_size=4,data_mode='static') for a in s['arms']}
    r['replay']=dict(cell='static',seconds=30);r['evaluation']=dict(role='selection',seconds=30,checkpoints=[])
    monkeypatch.setattr(generalization_workflow,'replay',lambda *a,**kw:dict(status='PASS',rounds=1,runs=[dict(cumulative_discoveries=1,search={},paired={'runs':[{'result':{}},{'result':{}}]},resume={'status':'PASS'})]))
    result=run_case('replay-resume',None,tmp_path,tmp_path/'case',tmp_path,s,r,{},qualification=digest('fixture'))
    assert result['status']=='FAIL'
