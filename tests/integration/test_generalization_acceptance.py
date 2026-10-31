"""No HIP is launched: subprocess boundary is replaced with a recording adapter."""
import importlib.util
from pathlib import Path
import json
import pytest


def helper():
    path=Path(__file__).resolve().parents[2]/'scripts/generalization_hip_serial.py'
    spec=importlib.util.spec_from_file_location('generalization_serial',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_serial_plan_one_qualification_and_no_research(tmp_path):
    h=helper();plan=h.plan('docker','manifest.json','run-state/new')
    assert [p['operation'] for p in plan]==['preflight','qualify','prepare']+['acceptance']*5+['report','verify-bundle']
    assert sum(p['seconds'] for p in plan)+600==18000
    from oeis_learn.generalization.resources import reserve_stage
    elapsed=0
    for stage in plan:
        reserve_stage(stage['category'],elapsed_seconds=elapsed,requested_seconds=stage['seconds'])
        elapsed+=stage['seconds']
    consumers=[p for p in plan if p['operation'] in ('prepare','acceptance')]
    assert all(p['argv'][p['argv'].index('--qualification')+1]=='run-state/new/qualify' for p in consumers)
    assert not any(p['operation'] in ('train','replay') for p in plan)
    assert '--start-step' not in str(plan)


def test_serial_failure_still_reports_and_preserves_ledger(tmp_path):
    h=helper();calls=[];out=tmp_path/'new'
    def run(stage,root,seconds):
        calls.append(stage['name'])
        return 1 if stage['name']=='refresh' else 0
    code=h.execute(h.plan('docker','manifest.json',str(out)),out,runner=run)
    ledger=json.loads((out/'workflow.json').read_text())
    assert code==1 and 'refresh' in calls and 'parameter-targets' not in calls
    assert 'report' in calls
    assert next(r for r in ledger['receipts'] if r['stage']=='refresh')['exit_code']==1
    assert (tmp_path/'new.zip').exists()  # Missing canonical archive invokes recovery.
    from oeis_learn.pipeline_validation.bundle import verify
    assert verify(tmp_path/'new.zip')['summary_state']=='FAIL'
    with pytest.raises(ValueError):h.execute(h.plan('docker','manifest.json',str(out)),out,runner=run)


def test_restart_is_forwarded_to_canonical_endpoints(tmp_path):
    plan=helper().plan('podman','manifest.json','run-state/new',reuse_run='run-state/old')
    assert all('--reuse-run' in row['argv'] for row in plan if row['operation']!='verify-bundle')
    # The helper does not read an old PASS flag or suppress qualification itself.
    assert sum(row['operation']=='qualify' for row in plan)==1


def test_recovery_archive_excludes_unreviewed_stage_payloads(tmp_path):
    h=helper();root=tmp_path/'new';root.mkdir()
    (root/'prepare').mkdir();(root/'prepare/candidates.json').write_text('{"secret_target":"hidden"}')
    ledger=dict(receipts=[dict(exit_code=1)])
    (root/'workflow.json').write_text(json.dumps(ledger))
    archive=h.recover(root,ledger)
    import zipfile
    with zipfile.ZipFile(archive) as z:
        assert 'prepare/candidates.json' not in z.namelist()
        assert set(z.namelist())=={'summary.json','workflow.json','manifest.json','diagnostics/index.json'}


def test_original_consumer_uses_fresh_inspection_and_role_verification(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_qualification_io as io
    from oeis_learn.pipeline_validation import generalization_roles as roles
    from oeis_learn.pipeline_validation.identity import seal
    producer=tmp_path/'producer';producer.mkdir()
    q=seal(dict(version='generalization-qualification/v2',images=['cpu','gpu'],source_closure='source',build_dependencies=[]))
    (producer/'qualification.json').write_text(json.dumps(q))
    calls=[]
    def inspect(*args,**kwargs):calls.append('inspect');return {'measured':'new'}
    def invoke(*args,**kwargs):
        calls.append('consume');assert kwargs['settings']['current']=={'measured':'new'}
        assert kwargs['source_roots']['public_evidence']==producer
        return dict(status='PASS',original='original-producer')
    monkeypatch.setattr(io,'current_facts',inspect);monkeypatch.setattr(roles,'project_and_invoke',invoke)
    result=io.consume_original(None,tmp_path,producer,tmp_path/'consumer',required_suites=['cpu','hip'],required_capabilities=['load'])
    assert calls==['inspect','consume'] and result['original']=='original-producer'
    def stale(*args,**kwargs):raise ValueError('current source closure changed')
    monkeypatch.setattr(io,'current_facts',stale)
    with pytest.raises(ValueError,match='source closure'):io.consume_original(None,tmp_path,producer,tmp_path/'stale',required_suites=['cpu'],required_capabilities=['load'])
    assert calls==['inspect','consume']


def test_qualification_reserves_post_run_inspection_without_launching_hip(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_qualification_io as io,runner
    from oeis_learn.experiments import source_snapshot
    observed={}
    def execute(stages,output,**kwargs):
        observed.update(kwargs)
        assert len([s for s in stages if s.adapter=='hip'])==1
        assert [s.seconds for s in stages if s.adapter=='inspect']==[900,900]
        assert sum(s.seconds for s in stages)+kwargs['reserve']<=kwargs['cap']
        return 1
    monkeypatch.setattr(runner,'execute',execute)
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda root:{})
    result=io.produce(None,tmp_path,tmp_path/'qualification',images=['cpu','gpu'],contract={})
    assert result['status']=='FAIL'
    # Two image inspections, one device inspection and authority validation.
    assert observed['cap']+3*120+120+120<=8400


def test_cold_session_preflight_budget_fits_all_four_roles(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import generalization_session as session
    from oeis_learn.generalization.resources import CAPS
    children={name:(tmp_path,{},dict(images=['cpu','gpu'],storage={k:0 for k in CAPS}),None,None)
              for name in ('retained-eval','refresh','parameter-targets','replay-resume')}
    monkeypatch.setattr(session,'load_session',lambda *a:({'identity':'fixture'},children))
    seconds=[]
    def validate(*args,**kw):seconds.append(kw['seconds']);return {'validated':True}
    monkeypatch.setattr(session,'_validate',validate)
    args=SimpleNamespace(operation='preflight',manifest='unused',output='preflight',image='cpu')
    assert session._execute(args,tmp_path,lambda args:object())==0
    assert seconds==[600]*4
    plan=helper().plan('podman','manifest','run')
    assert sum(seconds)+4*5+120<=plan[0]['seconds']
    result=json.loads((tmp_path/'preflight/result.json').read_text())
    assert result['result']['reservations']['minutes']==300


def test_report_startup_and_packaging_fit_parent_budget(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn import container_validation
    from oeis_learn.pipeline_validation import generalization_session as session
    (tmp_path/'run').mkdir()
    observed={}
    def inspect(*a,**kw):observed.update(kw);raise TimeoutError('bounded startup')
    monkeypatch.setattr(container_validation,'inspect_current_image',inspect)
    host=SimpleNamespace(engine=SimpleNamespace(image=lambda image:image))
    args=SimpleNamespace(inputs='run',output='report',image='cpu')
    with pytest.raises(TimeoutError,match='bounded startup'):
        session.report(args,tmp_path,lambda args:host,{}, {'refresh':(None,None,{'images':['cpu','gpu']})})
    assert observed['seconds']==300
    report=next(s for s in helper().plan('podman','manifest','run') if s['operation']=='report')
    assert observed['seconds']+480+2*5+60<=report['seconds']


def test_research_consumer_projects_complete_verified_feature_closure(tmp_path,monkeypatch):
    from tests.unit.test_generalization_feature_acceptance import fixture
    from oeis_learn.pipeline_validation import generalization_qualification_io as io,generalization_roles as roles
    from oeis_learn.pipeline_validation.identity import seal
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.generalization.feature_acceptance import verify_feature
    producer=tmp_path/'producer';producer.mkdir()
    original,facts,feature=fixture(producer)
    (producer/'producer.json').write_text(json.dumps(original))
    (producer/'feature.json').write_text(json.dumps(feature))
    q=seal(dict(version='generalization-qualification/v2',**facts,
                producer={'receipt':io.reference(producer/'producer.json',producer)}))
    (producer/'qualification.json').write_text(json.dumps(q))
    monkeypatch.setattr(io,'current_facts',lambda *a,**kw:facts)
    def invoke(host,image,root,role,artifacts,**kwargs):
        assert kwargs['settings']['research'] is True
        projection=tmp_path/'projected';projection.mkdir()
        for ref in artifacts['public_evidence']:
            target=projection/ref['path'];target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(verify_artifact(producer,ref))
        assert verify_feature(feature,qualification=original['identity'],root=projection)==feature['identity']
        return dict(status='PASS',original=original['identity'])
    monkeypatch.setattr(roles,'project_and_invoke',invoke)
    result=io.consume_original(None,tmp_path,producer,tmp_path/'consumer',required_suites=['hip'],
        required_capabilities=['load'],research=True,feature_receipt=io.reference(producer/'feature.json',producer))
    assert result['original']==original['identity']


def test_canonical_session_cases_report_and_verify_preserve_originals(tmp_path,monkeypatch,capsys):
    """Fake only expensive boundaries; real session/case/receipt/archive routing."""
    import copy
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import generalization_session as session,generalization_commands as commands,generalization_workflow as workflow,generalization_acceptance as cases,generalization_roles as roles,generalization_role as role
    from oeis_learn.pipeline_validation.generalization_cli import main
    from oeis_learn.pipeline_validation.identity import seal,write,digest
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.generalization.qualification import FACTS
    from oeis_learn import container_validation
    from tests.integration.test_generalization_runtime import fixture
    from tests.unit.test_generalization_qualification import producer
    input_root=tmp_path/'input';input_root.mkdir()
    study,runtime,put=fixture(input_root)
    runtime['purpose']='acceptance';runtime['storage']=dict(study_bytes=64<<20,checkpoint_bytes=32<<20,shard_bytes=8<<20,raw_bytes=4<<20,sharing_bytes=1<<20)
    mapping={}
    for name in ('retained-eval','refresh','parameter-targets','replay-resume'):
        chosen=copy.deepcopy(study);sidecar=copy.deepcopy(runtime)
        if name=='parameter-targets':chosen['arms']=['literal','parameterized'];chosen['progression']['stage']='P3'
        if name=='replay-resume':chosen['arms']=['continued_sft','verified_replay'];chosen['progression']['stage']='P4'
        sidecar['training']={a:dict(shards=runtime['training']['refreshed' if a=='refreshed' else 'static']['shards'],batch_size=1,data_mode='refresh' if a=='refreshed' else 'static') for a in chosen['arms']}
        from oeis_learn.generalization.execution import bind,RECIPES,FACTORS
        if name=='retained-eval':
            chosen['progression']['stage']='E0';chosen['arms']=['retained'];chosen['budgets']['target_tokens']=0;chosen['checkpoints']['milestones']=[]
            sidecar['training']={};sidecar['evaluation']=dict(role='selection',checkpoints=[put('fake-checkpoint.json',{})],seconds=60)
        if name=='replay-resume':
            chosen['model']['initialization']=dict(kind='retained',artifact=put('fixture-initializer.json',{}))
            sidecar['replay']=dict(tasks=put('fixture-tasks.json',[]),seconds=30)
            sidecar['evaluation']=dict(role='selection',checkpoints=[],seconds=30)
        stage=chosen['progression']['stage'];chosen['recipe']=RECIPES[stage];chosen['changing_factors']=[FACTORS[stage]]
        chosen,sidecar=bind(chosen,sidecar)
        mapping[name]=dict(manifest=put(name+'.json',chosen),runtime=put(name+'-runtime.json',sidecar))
    from oeis_learn.pipeline_validation.generalization_inputs import assemble_session
    assemble_session(input_root,{k:{kind:ref['path'] for kind,ref in v.items()} for k,v in mapping.items()},'session.json')
    serial=helper();monkeypatch.setattr(serial,'ROOT',input_root)
    assert serial.main(['--engine','podman','--manifest','session.json','--output','dry-run','--dry-run'])==0
    dry=json.loads(capsys.readouterr().out)
    assert len(dry['stages'])==10 and sum(s['operation']=='qualify' for s in dry['stages'])==1
    assert not (input_root/'dry-run').exists()
    run=tmp_path/'run';run.mkdir();qroot=run/'qualify';qroot.mkdir()
    raw=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())['qualification']
    base=qroot/'base.json';base.write_text('{}');base_ref=reference(base,qroot)
    def refs(value):
        if isinstance(value,dict):
            if set(value)=={'path','bytes','sha256'}:return base_ref
            return {k:refs(v) for k,v in value.items()}
        if isinstance(value,list):return [refs(v) for v in value]
        return value
    q=refs(raw);facts={k:q[k] for k in FACTS};original,_=producer(qroot)
    original=seal(dict(original,facts=facts,suites=q['suite_selection'],capabilities=q['covered_capabilities']))
    write(qroot/'producer.json',original)
    q.update(producer=dict(status='PASS',receipt=reference(qroot/'producer.json',qroot),artifacts=original['artifacts']),artifact_inventory=original['artifacts'])
    write(qroot/'qualification.json',seal(q))
    observed=[]
    def qualified(*a,**kw):observed.append('consume-original');return dict(original=original['identity'])
    monkeypatch.setattr(session,'_qualification',qualified)
    monkeypatch.setattr(session,'_validate',lambda *a,**kw:dict(recipe={},initializer=None,target_pairs_verified=1,target_pairs_parameterized=1))
    monkeypatch.setattr(container_validation,'inspect_current_image',lambda *a,**kw:None)
    host=SimpleNamespace(engine=SimpleNamespace(image=lambda x:x))
    def invoke(host,image,workspace,name,artifacts,**kw):
        target=workspace/kw['output']/'role';target.mkdir(parents=True)
        return role.HANDLERS[(__import__('oeis_learn.generalization.role_contracts',fromlist=['Role']).Role(name),kw['operation'])](kw.get('source_roots',{'public_evidence':workspace}),target,kw['settings'],seconds=kw['seconds'])
    monkeypatch.setattr(roles,'project_and_invoke',invoke)
    monkeypatch.setattr(workflow,'_checkpoints',lambda *a:[(dict(identity=digest('checkpoint')),tmp_path)])
    monkeypatch.setattr(workflow,'evaluate',lambda *a,**kw:dict(status='PASS',runs=[dict(generation=dict(status='PASS'))]))
    def train(*a,**kw):return dict(status='PASS',runs=[dict(arm=arm,root=arm,result=dict(status='PASS',updates=2,tokens=10)) for arm in kw['arms']])
    monkeypatch.setattr(commands,'train',train)
    monkeypatch.setattr(cases,'training_proof',lambda *a,**kw:dict(status='PASS',runs=[dict(finite=True,changed=True,tokens=10,updates=2)]))
    monkeypatch.setattr(workflow,'replay',lambda *a,**kw:dict(status='PASS',rounds=1,runs=[dict(cumulative_discoveries=1,paired_proof=dict(status='PASS'),search=dict(status='PASS',admitted=1),paired=train(arms=['continued_sft','verified_replay']),resume=dict(status='PASS',resume_equal=True))]))
    def args(operation,output,**kw):return SimpleNamespace(operation=operation,manifest='input/session.json',runtime=None,output=output,image=None,qualification='run/qualify',reuse_run=None,**kw)
    assert main(args('prepare','run/prepare'),tmp_path,lambda a:host)==0
    for name in helper().CASES:
        assert main(args('acceptance','run/'+name,case=name,inputs='run/prepare'),tmp_path,lambda a:host)==0
    assert observed==['consume-original']*6
    diagnostic=run/'preflight/parameter-targets/validation/role/supervision/process.json'
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text('{"outcome":"COMPLETE"}')
    assert main(args('report','run/report',inputs='run'),tmp_path,lambda a:host)==0
    import zipfile
    with zipfile.ZipFile(run/'report.zip') as z:
        assert any(n.endswith('diagnostics/preflight/parameter-targets/validation/role/supervision/process.json') for n in z.namelist())
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    checked=verify_archive(run/'report.zip')
    assert checked['feature_qualified'] and checked['cases']==dict.fromkeys(helper().CASES,'PASS')
    assert main(SimpleNamespace(operation='verify-bundle',archive='run/report.zip',output='run/verified',image=runtime['images'][0]),tmp_path,lambda a:host)==0
    # Reuse verifies the original case payloads; changing a private worker boundary
    # cannot make it execute again, and stale evidence fails in the fresh root.
    reuse=args('acceptance','again/refresh',case='refresh',inputs='run/prepare');reuse.reuse_run='run'
    monkeypatch.setattr(commands,'train',lambda *a,**kw:pytest.fail('reused learner reran'))
    assert main(reuse,tmp_path,lambda a:host)==0
    (run/'refresh/case/training.json').write_text('{}')
    reuse.output='stale/refresh'
    assert main(reuse,tmp_path,lambda a:host)==1
    assert json.loads((tmp_path/'stale/refresh/result.json').read_text())['status']=='FAIL'


@pytest.mark.parametrize('failure', ['cpu', 'inspection'])
def test_qualification_runner_owns_receipts_and_inspection_gets_fresh_child(tmp_path,monkeypatch,capsys,failure):
    """Keep the real runner and image inspector; replace only process/device work."""
    from types import SimpleNamespace
    from oeis_learn import container_validation
    from oeis_learn.pipeline_validation import generalization_qualification_io as io, supervision
    from oeis_learn.pipeline_validation.bundle import verify
    calls=[]
    class Engine:
        def image(self,image):return image
        def command(self,**kwargs):
            calls.append(kwargs)
            assert kwargs['gpu'] is False
            target=next(Path(src) for src,dest,write in kwargs['mounts'] if dest=='/work/output')
            assert (target.parent/'transitions.jsonl').is_file()
            assert target.name=='inspection'
            return ['fixture-container']
    workspace=Path(__file__).resolve().parents[2]
    host=SimpleNamespace(engine=Engine(),policy=dict(workspace=str(workspace),memory_bytes=2<<30,cpus=1))
    def launch(*args,**kwargs):
        if failure=='inspection':raise RuntimeError('fixture inspection failure')
    def cpu(*args,**kwargs):raise RuntimeError('fixture CPU stop before HIP')
    monkeypatch.setattr(supervision,'launch',launch)
    monkeypatch.setattr(supervision,'reclaim',lambda *args,**kwargs:None)
    monkeypatch.setattr(container_validation,'_stages',cpu)
    out=tmp_path/'qualification'
    result=io.produce(host,workspace,out,images=['cpu-image','gpu-image'],contract={})
    assert result['status']=='FAIL'
    assert [c['image'] for c in calls]==['cpu-image','gpu-image']
    for name in ('image.qualification','image.gpu'):
        receipt=json.loads((out/'pipeline/stages'/name/'receipt.json').read_text())
        assert receipt['state']==('PASS' if failure=='cpu' else 'FAIL')
        if failure=='inspection':assert 'fixture inspection failure' in receipt['reason']
    assert not (out/'qualification.json').exists()
    assert verify(out/'pipeline.zip')['summary_state']=='FAIL'
    log=capsys.readouterr().out
    assert ('fixture CPU stop before HIP' if failure=='cpu' else 'fixture inspection failure') in log
