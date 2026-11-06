import importlib.util
from pathlib import Path
import pytest


def helper():
    spec=importlib.util.spec_from_file_location('pac_serial',Path(__file__).with_name('run.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_scope_order_gates_research_and_no_gpu_before_admission():
    h=helper()
    assert h.stage_order()==['admission','qualification','calibration','acceptance']
    order=h.stage_order(research=True)
    assert order.index('admission')<order.index('baseline')<order.index('train-B')
    assert not any('D' in n for n in order)
    assert 'train-D' in h.stage_order(research=True,arms=['B','C','D'])


def test_wrapper_delegates_to_canonical_cli_with_explicit_engine_and_qualified_source():
    h=helper();calls=[]
    result=h.invoke('podman','evaluate','study.json','output',qualification='producer',extra=['--checkpoints','A=descriptor.json'],
        seconds=90000,runner=lambda argv,**kw:calls.append((argv,kw)) or {'status':'PASS'})
    assert result['status']=='PASS'
    argv,options=calls[0]
    assert argv[1:6]==['scripts/foundation_run.py','--engine','podman','generalization','evaluate']
    assert options['seconds']==90000
    with pytest.raises(ValueError):h.canonical_command('auto','train','study','out')
    with pytest.raises(ValueError):h.canonical_command('podman','train','study','out')


@pytest.mark.parametrize('control',['reference-only','fresh-control'])
def test_real_baseline_branch_claims_owned_root_and_verifies_retained_weights(tmp_path,monkeypatch,control):
    import json
    from oeis_learn.pipeline_validation.identity import digest,seal,write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_pac_evaluation import checkpoint_request
    h=helper();source=tmp_path/'original';source.mkdir();(source/'weights.pt').write_bytes(b'original retained tensors')
    descriptor=seal(dict(version='generalization-checkpoint/v1',kind='full_state',weights=reference(source/'weights.pt',source),
        state=None,initialization=digest('initial'),parent=None,tokens=25000000,updates=100,pins=['latest'],compatibility=digest('old control')))
    study=dict(control_mode=control,arms=['A','B','C'] if control=='fresh-control' else ['B','C'])
    runtime=dict(control=dict(checkpoint=descriptor['identity']))
    write(tmp_path/'prepared/study.json',study);write(tmp_path/'prepared/runtime.json',runtime)
    write(tmp_path/'context.json',dict(checkpoints=dict(static=dict(weights_root='original',checkpoint=descriptor))))
    receipts=dict(acceptance=dict(resolved_manifest='prepared/study.json',resolved_runtime='prepared/runtime.json'))
    plan=dict(stages=[dict(extra=['context.json','20261016'])])
    stage=dict(name='baseline',operation='evaluate',manifest='@admission');target=tmp_path/'baseline';target.mkdir();seen=[]
    def invoke(plan,stage,target,workspace,receipts):
        request=stage['extra'][stage['extra'].index('--checkpoints')+1]
        arm,root,actual=checkpoint_request(workspace,request,study,runtime,{}, {})
        assert arm==('A-reference' if control=='fresh-control' else 'A')
        assert root==target/'private/retained' and actual==descriptor
        assert (root/'weights.pt').read_bytes()==(source/'weights.pt').read_bytes()
        seen.append(request);return dict(status='PASS')
    monkeypatch.setattr(h,'_canonical_command_stage',invoke)
    assert h._canonical_stage(plan,stage,target,tmp_path,receipts)['status']=='PASS'
    assert seen
    # A new helper invocation cannot overwrite previously owned retained bytes.
    with pytest.raises(FileExistsError):h._canonical_stage(plan,stage,target,tmp_path,receipts)
    (source/'weights.pt').write_bytes(b'changed tensor bytes')
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError):h._canonical_stage(plan,stage,other,tmp_path,receipts)
    assert len(seen)==1


@pytest.mark.parametrize('minimal',[False,True])
def test_helper_exception_is_private_locally_and_bound_in_normal_or_minimal_report(tmp_path,monkeypatch,minimal):
    import json,zipfile
    from oeis_learn.pipeline_validation.identity import digest
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    from oeis_learn.pipeline_validation import generalization_pac_export as exporter
    h=helper();order=h.stage_order()
    plan=h.prepare_plan(tmp_path,engine='podman',stages=[dict(name=n,operation='prepare',manifest='unused',seconds=3600,
        extra=[],requires=[] if i==0 else [order[i-1]]) for i,n in enumerate(order)],inputs=[],source=digest('source'),images=[digest('cpu'),digest('gpu')])
    if minimal:monkeypatch.setattr(exporter,'collect',lambda *a,**kw:(_ for _ in ()).throw(ValueError('private rich export problem')))
    def fail(stage,target):raise FileNotFoundError('/home/private/operator/frozen-input')
    result=h.execute_plan(plan,tmp_path/'failed',workspace=tmp_path,runner=fail)
    assert result['status']=='FAIL' and result['errors'][0]['kind']=='FileNotFoundError'
    payload=tmp_path/'failed/admission/helper-failure.json'
    assert payload.is_file() and (tmp_path/'failed/admission/private/helper-exception.json').is_file()
    public=json.loads(payload.read_text());assert public['location']['function']=='execute_plan'
    assert verify_archive(tmp_path/'failed/report.zip')['software_status']=='FAIL'
    with zipfile.ZipFile(tmp_path/'failed/report.zip') as z:
        assert json.loads(z.read('failures/helper-admission.json'))==public
        assert all(b'/home/private' not in z.read(n) for n in z.namelist())
    original=(tmp_path/'failed/workflow.json').read_bytes()
    h.recover(tmp_path/'failed',tmp_path/'recovered')
    assert (tmp_path/'failed/workflow.json').read_bytes()==original
    payload.write_text('{}')
    bad=h.recover(tmp_path/'failed',tmp_path/'unverified')
    with zipfile.ZipFile(bad) as z:
        assert 'failures/helper-admission.json' not in z.namelist()
        assert any('helper' in s for s in json.loads(z.read('evidence.json'))['omissions'])
    # Even a resealed workflow cannot turn an arbitrary string into public data.
    from oeis_learn.pipeline_validation.identity import seal,write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    write(payload,seal(dict(public,failure='private teacher payload')),exclusive=False)
    workflow=h._read(tmp_path/'failed/workflow.json')
    workflow['receipts'][0]['artifacts']=[reference(payload,payload.parent) if r['path']=='helper-failure.json' else r
        for r in workflow['receipts'][0]['artifacts']]
    write(tmp_path/'failed/workflow.json',seal(workflow),exclusive=False)
    unsafe=h.recover(tmp_path/'failed',tmp_path/'unsafe')
    with zipfile.ZipFile(unsafe) as z:
        assert 'failures/helper-admission.json' not in z.namelist()
        assert all(b'private teacher payload' not in z.read(n) for n in z.namelist())


def test_campaign_resume_verifies_outputs_and_recovery_never_launches(tmp_path):
    import json
    from oeis_learn.pipeline_validation.identity import write,seal,digest
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    h=helper();write(tmp_path/'inputs.json',{'source':'mechanical'})
    stages=[dict(name=name,operation='prepare' if name=='admission' else 'qualify',manifest='inputs.json',
        seconds=3600,extra=[],requires=[] if i==0 else [h.stage_order()[i-1]]) for i,name in enumerate(h.stage_order())]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[reference(tmp_path/'inputs.json',tmp_path)],
        source=digest('source'),images=[digest('cpu'),digest('gpu')])
    calls=[]
    def runner(stage,target):
        calls.append(stage['name']);write(target/'observed.json',{'status':'PASS'})
        return dict(status='PASS',process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True),
            producer=digest(stage['name']),artifacts=[reference(target/'observed.json',target)])
    one=h.execute_plan(plan,tmp_path/'one',workspace=tmp_path,runner=runner)
    assert one['status']=='PARTIAL' and calls==h.stage_order()
    calls.clear()
    two=h.execute_plan(plan,tmp_path/'two',workspace=tmp_path,runner=runner,previous=tmp_path/'one')
    assert two['status']=='PARTIAL' and calls==[]
    immutable=(tmp_path/'one/workflow.json').read_bytes()
    h.recover(tmp_path/'one',tmp_path/'recovered')
    assert calls==[] and (tmp_path/'one/workflow.json').read_bytes()==immutable
    (tmp_path/'one/admission/observed.json').write_text('changed')
    calls.clear();h.execute_plan(plan,tmp_path/'three',workspace=tmp_path,runner=runner,previous=tmp_path/'one')
    assert calls==['admission'] # identical regenerated producer permits descendant reuse
    changed=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=plan['inputs'],source=digest('new-source'),images=plan['images'])
    calls.clear();h.execute_plan(changed,tmp_path/'four',workspace=tmp_path,runner=runner,previous=tmp_path/'one')
    assert calls==h.stage_order()


@pytest.mark.parametrize('outcome',['FAIL','PARTIAL','BLOCKED','INCONCLUSIVE','INTERRUPTED'])
def test_failure_suppresses_descendants_and_preserves_primary(tmp_path,outcome):
    from oeis_learn.pipeline_validation.identity import digest
    h=helper();stages=[dict(name=name,operation='prepare',manifest='inputs',seconds=3600,extra=[],requires=[] if i==0 else [h.stage_order()[i-1]]) for i,name in enumerate(h.stage_order())]
    plan=h.prepare_plan(tmp_path,engine='docker',stages=stages,inputs=[],source=digest('src'),images=[digest('cpu'),digest('gpu')])
    calls=[]
    def runner(stage,target):calls.append(stage['name']);return dict(status=outcome,process=dict(outcome='COMPLETE',returncode=1),cleanup=dict(acknowledged=True,within_bound=True),producer=digest('failed'),artifacts=[])
    result=h.execute_plan(plan,tmp_path/'run',workspace=tmp_path,runner=runner)
    assert calls==['admission'] and result['status']!='PASS'
    assert result['errors'][0]['stage']=='admission' and (tmp_path/'run/report.zip').is_file()


def test_export_failure_after_primary_failure_retains_both_and_last_resort(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.identity import digest
    h=helper();stages=[dict(name=name,operation='prepare',manifest='inputs',seconds=3600,extra=[],requires=[] if i==0 else [h.stage_order()[i-1]]) for i,name in enumerate(h.stage_order())]
    plan=h.prepare_plan(tmp_path,engine='docker',stages=stages,inputs=[],source=digest('src'),images=[digest('cpu'),digest('gpu')])
    def failed(stage,target):raise RuntimeError('primary private details')
    def broken(*a):raise OSError('secondary exporter failure')
    monkeypatch.setattr(h,'recover',broken)
    result=h.execute_plan(plan,tmp_path/'failed',workspace=tmp_path,runner=failed)
    assert [e['kind'] for e in result['errors']]==['RuntimeError','OSError']
    assert 'private details' not in str(result) and (tmp_path/'failed/workflow.json').is_file()


def test_interrupted_campaign_resumes_only_unfinished_closure_and_binds_identities(tmp_path,capsys):
    from oeis_learn.pipeline_validation.identity import digest,write,seal
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    h=helper();stages=[dict(name=name,operation='prepare',manifest='inputs',seconds=3600,extra=[],requires=[] if i==0 else [h.stage_order()[i-1]]) for i,name in enumerate(h.stage_order())]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('src'),images=[digest('cpu'),digest('gpu')])
    calls=[]
    def runner(stage,target):
        calls.append(stage['name'])
        if stage['name']=='calibration':raise KeyboardInterrupt()
        write(target/'result.json',seal(dict(status='PASS',source=plan['source'],images=plan['images'],engine=plan['engine'])))
        return dict(status='PASS',producer=digest(stage),artifacts=[reference(target/'result.json',target)],
            process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True))
    first=h.execute_plan(plan,tmp_path/'first',workspace=tmp_path,runner=runner)
    assert first['status']=='INTERRUPTED' and calls==['admission','qualification','calibration']
    original=(tmp_path/'first/workflow.json').read_bytes()
    def completed(stage,target):
        calls.append(stage['name']);write(target/'result.json',{'status':'PASS'})
        return dict(status='PASS',producer=digest(stage),artifacts=[reference(target/'result.json',target)],
            process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True))
    calls.clear();resumed=h.execute_plan(plan,tmp_path/'resumed',workspace=tmp_path,runner=completed,previous=tmp_path/'first')
    assert calls==['calibration','acceptance'] and (tmp_path/'first/workflow.json').read_bytes()==original
    assert all(e['binding'].startswith('sha256:') for e in resumed['receipts'])
    assert verify_archive(tmp_path/'resumed/report.zip')['software_status']=='PARTIAL'
    assert 'measured progress unavailable' in capsys.readouterr().out


def test_bootstrap_failure_packages_independent_negative_evidence(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    h=helper()
    from oeis_learn.experiments import source_snapshot
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda root:{})
    args=SimpleNamespace(output='blocked',engine='podman',fresh_control=False,bandit_only=False,research=False,seed=7)
    assert h.bootstrap_failure(args,FileNotFoundError('/private/original missing'),workspace=tmp_path)==1
    verdict=verify_archive(tmp_path/'blocked/bootstrap-failure/report.zip')
    assert verdict['software_status']=='BLOCKED' and not verdict['feature_qualified']
    import zipfile
    with zipfile.ZipFile(tmp_path/'blocked/bootstrap-failure/report.zip') as archive:
        assert all(b'/private/original' not in archive.read(n) for n in archive.namelist())


def test_failed_train_resume_requires_verified_full_state_and_exact_config(tmp_path,monkeypatch):
    import json
    from oeis_learn.pipeline_validation import generalization_commands as commands,generalization_runtime as runtime_module
    from oeis_learn.pipeline_validation.identity import digest,write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_execution import bind_compatibility
    h=helper();stage=dict(name='train-C',manifest='@acceptance')
    base=tmp_path/'base';base.mkdir()
    write(base/'bundle.json',dict(identity=digest('bundle')));write(base/'recipe.json',{})
    write(base/'study.json',dict(seed=7));write(base/'runtime.json',dict(prepared=dict(training=reference(base/'bundle.json',base)),recipe=reference(base/'recipe.json',base)))
    receipts={'acceptance':dict(resolved_manifest='base/study.json',resolved_runtime='base/runtime.json')}
    old=tmp_path/'old';root=old/'train-C/execution/training-C/role/private/learner/checkpoints';root.mkdir(parents=True)
    path=root/'generation/manifest.json';path.parent.mkdir()
    config=dict(batch_size=32)
    data=digest(dict(prepared=digest('bundle'),arm='C',batch_size=32))
    contract=digest(dict(config=config,data=data,**bind_compatibility('cuda',{})))
    write(path,dict(contract=contract));ref=reference(path,root);write(root/'generalization-latest.json',ref)
    calls=[]
    def verified(workspace,value):calls.append(value);return root,ref,[]
    monkeypatch.setattr(commands,'_resume_inputs',verified)
    monkeypatch.setattr(runtime_module,'learner_config',lambda *a,**kw:config)
    result=h._model_resume({},stage,old,tmp_path,receipts)
    assert result==str(path.relative_to(tmp_path)) and calls==[result]
    write(path,dict(contract=digest('wrong')),exclusive=False)
    with pytest.raises(ValueError):h._model_resume({},stage,old,tmp_path,receipts)


def test_failed_admission_exports_original_quota_ledger_without_qualifying_it(tmp_path):
    import json,zipfile
    from oeis_learn.pipeline_validation.identity import digest,write
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    h=helper();order=h.stage_order()
    stages=[dict(name=n,operation='prepare',manifest='unavailable',seconds=3600,extra=[],requires=[] if i==0 else [order[i-1]]) for i,n in enumerate(order)]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('src'),images=[digest('cpu'),digest('gpu')])
    quota=dict(families=[dict(family=digest('family'),draws=4,before_exclusion=3,available=1,required=2,deficit=1)],native_completed=1,native_seconds=.1)
    def failed(stage,target):
        write(target/'execution/execution/role/private/assembly/resolved/availability.json',quota)
        raise ValueError('scientific quota deficit')
    result=h.execute_plan(plan,tmp_path/'failed',workspace=tmp_path,runner=failed)
    assert result['status']=='FAIL' and result['receipts'][0]['artifacts']
    assert verify_archive(tmp_path/'failed/report.zip')['software_status']=='FAIL'
    with zipfile.ZipFile(tmp_path/'failed/report.zip') as archive:
        assert json.loads(archive.read('admission/availability.json'))==quota


def test_documented_operator_block_routes_once_and_resume_recover_are_separate(tmp_path,monkeypatch,capsys):
    h=helper();monkeypatch.setattr(h,'ROOT',tmp_path);calls=[]
    from oeis_learn.pipeline_validation.identity import seal,write
    def build(args):
        calls.append(('prepare',args.engine,args.prepared,args.pilot,args.output,args.research))
        return tmp_path/args.output,dict(identity='fixture-plan',engine='podman')
    def execute(plan,output,**kw):
        calls.append(('execute',plan['identity'],str(output.relative_to(tmp_path)),str(kw.get('previous'))))
        return dict(status='PASS')
    monkeypatch.setattr(h,'build_plan',build);monkeypatch.setattr(h,'execute_plan',execute)
    monkeypatch.setattr(h,'recover',lambda previous,output:calls.append(('recover',str(previous),str(output))) or output.with_suffix('.zip'))
    original=['--prepared','run-state/generalization/fresh-p1-v2-20261103T035534Z','--pilot','run-state/generalization/fresh-p1-hourly-20261104T093555Z']
    assert h.main(['run','--engine','podman',*original,'--output','run-state/generalization/pac-012-acceptance'])==0
    assert calls[0]==('prepare','podman',original[1],original[3],'run-state/generalization/pac-012-acceptance/inputs/plan.json',False)
    assert calls[1][0:3]==('execute','fixture-plan','run-state/generalization/pac-012-acceptance/campaign')
    write(tmp_path/'plan.json',dict(identity='fixture-plan',engine='podman'));(tmp_path/'previous').mkdir()
    assert h.main(['resume','--plan','plan.json','--previous','previous','--output','resumed'])==0
    assert calls[-1][2]=='resumed'
    assert h.main(['recover','--previous','previous','--output','recovered'])==0
    assert calls[-1][0]=='recover' and 'Verified recovered report' in capsys.readouterr().out
    with pytest.raises(ValueError):h.main(['run','--engine','docker','--plan','plan.json','--output','wrong'])


def test_real_nonzero_subprocess_transports_sealed_terminal_instead_of_hiding_cause(tmp_path):
    import sys,json
    h=helper();target=tmp_path/'role';target.mkdir()
    terminal=target/'result.json'
    code="from pathlib import Path; import json,sys; from oeis_learn.pipeline_validation.identity import seal; Path(sys.argv[1]).write_text(json.dumps(seal(dict(status='FAIL',reason='QuotaDeficit')))); raise SystemExit(1)"
    value=h.launch_terminal([sys.executable,'-c',code,str(terminal)],tmp_path/'supervision',terminal,seconds=30)
    assert value['reason']=='QuotaDeficit' and value['status']=='FAIL'
    process=json.loads((tmp_path/'supervision/process.json').read_text())
    assert process['returncode']==1 and process['outcome']=='FAIL'


def test_nested_quota_cause_and_wrapper_error_both_survive_read_only_recovery(tmp_path):
    import json,zipfile
    from oeis_learn.pipeline_validation.identity import digest,seal,write
    h=helper();order=h.stage_order()
    stages=[dict(name=n,operation='prepare',manifest='unavailable',seconds=3600,extra=[],requires=[] if i==0 else [order[i-1]]) for i,n in enumerate(order)]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('source'),images=[digest('cpu'),digest('gpu')])
    def failed(stage,target):
        write(target/'execution/execution/role/result.json',seal(dict(status='FAIL',reason='QuotaDeficit',error='completed post-exclusion quota deficit; no timeout/reseed retry',secret='private teacher')))
        raise RuntimeError('outer process exited; /home/private')
    result=h.execute_plan(plan,tmp_path/'original',workspace=tmp_path,runner=failed)
    assert result['status']=='FAIL'
    original=(tmp_path/'original/workflow.json').read_bytes()
    archive=h.recover(tmp_path/'original',tmp_path/'recovery')
    assert (tmp_path/'original/workflow.json').read_bytes()==original
    with zipfile.ZipFile(archive) as z:
        evidence=json.loads(z.read('evidence.json'))
        assert evidence['errors'][0]['kind']=='QuotaDeficit'
        assert any(e['kind']=='RuntimeError' for e in evidence['errors'])
        detail=json.loads(z.read('failures/admission.json'))
        assert detail['failures'][0]['detail']['code']=='structural_quota_deficit'
        assert all(b'private teacher' not in z.read(n) and b'/home/private' not in z.read(n) for n in z.namelist())


def test_minimal_export_after_rich_failure_retains_old_nested_cause(tmp_path,monkeypatch):
    import json,zipfile
    from oeis_learn.pipeline_validation.identity import seal,write,digest
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation import generalization_pac_export as export
    h=helper();order=h.stage_order()
    stages=[dict(name=n,operation='prepare',manifest='unavailable',seconds=3600,extra=[],requires=[] if i==0 else [order[i-1]]) for i,n in enumerate(order)]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('source'),images=[digest('cpu'),digest('gpu')])
    old=tmp_path/'old';path=old/'admission/execution/execution/role/result.json'
    write(path,seal(dict(status='FAIL',reason='QuotaDeficit',error='completed post-exclusion quota deficit; no timeout/reseed retry')))
    write(old/'plan.json',plan)
    write(old/'workflow.json',seal(dict(plan=plan['identity'],status='FAIL',errors=[dict(stage='admission',kind='RuntimeError',status='FAIL')],
        receipts=[dict(stage='admission',status='FAIL',directory='admission',artifacts=[reference(path,old/'admission')])])))
    original=(old/'workflow.json').read_bytes()
    monkeypatch.setattr(export,'collect',lambda *a:(_ for _ in ()).throw(ValueError('bad rich closure /private/path')))
    archive=export.export(old,tmp_path/'recovered')
    assert archive.name=='recovered-minimal.zip' and (old/'workflow.json').read_bytes()==original
    with zipfile.ZipFile(archive) as z:
        errors=json.loads(z.read('evidence.json'))['errors']
        assert [e['kind'] for e in errors]==['QuotaDeficit','RuntimeError','ValueError']
        assert 'failures/admission.json' in z.namelist()
        assert all(b'/private/path' not in z.read(n) for n in z.namelist())
