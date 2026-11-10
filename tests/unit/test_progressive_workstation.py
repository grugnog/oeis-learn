"""Host orchestration boundaries, never simulated hardware qualification."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from oeis_learn.pipeline_validation.progressive_workstation_state import Session, Blocked, read, write


def test_independent_stage_continues_failed_dependency_blocks_and_zip_survives(tmp_path):
    s=Session(tmp_path,'run-state/session',dict(source='test'),'rehearse')
    visited=[]
    with s.own():
        def fail(): raise RuntimeError('first original cause')
        s.step('bad',fail)
        s.step('dependent',lambda:pytest.fail('dependency ran'),dependencies=('rehearse/bad',))
        s.step('independent',lambda:(visited.append('yes'),[]))
    assert visited==['yes'] and s.code()==1
    with zipfile.ZipFile(s.folder/'share/rehearse.zip') as z:
        journal=json.loads(z.read('journal.json'))
        assert journal['steps']['rehearse/bad']['reason']=='first original cause'
        assert journal['steps']['rehearse/dependent']['status']=='BLOCKED'
        assert journal['steps']['rehearse/independent']['status']=='COMPLETE'
        assert json.loads(z.read('manifest.json'))['qualified'] is False


def test_failed_and_unfinished_work_never_gets_fresh_allowance(tmp_path):
    s=Session(tmp_path,'run-state/session',{},'rehearse')
    with s.own():
        s.journal['steps']['rehearse/lost']=dict(status='RUNNING'); s.save()
    other=Session(tmp_path,'run-state/session',{},'rehearse')
    with other.own():
        other.step('lost',lambda:pytest.fail('restarted without entitlement'))
        assert other.journal['steps']['rehearse/lost']['status']=='INTERRUPTED'


def test_changed_source_blocks_before_action_and_retains_archive(tmp_path):
    with Session(tmp_path,'run-state/s',dict(source='a'),'rehearse').own(): pass
    s=Session(tmp_path,'run-state/s',dict(source='b'),'rehearse')
    with pytest.raises(Blocked,match='SOURCE'):
        with s.own(): pytest.fail('changed source ran')
    assert read(s.folder/'session.json')['binding']['source']=='a'
    with zipfile.ZipFile(s.folder/'share/rehearse.zip') as z:
        assert 'SOURCE_OR_RESOURCE_SCOPE_CHANGED' in z.read('journal.json').decode()


def test_completed_stage_replays_original_and_rejects_tampering(tmp_path):
    target=tmp_path/'producer.json'; target.write_text('original')
    s=Session(tmp_path,'run-state/s',{},'rehearse')
    with s.own():
        assert s.step('one',lambda:('ok',[target]))=='ok'
        assert s.step('one',lambda:pytest.fail('reexecuted'))=='ok'
        target.write_text('changed')
        with pytest.raises(ValueError,match='changed'):
            s.step('one',lambda:pytest.fail('changed receipt accepted'))


def test_packaging_excludes_private_tensors_and_symlinks_retains_log_tail(tmp_path):
    s=Session(tmp_path,'run-state/s',{},'rehearse')
    with s.own():
        for name in ('private/truth.json','checkpoints/weights.pt','inputs/truth.log','work/truth.log','model.pt'):
            p=s.folder/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('PRIVATE')
        (s.folder/'alias.log').symlink_to(s.folder/'inputs/truth.log')
        (s.folder/'failure.log').write_bytes(b'begin\n'+b'x'*(5<<20)+b'\nlast error')
    with zipfile.ZipFile(s.folder/'share/rehearse.zip') as z:
        assert not any('PRIVATE' in z.read(n).decode(errors='ignore') for n in z.namelist())
        assert z.read('failure.log.excerpt').endswith(b'last error')
        manifest=json.loads(z.read('manifest.json'))
        assert {v['reason'].split(';')[0] for v in manifest['omissions']}=={'symlink','size bound'}


def test_real_system_python_dry_run_imports_no_ml_and_has_no_side_effects(tmp_path):
    root=Path(__file__).resolve().parents[2]
    code="""import sys
sys.path.insert(0,'src')
from oeis_learn.pipeline_validation.progressive_workstation import main
from pathlib import Path
assert main(Path.cwd(),['--engine','podman','--session',sys.argv[1],'--dry-run','rehearse'])==0
assert not {'torch','numpy','optuna','jsonschema'} & set(sys.modules)
"""
    target=tmp_path/'absent'
    p=subprocess.run([sys.executable,'-S','-c',code,str(target)],cwd=root,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    assert not target.exists()


def test_fresh_rehearsal_keeps_inventory_when_engine_missing(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_workstation as w
    from oeis_learn.experiments import source_snapshot
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda root:{'test':'scope'})
    def absent(self): raise RuntimeError('podman unavailable')
    monkeypatch.setattr(w.Workflow,'host',absent)
    def inventory(self,name,operation,**kw):
        assert operation=='init'
        return self.s.step(name,lambda:({'missing':['retained-checkpoint']},[]))
    monkeypatch.setattr(w.Workflow,'run_command',inventory)
    code=w.main(tmp_path,['--engine','podman','--session','run-state/session','rehearse','--reference-campaign','missing'])
    assert code==1
    with zipfile.ZipFile(tmp_path/'run-state/session/share/rehearse.zip') as z:
        j=json.loads(z.read('journal.json'))
        assert j['steps']['rehearse/environment']['reason']=='podman unavailable'
        assert j['steps']['rehearse/inventory']['status']=='COMPLETE'
        assert j['steps']['rehearse/image-gpu']['status']=='BLOCKED'


def test_actual_cpu_assets_match_selected_profile(tmp_path):
    from oeis_learn.pipeline_validation.progressive_workstation_fixture import assets,profile
    from oeis_learn.generalization.progressive.contracts import load
    assets(tmp_path)
    package=load(tmp_path/'mechanics/training-input.json')
    assert package['profile']==profile()==load(tmp_path/'profile.json')
    assert package['scope']=='DIAGNOSTIC_ONLY'
    assert (tmp_path/'initial.pt').stat().st_size>1000


def test_qualification_continues_after_one_case_failure(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_acceptance as a
    from oeis_learn.pipeline_validation.progressive_qualification import matrix
    from tests.integration.test_progressive_acceptance import hardware_fixture
    from oeis_learn.generalization.progressive.contracts import atomic_write,digest,seal
    request,producer,_=hardware_fixture(tmp_path)
    request=matrix([request['cases'][0]['profile']],**{k:request[k] for k in ('source','image','engine','hardware')},input_lengths=(4,20),microbatches=(7,))
    monkeypatch.setattr(a,'_current_hardware',lambda r:None)
    calls=[]
    def execute(argv,**kwargs):
        if len(argv)==3: return 0 # Explicit boundary-only device health double.
        calls.append(argv[-2])
        if len(calls)==1:return 1
        case=next(c for c in request['cases'] if c['identity']==argv[-2])
        value=seal(dict(producer,scope=digest([request[k] for k in ('source','image','engine','hardware')]+[case])))
        atomic_write(Path(argv[-1])/'producer.json',value)
        return 0
    monkeypatch.setattr(a,'_bounded_process',execute)
    result=a.run_hardware_delta(tmp_path,request,[],tmp_path/'batch',seconds=30)
    assert len(calls)==2 and result['status']=='FAIL'
    assert len(result['producers'])==1 and len(result['failures'])==1


def test_device_case_failure_does_not_prevent_independent_case(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_acceptance as a
    from oeis_learn.pipeline_validation.progressive_case_batch import pytest_cases
    import xml.etree.ElementTree as ET
    calls=[]
    def execute(argv,**kwargs):
        if len(argv)==3: return 0
        node=argv[-1]; calls.append(node)
        if len(calls)==1:return 1
        module,name=node.split('::')
        suite=ET.Element('testsuite'); ET.SubElement(suite,'testcase',classname=module[:-3].replace('/','.'),name=name)
        ET.ElementTree(suite).write(argv[argv.index('--junitxml')+1])
        return 0
    monkeypatch.setattr(a,'_bounded_process',execute)
    nodes=['tests/test_a.py::test_a','tests/test_b.py::test_b']
    assert pytest_cases(nodes,tmp_path,env={},seconds=30,cwd=tmp_path)==1
    assert calls==nodes
    cases=ET.parse(tmp_path/'cases.xml').findall('.//testcase')
    assert len(cases)==2 and cases[0].find('error') is not None and cases[1].find('error') is None


def test_real_fixture_prepare_and_freeze_with_boundary_only_qualification(tmp_path,monkeypatch):
    """Original qualification is doubled; all fixture/prepare/freeze checks run."""
    from tests.unit.test_progressive_freeze import test_complete_diagnostic_freeze_binds_original_producer_and_task_roster
    from oeis_learn.pipeline_validation.progressive_workstation_fixture import assets,preparation_inputs,profile
    from oeis_learn.generalization.progressive.contracts import load,reference,digest,seal
    from oeis_learn.pipeline_validation.progressive_capability import prepare
    from oeis_learn.pipeline_validation.progressive_commands import freeze_command
    from oeis_learn.pipeline_validation import progressive_handoff
    from types import SimpleNamespace
    test_complete_diagnostic_freeze_binds_original_producer_and_task_roster(tmp_path,monkeypatch)
    folder=tmp_path/'assets'; folder.mkdir(); assets(folder)
    model=digest({k:v for k,v in profile()['model'].items() if k!='seed'})
    qroot=tmp_path/'qualification'
    producer=load(qroot/'producer.json'); q=load(qroot/'qualification.json')
    producer['capabilities']=q['covered_capabilities']=['model:'+model,'language:bigint-body/v1']
    write(qroot/'producer.json',seal(producer)); q['producer']['receipt']=reference(qroot,'producer.json')
    write(qroot/'qualification.json',seal(q))
    monkeypatch.setattr(progressive_handoff,'validate_device_proofs',lambda *a,**kw:None)
    b=preparation_inputs(tmp_path,tmp_path/'plans',qualification=qroot,assets_root=folder)
    _,code=prepare(SimpleNamespace(blueprint=str(b.relative_to(tmp_path)),qualification='qualification',output='prepared-real'),tmp_path)
    assert code==0
    result,code=freeze_command(SimpleNamespace(blueprint=str(b.relative_to(tmp_path)),qualification='qualification',prepared='prepared-real',output='frozen-real'),tmp_path)
    assert code==0,result
    campaign=load(tmp_path/'frozen-real/campaign.json')
    assert campaign['purpose']=='acceptance' and campaign['track']=='synthetic_strict'
    assert load(tmp_path/'frozen-real/runtime.json')['payload']['scope']['device']=='hip'


def test_container_failure_inspection_precedes_removal_preserves_original(tmp_path,monkeypatch):
    import runpy
    from contextlib import nullcontext
    from types import SimpleNamespace
    root=Path(__file__).resolve().parents[2]
    module=runpy.run_path(str(root/'scripts/qualify_bigint_phase5.py'))
    stage=module['container_stage']; events=[]
    class Engine:
        name='podman'
        def command(self,**kwargs):return ['test']
        def call(self,*args,**kw):
            events.append('inspect'); return SimpleNamespace(stdout=json.dumps([dict(State=dict(OOMKilled=True),HostConfig=dict(NetworkMode='none',ReadonlyRootfs=True,Privileged=False),Mounts=[])]).encode())
        def remove(self,name): events.append('remove')
    host=SimpleNamespace(engine=Engine(),policy=dict(memory_bytes=1<<30,cpus=1),gpu_lock=nullcontext)
    def fail(*args,**kw): raise RuntimeError('original producer failed')
    monkeypatch.setitem(stage.__globals__,'launch',fail)
    with pytest.raises(RuntimeError,match='original producer'):
        stage(host,'image',tmp_path,['test'])
    assert events==['inspect','remove']
    assert json.loads((tmp_path/'container-inspect.json').read_text())[0]['State']['OOMKilled']


def test_cpu_work_continues_when_gpu_image_build_fails(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_workstation as w
    s=Session(tmp_path,'run-state/s',{'reference_campaign':'reference'},'rehearse')
    workflow=w.Workflow(s,SimpleNamespace(phase='rehearse'))
    monkeypatch.setattr(workflow,'run_command',lambda *a,**kw:None)
    monkeypatch.setattr(workflow,'host',lambda:SimpleNamespace(engine=SimpleNamespace(facts=lambda:{}),policy={}))
    def images():
        s.journal['steps']['rehearse/image-qualification']={'status':'COMPLETE','value':'cpu'}
        s.journal['steps']['rehearse/image-gpu']={'status':'FAILED'}
        return {'qualification':'cpu','gpu':None}
    monkeypatch.setattr(workflow,'images',images)
    visited=[]
    monkeypatch.setattr(workflow,'installed',lambda name,*a,**kw:visited.append(name))
    with s.own(): workflow.rehearse()
    assert visited==['assessment-contracts','assets']
    assert s.journal['steps']['rehearse/base-qualification']['status']=='BLOCKED'


def test_packaging_failure_does_not_replace_original_exception(tmp_path,monkeypatch):
    s=Session(tmp_path,'run-state/s',{},'rehearse')
    def fail(): raise OSError('archive disk failure')
    monkeypatch.setattr(s,'package',fail)
    with pytest.raises(RuntimeError,match='original cause'):
        with s.own(): raise RuntimeError('original cause')
    assert [e['type'] for e in read(s.folder/'journal.json')['errors']]==['RuntimeError','PACKAGING_FAILED']


@pytest.mark.parametrize('healthy,expected',[(True,[0,1]),(False,[0])])
def test_calibration_failure_keeps_independent_results_and_blocks_bad_device(tmp_path,monkeypatch,healthy,expected):
    """Double only execution boundaries; exercise real batch and partial receipts."""
    from oeis_learn.pipeline_validation import progressive_calibration as c, progressive_acceptance as a
    from oeis_learn.pipeline_validation import progressive_qualification as q, generalization_execution as execution, progressive_learner as learner
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.generalization.progressive.contracts import digest,reference,seal,load
    cases=[]
    for i in range(2):
        key=digest(i); write(tmp_path/f'input-{i}.json',{'compatibility':key})
        cases.append(dict(compatibility=key,artifact=reference(tmp_path,f'input-{i}.json')))
    request=seal(dict(source=digest({}),image='image',engine='podman',hardware={},execution_policy={},
        seconds=60,caps={'disk_bytes':64<<20},maximum_inodes=1000,cases=cases))
    write(tmp_path/'request.json',request)
    monkeypatch.setattr(c,'validate_request',lambda r:None)
    monkeypatch.setattr(source_snapshot,'source_snapshot',lambda:{})
    monkeypatch.setattr(q,'require_hip',lambda:None)
    monkeypatch.setattr(q,'hardware_facts',lambda:{})
    monkeypatch.setattr(execution,'verify_execution',lambda:{})
    monkeypatch.setattr(learner,'compatibility',lambda p:p['compatibility'])
    monkeypatch.setenv('OEIS_QUALIFICATION_IMAGE','image'); monkeypatch.setenv('OEIS_QUALIFICATION_ENGINE','podman')
    calls=[]
    def execute(argv,**kw):
        if len(argv)==3:return 0 if healthy else 1
        i=int(argv[-2]); calls.append(i)
        if i==0:return 1
        write(tmp_path/'cases'/cases[i]['compatibility'][7:]/'producer.json',seal({'boundary_only':True}))
        return 0
    monkeypatch.setattr(a,'_bounded_process',execute)
    with pytest.raises(ValueError,match='incomplete'):c.run_inside(tmp_path)
    assert calls==expected
    partial=load(tmp_path/'partial.json')
    assert partial['status']=='FAIL' and len(partial['producers'])==int(healthy)
    assert not (tmp_path/'producer.json').exists()


@pytest.mark.parametrize('result_status',['PARTIAL','FAILED'])
def test_research_resume_uses_original_run_and_reports_even_failure(tmp_path,monkeypatch,result_status):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_workstation as w, supervision
    s=Session(tmp_path,'run-state/s',{},'run-research')
    args=SimpleNamespace(phase='run-research',engine='podman',memory_gib=64,cpus=8,render_node=None)
    workflow=w.Workflow(s,args)
    write(s.folder/'prepare-research/frozen/campaign.json',{'payload':{'caps':{'elapsed_us':10000000}}})
    write(workflow.base/'execution-0000/result.json',{'mechanical_status':'PARTIAL'})
    (workflow.base/'execution-0000-verified').mkdir()
    calls=[]
    def launch(argv,*a,**kw):
        calls.append(argv)
        write(tmp_path/argv[argv.index('--output')+1]/'result.json',{'mechanical_status':result_status})
    monkeypatch.setattr(supervision,'launch',launch)
    reported=[]
    monkeypatch.setattr(workflow,'run_command',lambda name,op,**kw:reported.append(op))
    with s.own():
        s.journal['steps']['prepare-research/acceptance']={'status':'COMPLETE'}
        workflow.research()
    assert 'resume' in calls[0]
    assert calls[0][calls[0].index('--run')+1]=='run-state/s/run-research/execution-0000'
    assert calls[0][calls[0].index('--output')+1]=='run-state/s/run-research/execution-0001'
    assert reported==['report']
    assert s.journal['steps']['run-research/invocation-0001']['status']==('COMPLETE' if result_status=='PARTIAL' else 'FAILED')
