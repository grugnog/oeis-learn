"""Installed closed role wiring, not an invented success receipt."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal
from oeis_learn.generalization.progressive.projections import describe, project
from oeis_learn.pipeline_validation.progressive_roles import REGISTRY


def run(root, operation, data, kind, channel):
    atomic_write(root/'input.json',seal(dict(version='discovery-channel/v1',kind=kind,role=REGISTRY[operation].role,data=data)))
    project(root,root/'projected',operation,{channel:[describe(root,'input.json')]},registry=REGISTRY)
    code='from pathlib import Path;from oeis_learn.pipeline_validation.progressive_role import execute;import sys;execute(sys.argv[1],inputs=Path(sys.argv[2]),output=Path(sys.argv[3]))'
    p=subprocess.run([sys.executable,'-c',code,operation,str(root/'projected'),str(root/'out')],env=os.environ.copy(),capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    return json.loads((root/'out/result.json').read_text())


def test_installed_source_pilot_reports_independent_deficits(tmp_path):
    from tests.contract.test_progressive_sources import source
    result=run(tmp_path,'source-pilot',dict(version='progressive-source-pilot-input/v1',sources=[source()],roles=['training']),'source','sources')
    assert result['mechanical_status']=='PARTIAL'
    assert not json.loads((tmp_path/'out/feasibility.json').read_text())['primary_scaling_eligible']


def test_installed_verifier_recomputes_public_bundle(tmp_path):
    from tests.contract.test_progressive_bundle import fixture
    from oeis_learn.generalization.progressive.report import build
    request=fixture(tmp_path)
    build(tmp_path,tmp_path/'report.zip',request)
    result=run(tmp_path,'verify',dict(version='progressive-bundle-input/v1',artifact=reference(tmp_path,'report.zip')),'public_evidence','public')
    assert result['mechanical_status']=='COMPLETE'
    verdict=json.loads((tmp_path/'out/verification.json').read_text())
    assert verdict['status']=='VERIFIED' and verdict['decision']=='INCOMPLETE'


def test_verifier_projection_rejects_changed_original_archive(tmp_path):
    atomic_write(tmp_path/'input.json',seal(dict(version='discovery-channel/v1',kind='public_evidence',role='verifier',data=dict(version='progressive-bundle-input/v1',artifact=dict(path='missing.zip',sha256='sha256:'+'1'*64,bytes=10)))))
    with pytest.raises((ValueError,FileNotFoundError)):
        project(tmp_path,tmp_path/'projected','verify',{'public':[describe(tmp_path,'input.json')]},registry=REGISTRY)


@pytest.mark.parametrize('operation', ['qualify','acceptance'])
def test_host_commands_report_missing_immutable_plan_before_touching_engine(tmp_path, operation, capsys):
    from types import SimpleNamespace
    from tests.helpers.progressive_fixture import blueprint
    from tests.unit.test_progressive_freeze import campaign
    from oeis_learn.pipeline_validation.progressive_commands import main
    document=blueprint() if operation=='qualify' else campaign()
    atomic_write(tmp_path/'document.json',document)
    def forbidden(args):
        raise AssertionError('engine contacted without prospective plan')
    args=SimpleNamespace(operation=operation,blueprint='document.json',campaign='document.json',output='out')
    assert main(args,tmp_path,forbidden)==2
    result=json.loads(capsys.readouterr().out)
    assert result['reason']==('QUALIFICATION' if operation=='qualify' else 'ACCEPTANCE')+'_PLAN_DEFICIT'
    assert 'implementation' not in result['missing_evidence']


def test_changed_or_unbounded_qualification_plan_is_rejected_before_dispatch(tmp_path):
    from oeis_learn.pipeline_validation.progressive_handoff import named_plan
    from tests.helpers.progressive_fixture import blueprint
    plan=seal(dict(version='progressive-qualification-plan/v1',matrix={},base_qualification={},seconds=3601))
    atomic_write(tmp_path/'plan.json',plan)
    b=blueprint();b['payload']['references']=[dict(name='qualification_plan',role='custodian',kind='policy',artifact=reference(tmp_path,'plan.json'))]
    with pytest.raises(ValueError,match='allowance'):
        named_plan(tmp_path,b,'qualification_plan','progressive-qualification-plan/v1',{'matrix','base_qualification','seconds'})


@pytest.mark.parametrize('change',['cpu','missing','skip','scope'])
def test_device_receipt_cannot_infer_coverage_from_a_pass_label(tmp_path,change):
    import xml.etree.ElementTree as ET
    from oeis_learn.pipeline_validation.progressive_device_acceptance import NODES,verify_producer
    from oeis_learn.generalization.progressive.contracts import digest
    request=seal(dict(campaign=digest('c'),runtime=digest('r'),source=digest('s'),image=digest('i'),engine='docker',hardware={},profile={}))
    suite=ET.Element('testsuite')
    for node in NODES:
        module,name=node.split('::')
        ET.SubElement(suite,'testcase',classname=module[:-3].replace('/','.'),name=name)
    if change=='missing': suite.remove(suite[0])
    if change=='skip': ET.SubElement(suite[0],'skipped')
    ET.ElementTree(suite).write(tmp_path/'cases.xml')
    producer=seal(dict(version='progressive-device-acceptance/v1',request=digest(request),
        **{k:request[k] for k in ('campaign','runtime','source','image','engine','hardware','profile')},
        status='PASS',device='cpu' if change=='cpu' else 'actual_HIP',cases=list(NODES),
        junit=reference(tmp_path,'cases.xml'),actual_elapsed_us=10))
    if change=='scope': request=seal(dict(request,campaign=digest('changed')))
    with pytest.raises(ValueError): verify_producer(tmp_path,producer,request)


def test_selected_model_claim_requires_original_matching_device_cases(tmp_path):
    from tests.integration.test_progressive_acceptance import hardware_fixture
    from oeis_learn.pipeline_validation.progressive_handoff import validate_device_proofs
    from oeis_learn.generalization.progressive.contracts import digest
    request,_,ref=hardware_fixture(tmp_path)
    atomic_write(tmp_path/'matrix.json',request)
    q=dict(covered_capabilities=['model:'+digest({k:v for k,v in request['cases'][0]['profile']['model'].items() if k!='seed'})],
           producer=dict(artifacts=[reference(tmp_path,'matrix.json'),ref]))
    validate_device_proofs(tmp_path,q,**{k:request[k] for k in ('source','image','engine')})
    q['covered_capabilities']=['model:'+digest('uncovered-model')]
    with pytest.raises(ValueError,match='model'): validate_device_proofs(tmp_path,q,**{k:request[k] for k in ('source','image','engine')})


def test_mandatory_device_selection_has_no_cpu_fallback(monkeypatch):
    from oeis_learn.pipeline_validation.progressive_device_acceptance import device
    from oeis_learn.pipeline_validation import progressive_qualification as q
    monkeypatch.setenv('OEIS_PROGRESSIVE_ACCEPTANCE_DEVICE','actual_HIP')
    def absent(): raise RuntimeError('actual HIP unavailable')
    monkeypatch.setattr(q,'require_hip',absent)
    with pytest.raises(RuntimeError,match='HIP'): device('acceptance')


def sustained_fixture():
    return dict(version='progressive-sustained-plan/v1',tasks=[dict(slot='first',active_us=3600000000),dict(slot='second',active_us=3600000000)],
                interval_us=60000000,interrupt_task='first',interrupt_after_us=1000000,finalization_us=1800000000,
                sensors={k:dict(available=False,reason='not exposed by selected runtime') for k in ('memory','disk','thermal','throughput')},adoption='DIAGNOSTIC_ONLY')


@pytest.mark.parametrize('change',['none','overspend','late','progress','duration','sensor'])
def test_sustained_preflight_never_claims_simulated_elapsed_time(change):
    from oeis_learn.generalization.progressive.preflight import sustained
    plan=sustained_fixture()
    if change=='overspend': plan['tasks'][0]['active_us']*=100
    if change=='late': plan['interrupt_after_us']=plan['tasks'][0]['active_us']
    if change=='progress': plan['interval_us']+=1
    if change=='duration': plan['tasks'][0]['active_us']=86400000001
    if change=='sensor': plan['sensors']['thermal']['reason']=None
    if change=='none':
        result=sustained(plan,allowance_us=7200000000)
        assert not result['executable'] and result['actual_elapsed_us'] is None
    else:
        with pytest.raises(ValueError): sustained(plan,allowance_us=7200000000)


def test_outcome_blind_preparation_binds_original_panels_without_freezing(tmp_path):
    from tests.unit.test_progressive_cohorts import fixture
    from tests.unit.test_progressive_sensitivity import scenario
    from oeis_learn.generalization.progressive.preflight import prepare
    panel,exposure=fixture(tmp_path)
    atomic_write(tmp_path/'panel.json',panel);atomic_write(tmp_path/'exposure.json',exposure)
    result=prepare(tmp_path,cohorts=[reference(tmp_path,'panel.json')],exposure=reference(tmp_path,'exposure.json'),
                   sensitivity_design=dict(seed=13,repeats=1,scenarios=[scenario(groups=1,family_sizes=[1])]))
    assert not result['executable'] and result['status']=='PENDING_QUALIFICATION_AND_FREEZE'


@pytest.mark.parametrize('change',['complete','missing_case','wrong_engine'])
def test_qualification_host_orchestration_requires_original_complete_device_roster(tmp_path,monkeypatch,change):
    """Boundary-only producers; simulated facts never qualify real hardware."""
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_handoff as handoff
    from oeis_learn.pipeline_validation import generalization_qualification_io as io
    from oeis_learn.generalization.qualification import FACTS
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.pipeline_validation.progressive_qualification import matrix
    from oeis_learn.pipeline_validation.progressive_acceptance import HARDWARE_CHECKS
    from tests.unit.test_progressive_training_policy import profile
    from tests.helpers.progressive_fixture import blueprint
    from oeis_learn.generalization.progressive.contracts import digest
    base=tmp_path/'base';base.mkdir()
    source={'boundary':'not actual qualification'}
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda root:source)
    atomic_write(base/'source.json',source);atomic_write(base/'build.json',{})
    template=json.loads((Path(__file__).resolve().parents[1]/'fixtures/generalization/authority.json').read_text())['qualification']
    facts={k:template[k] for k in FACTS}
    facts.update(source_closure=reference(base,'source.json'),build_dependencies=[reference(base,'build.json')],images=[digest('cpu'),digest('gpu')])
    facts['runtime']['execution_policy']=digest('synthetic-policy')
    e=seal(dict(version='generalization-device-evidence/v1',device='hip',loaded_weights='boundary-only',finite_forward=True,finite_backward=True,parameters_changed=True,resume_equal=True,execution_policy=facts['runtime']['execution_policy']))
    atomic_write(base/'device.json',e)
    producer=seal(dict(version='generalization-producer/v1',status='PASS',facts=facts,suites=['bigint-cpu','bigint-gpu'],capabilities=['weight-load','finite-update','native-exact'],artifacts=[reference(base,'device.json')],dependencies=[],feature_acceptance=False))
    atomic_write(base/'producer.json',producer)
    q=seal(dict(version='generalization-qualification/v2',**facts,status='PASS',suite_selection=producer['suites'],covered_capabilities=producer['capabilities'],producer=dict(receipt=reference(base,'producer.json'),status='PASS',artifacts=producer['artifacts']),artifact_inventory=io.public_closure(base,producer)))
    atomic_write(base/'qualification.json',q)
    request=matrix([profile(.1)],source=digest(source),image=facts['images'][1],engine='docker',hardware={'boundary':True})
    atomic_write(tmp_path/'matrix.json',request)
    plan=seal(dict(version='progressive-qualification-plan/v1',matrix=reference(tmp_path,'matrix.json'),base_qualification=reference(tmp_path,'base/qualification.json'),seconds=60))
    atomic_write(tmp_path/'plan.json',plan)
    b=blueprint();b['payload']['references']=[dict(name='qualification_plan',role='custodian',kind='policy',artifact=reference(tmp_path,'plan.json'))]
    atomic_write(tmp_path/'blueprint.json',seal(b))
    monkeypatch.setattr(io,'current_facts',lambda *a,**k:facts)
    calls=[]
    def stage(host,root,image,output,argv,*,gpu,seconds):
        calls.append((gpu,seconds,image))
        # Execute the real emitted installed command and delta dispatcher. Only
        # GPU process execution/facts are doubled at the engine boundary.
        from oeis_learn.pipeline_validation import progressive_acceptance as acceptance
        def process(command,**kwargs):
            if 'synchronize' in command[2]:return 0
            selected=next(c for c in request['cases'] if c['identity']==command[-2])
            if change=='missing_case' and selected==request['cases'][-1]:return 1
            measures=dict.fromkeys(HARDWARE_CHECKS,True)|dict(parameters=1,peak_allocated_bytes=1,cleanup_allocated_bytes=0,baseline_allocated_bytes=0,elapsed_us=1)
            atomic_write(Path(command[-1])/'producer.json',seal(dict(version='progressive-hip-case/v1',source=request['source'],
                scope=digest([request[k] for k in ('source','image','engine','hardware')]+[selected]),device='actual_HIP',status='PASS',
                requested=list(HARDWARE_CHECKS),executed=list(HARDWARE_CHECKS),measurements=measures)))
            return 0
        with monkeypatch.context() as boundary:
            boundary.setattr(acceptance,'_current_hardware',lambda r:None)
            boundary.setattr(acceptance,'_bounded_process',process)
            try:exec(argv[-1].replace('/work/output',str(output)),{})
            except SystemExit as exc:
                if exc.code:raise ValueError('installed qualification batch incomplete')
    monkeypatch.setattr(handoff,'stage',stage)
    host=SimpleNamespace(engine=SimpleNamespace(name='podman' if change=='wrong_engine' else 'docker'))
    args=SimpleNamespace(blueprint='blueprint.json',output='out')
    if change=='complete':
        original_bytes=(base/'qualification.json').read_bytes()
        result,code=handoff.qualify_command(args,tmp_path,lambda a:host)
        assert code==0 and len(calls)==1 and calls[0][0] is True
        assert 0<calls[0][1]<=60 and calls[0][2]==facts['images'][1]
        assert (tmp_path/'out/device/cases').is_dir()
        assert (base/'qualification.json').read_bytes()==original_bytes
        assert json.loads((tmp_path/'out/qualification.json').read_text())['covered_capabilities']!=q['covered_capabilities']
        script="""import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[2])
from oeis_learn.pipeline_validation.progressive_commands import main
def absent(args): raise RuntimeError('engine unavailable')
assert main(SimpleNamespace(operation='qualify',blueprint='blueprint.json',output='no-site-out'),Path(sys.argv[1]),absent)==3  # sibling output cannot reset original qualification allowance
assert not {'torch','numpy','optuna','jsonschema'} & set(sys.modules)
"""
        probe=subprocess.run([sys.executable,'-S','-c',script,str(tmp_path),str(Path(__file__).resolve().parents[2]/'src')],capture_output=True,text=True,timeout=30)
        assert probe.returncode==0,probe.stderr

    else:
        with pytest.raises(ValueError): handoff.qualify_command(args,tmp_path,lambda a:host)
        assert not (tmp_path/'out/qualification.json').exists()


def load_boundary(path):
    return json.loads(path.read_text())


def test_gpu_launcher_configures_numerical_policy_before_python_import(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_handoff as h
    from oeis_learn.pipeline_validation.generalization_execution import startup_environment
    expected=startup_environment()
    code='import os,sys; assert "torch" not in sys.modules; assert all(os.environ.get(k)==v for k,v in '+repr(expected)+'.items())'
    def launcher(host,image,output,argv,**kwargs):
        assert kwargs['gpu'] and kwargs['seconds']==5
        subprocess.run(argv,cwd=tmp_path,check=True,timeout=5)
    monkeypatch.setattr(h.runpy,'run_path',lambda path:{'container_stage':launcher})
    h.stage(None,tmp_path,'test-image',tmp_path/'out',[sys.executable,'-S','-c',code],gpu=True,seconds=5)


def test_failed_packaging_cannot_leave_a_launch_receipt(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_device_acceptance as d
    def failed(*args,**kwargs):
        assert not (tmp_path/'acceptance.json').exists()
        raise ValueError('complete metadata cannot fit')
    monkeypatch.setattr(d,'export_package',failed)
    with pytest.raises(ValueError,match='metadata'):
        d._publish_receipt(tmp_path,seal(dict(version='boundary-only')),tmp_path,{})
    assert not (tmp_path/'acceptance.json').exists()
    assert (tmp_path/'acceptance.pending.json').is_file()


def test_installed_refreshed_learner_consumes_native_verified_new_structure(tmp_path,fitting_client):
    import time
    from oeis_learn.generalization.progressive.contracts import load,record,digest
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture,invoke_cpu
    from oeis_learn.pipeline_validation.progressive_learner import compatibility
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    from oeis_learn.generalization.progressive.generator import structure_id
    from oeis_learn.pipeline_validation.progressive_device_acceptance import device
    c,tasks,channel,_=fixture(tmp_path,interrupt=False)
    package=load(tmp_path/'training-input.json')
    original=[]
    for row in package['shards'][0]:
        values=fitting_client.execute(row['source'],list(map(str,range(100))),deadline=time.monotonic()+4)
        assert values['outcome']=='ok' and values['values']==row['values']
        original.append(seal(dict(row,native_verified=True)))
    program='(program (return (select (lt $index (int 3)) (add $index (int 2)) (int 9))))'
    new,reason=verify_draw(dict(source=program,ordinal=0,draw_identity=digest('new-native-structure'),source_policy=digest('diagnostic-refresh'),structure=structure_id(program),detector_limit='forced diagnostic'),client=fitting_client)
    assert new is not None,reason
    assert new['native_verified']
    from oeis_learn.generalization.progressive.models import resolve_profile
    p=package['profile']
    p=resolve_profile(p['base_recipe'],shape=p['shape'],seed=2,dropout=p['model']['dropout'])
    package=seal(dict(package,profile=p,update_policy=seal(dict(package['update_policy'],model_profile=p['identity'])),shards=[original]+[original[:3]+[new]]*5,refresh_updates=1))
    atomic_write(tmp_path/'refresh-input.json',package)
    channel=seal(dict(channel,data=dict(channel['data'],artifact=reference(tmp_path,'refresh-input.json'),compatibility=compatibility(package))))
    atomic_write(tmp_path/'refresh-channel.json',channel)
    target=2*16*len(original[3]['tokens'])
    for plan in c['payload']['tasks']:
        plan.update(arm='refreshed',seed=2)
        plan['initializer']['compatibility']=compatibility(package)
    c['payload']['tasks'][0]['token_target']=target
    c=seal(c)
    task=record('task',dict(c['payload']['tasks'][0],campaign=c['identity']),created_at=c['created_at'],purpose=c['purpose'],track=c['track'])
    result,out=invoke_cpu(tmp_path,c,task,channel,ordinal=0,channel_path='refresh-channel.json')
    assert result['mechanical_status']=='COMPLETE',result
    assert result['metrics']['device']==device('acceptance')
    diversity=load(out/'role/diversity.json')
    assert diversity['counts']['consumed']['canonical_structures']>=2


def test_public_init_runs_only_outcome_blind_bound_preflight(tmp_path,capsys):
    from types import SimpleNamespace
    from tests.unit.test_progressive_cohorts import fixture
    from tests.unit.test_progressive_sensitivity import scenario
    from oeis_learn.pipeline_validation.progressive_commands import main
    from oeis_learn.generalization.progressive.contracts import load
    panel,exposure=fixture(tmp_path)
    atomic_write(tmp_path/'panel.json',panel);atomic_write(tmp_path/'exposure.json',exposure)
    atomic_write(tmp_path/'design.json',dict(seed=13,repeats=1,scenarios=[scenario(groups=1,family_sizes=[1])]))
    plan=seal(dict(version='progressive-preflight-plan/v1',cohorts=[reference(tmp_path,'panel.json')],exposure=reference(tmp_path,'exposure.json'),sensitivity=reference(tmp_path,'design.json')))
    atomic_write(tmp_path/'prior/plan.json',plan)
    atomic_write(tmp_path/'prior/retained-inputs.json',seal(dict(version='discovery-retained-inputs/v1',references=[dict(name='preflight_plan',role='custodian',kind='policy',artifact=reference(tmp_path,'prior/plan.json'))])))
    args=SimpleNamespace(operation='init',reference_campaign='prior',output='inputs')
    assert main(args,tmp_path,None)==0
    result=load(tmp_path/'inputs/preparation-preflight.json')
    assert result['sensitivity']['status']=='VALIDATED_UNEXECUTED'
    assert not result['executable']
    assert not any(name.startswith('artifact:') for name in load(tmp_path/'inputs/inputs.json')['unresolved'])
