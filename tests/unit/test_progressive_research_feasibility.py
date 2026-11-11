"""Timing/resource orchestration with explicit device process test boundaries."""
import copy
import json
import pytest
from oeis_learn.generalization.progressive.contracts import atomic_write,digest,load,reference,seal
from oeis_learn.pipeline_validation import progressive_research_feasibility as f


def request_fixture(root):
    from oeis_learn.fresh.bigint_config import defaults,resolve
    from oeis_learn.generalization.progressive.models import resolve_profile,SHAPES
    recipe=resolve(defaults())
    profiles=[resolve_profile(recipe,shape=s,dropout=d,seed=17) for s in SHAPES for d in (0,.1,.2)]
    cases=[dict(profile=p,control='random',methods=['correct']) for p in profiles]
    cases.extend(dict(profile=profiles[0],control=control,methods=list(f.METHODS)) for control in ('random','retained_D'))
    atomic_write(root/'private/training-samples.json',[dict(id=str(i),indices=list(map(str,range(100))),values=[str(n+i*1000) for n in range(100)]) for i in range(2)])
    req=seal(dict(version=f.VERSION,source=digest({}),image=digest('image'),engine='podman',hardware={},
        execution_policy=digest('policy'),qualification={'path':'explicit-test-boundary'},support=digest('training'),
        samples=reference(root,'private/training-samples.json'),checkpoint={'path':'explicit-test-boundary','sha256':digest('checkpoint'), 'bytes':1},
        cases=[seal(c) for c in cases],seconds=f.SECONDS,scope='training-only test boundary',safety_multiplier=4))
    atomic_write(root/'request.json',req)
    return req


def producer(root,req,index,us=1000):
    # Original-format artifacts from the explicit process boundary; never HIP
    # qualification evidence. Their complete closure is still verified normally.
    from oeis_learn.pipeline_validation.progressive_evidence import generation_timing
    case=req['cases'][index];base=root/'cases'/str(index)/('normal' if us==1000 else 'slow')
    samples=json.loads((root/req['samples']['path']).read_text())
    methods=['equal_budget' if m=='correct' else m for m in case['methods']]
    records={r['id']:dict(arm_evidence={m:dict(complete=True) for m in methods},costs={}) for r in samples}
    generated=seal(dict(version='discovery-candidates/v1',model=digest('process boundary model'),
        checkpoint=req['checkpoint']['sha256'] if case['control']=='retained_D' else None,
        model_control=case['control'],recipe=digest(case['profile']),methods=case['methods'],
        prompts=digest(sorted([dict(id=r['id'],indices=r['indices'][:20],values=r['values'][:20]) for r in samples],key=lambda r:r['id'])),
        records=records,clock={},counts={},elapsed_us=us))
    atomic_write(base/'generation.json',generated);generation_ref=reference(base,base/'generation.json')
    atomic_write(base/'scoring.json',seal(dict(version='discovery-scoring/v1',candidates=generation_ref['sha256'],
        outcomes={r['id']:{m:dict(complete=True) for m in methods} for r in samples},denominator=2,
        scientific_status='UNMEASURED',claim='finite_agreement_only')))
    atomic_write(base/'timing.json',generation_timing(generated))
    p=seal(dict(version='progressive-research-evaluation-case/v1',request=req['identity'],case=case['identity'],
        device='actual_HIP',status='PASS',samples=2,generation_us=us,scoring_us=us,
        peak_allocated_bytes=1,producer=generation_ref,scoring=reference(base,base/'scoring.json'),
        timing=reference(base,base/'timing.json'),actual_elapsed_us=2*us))
    atomic_write(base/'producer.json',p)
    # The batch subprocess contract locates this fixed path. Keep its relative
    # original references valid by putting the entire case closure at that root.
    if us==1000:
        import shutil
        for path in base.iterdir():shutil.copyfile(path,base.parent/path.name)
        return reference(root,base.parent/'producer.json')
    return reference(root,base/'producer.json')


def test_full_measured_roster_is_required_and_infeasibility_does_not_reduce_it(tmp_path):
    from tests.unit.test_progressive_research_roster import fixture
    from oeis_learn.generalization.progressive.research_roster import construct
    req=request_fixture(tmp_path);roster=construct(*fixture())
    rows=[dict(index=i,case=c['identity'],status='PASS',producer=producer(tmp_path,req,i)) for i,c in enumerate(req['cases'])]
    result=seal(dict(version='progressive-research-evaluation-run/v1',request=req['identity'],status='PASS',cases=rows,actual_elapsed_us=100000))
    atomic_write(tmp_path/'evaluation-cases.json',result)
    value=f.estimates(tmp_path,req,roster)
    assert value['status']=='READY' and not value['assessment_exposed']
    assert len(value['tasks'])==sum(r['operation'] in {'generate','score'} for r in roster['tasks'])
    rows[0]['producer']=producer(tmp_path,req,0,us=100000000)
    (tmp_path/'evaluation-cases.json').write_text(json.dumps(seal(dict(result,cases=rows))))
    slow=f.estimates(tmp_path,req,roster,publish=False)
    assert slow['status']=='BLOCKED' and slow['deficits']
    assert [r['task'] for r in slow['tasks']]==[r['task'] for r in value['tasks']]
    (tmp_path/'evaluation-cases.json').write_text(json.dumps(seal(dict(result,cases=rows[:-1]))))
    with pytest.raises(ValueError,match='complete original'):f.estimates(tmp_path,req,roster)
    with pytest.raises(ValueError):f.validate_request(seal(dict(req,cases=req['cases'][:-1])))


@pytest.mark.parametrize('health_ok',[True,False])
def test_failed_device_case_preserves_evidence_and_continues_only_when_safe(tmp_path,monkeypatch,health_ok):
    req=request_fixture(tmp_path)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_qualification.require_hip',lambda:None)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_qualification.hardware_facts',lambda:{})
    monkeypatch.setattr('oeis_learn.pipeline_validation.generalization_execution.verify_execution',lambda:req['execution_policy'])
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.source_snapshot',lambda:{})
    monkeypatch.setenv('OEIS_QUALIFICATION_IMAGE',req['image']);monkeypatch.setenv('OEIS_QUALIFICATION_ENGINE','podman')
    calls=[]
    def process(argv,**kw):
        if 'synchronize' in argv[2]:return 0 if health_ok else 1
        index=int(argv[4]);calls.append(index)
        if index==0:return 1
        producer(tmp_path,req,index)
        return 0
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_acceptance._bounded_process',process)
    with pytest.raises(ValueError,match='incomplete'):f.run_inside(tmp_path)
    result=load(tmp_path/'evaluation-cases.json')
    assert result['status']=='FAIL' and len(result['cases'])==14
    assert result['cases'][0]['status']=='FAIL'
    assert {r['status'] for r in result['cases'][1:]}==({'PASS'} if health_ok else {'BLOCKED'})
    assert calls==(list(range(14)) if health_ok else [0])
    assert load(tmp_path/'failures.json')[0]['error']=='RuntimeError'


def test_resource_finalization_only_shrinks_disk_and_reports_slow_choices(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_research_resources as r
    from tests.unit.test_progressive_continuation import campaign
    from tests.helpers.progressive_fixture import caps
    c=campaign();plans=c['payload']['tasks']
    for p in plans:
        p['caps']['disk_bytes']=1<<30
        p['initializer'].pop('parent_slot',None)
        p['initializer'].update(mode='random',artifact=None)
    atomic_write(tmp_path/'policy.json',r.policy())
    payload=dict(task_plans=plans,references=[dict(name='research_resource_policy',kind='policy',artifact=reference(tmp_path,'policy.json'))],
        checkpoint={},checkpoint_root='test-boundary',graph=c['payload']['graph'],caps=caps(disk_bytes=3<<30),deficits=[])
    for node in payload['graph']['nodes']:node['caps']['disk_bytes']=1<<30
    scope=plans[0]['initializer']['compatibility']
    cases=[dict(compatibility=scope,package=dict(shards=[[dict(tokens=[1,2])]],update_policy=dict(effective_batch=16)))]
    producers=[dict(compatibility=scope,checkpoint_bytes=1024,measurements=dict(update_us=100,probe_us=100,checkpoint_us=100))]
    monkeypatch.setattr('oeis_learn.generalization.progressive.storage.checkpoint_size',lambda *a,**k:(2048,1))
    original=copy.deepcopy(payload)
    result=r.finalize(tmp_path,payload,cases,producers)
    assert result['status']=='READY' and result['checkpoint_bytes']==2048
    assert result['storage_after_bytes']<result['storage_before_bytes']
    for before,after in zip(original['task_plans'],payload['task_plans']):
        assert {k:v for k,v in before['caps'].items() if k!='disk_bytes'}=={k:v for k,v in after['caps'].items() if k!='disk_bytes'}
    slow=copy.deepcopy(original);producers[0]['measurements']['update_us']=10**12
    result=r.finalize(tmp_path,slow,cases,producers)
    assert result['status']=='BLOCKED' and slow['deficits']==['training_time:first','training_time:second']


@pytest.mark.parametrize('failed_timing',[False,True])
def test_registered_handoff_uses_canonical_dependencies_and_stops_before_expensive_calibration(tmp_path,monkeypatch,failed_timing):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_research_launch as launch
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,write
    from tests.unit.test_progressive_research_roster import fixture
    from oeis_learn.generalization.progressive.research_roster import construct
    session=Session(tmp_path,'run-state/research',{},'prepare-research');base=session.folder/'prepare-research'
    for name in ('role-roster','sensitivity'):
        session.journal['steps']['prepare-research/'+name]=dict(status='COMPLETE')
    roster=construct(*fixture());atomic_write(base/'roster/private/role-roster.json',roster)
    (tmp_path/'weights.pt').write_bytes(b'original byte identity; model loading boundary is simulated')
    atomic_write(tmp_path/'manifest.json',dict(retained=dict(checkpoints=dict(D=dict(root='.',descriptor=dict(weights=reference(tmp_path,'weights.pt')))))))
    atomic_write(tmp_path/'support.json',{'explicit_native_producer_boundary':True})
    data=dict(manifest=reference(tmp_path,'manifest.json'));materials=dict(support=reference(tmp_path,'support.json'))
    for name,value in [('registered/registered-data.json',data),('materials/materials.json',materials),('research-round-design.json',{})]:
        atomic_write(base/name,value)
    calls=[]
    def qualify(w,*args):
        atomic_write(base/'qualification/qualification.json',{'explicit_qualification_boundary':True})
        atomic_write(base/'matrix/profiles.json',{'profiles':[]})
        session.journal['steps']['prepare-research/qualify']=dict(status='COMPLETE')
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_research_qualification.qualify',qualify)
    def req(root,output,support,profiles,q,*,checkpoint):
        assert support=={'explicit_native_producer_boundary':True}
        r=request_fixture(output)
        r=seal(dict(r,qualification=q,checkpoint=checkpoint));write(output/'request.json',r)
        return r
    monkeypatch.setattr(f,'request',req)
    def stage(host,root,image,output,argv,**kw):
        calls.append('timing')
        assert kw['gpu'] and kw['readonly_inputs']==[(tmp_path/'weights.pt','/work/inputs/weights.pt')]
        if failed_timing:raise RuntimeError('explicit failed HIP process boundary')
        r=load(output/'request.json')
        rows=[dict(index=i,case=c['identity'],status='PASS',producer=producer(output,r,i)) for i,c in enumerate(r['cases'])]
        atomic_write(output/'evaluation-cases.json',seal(dict(version='progressive-research-evaluation-run/v1',request=r['identity'],status='PASS',cases=rows,actual_elapsed_us=1000)))
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_handoff.stage',stage)
    def construct_process(argv,*args,**kw):
        calls.append('construction')
        assert argv[1]=='-S' and 'progressive_research_bindings' in argv[3]
        for flag in ('--data','--design','--materials','--execution-contract'):
            assert (tmp_path/argv[argv.index(flag)+1]).is_file()
        atomic_write(base/'construction/construction-summary.json',{'explicit_construction_boundary':True})
        atomic_write(base/'construction/blueprint.json',{'explicit_construction_boundary':True})
    monkeypatch.setattr('oeis_learn.pipeline_validation.supervision.launch',construct_process)
    def command(name,operation,*,dependencies,seconds,**kwargs):
        def action():
            calls.append(operation)
            assert kwargs['output'].startswith('run-state/research/prepare-research/')
            if operation=='prepare':assert kwargs['blueprint'].endswith('/construction/blueprint.json')
            return {'explicit_command_boundary':operation},[]
        return session.step(name,action,dependencies=dependencies)
    w=SimpleNamespace(s=session,root=tmp_path,base=base,host=lambda:object(),run_command=command)
    with session.own():launch.prepare(w,data,{},materials)
    assert calls==(['timing'] if failed_timing else ['timing','construction','prepare','freeze','acceptance'])
    accepted=session.journal['steps']['prepare-research/acceptance']['status']
    assert accepted==('BLOCKED' if failed_timing else 'COMPLETE')
    import zipfile
    with zipfile.ZipFile(session.folder/'share/prepare-research.zip') as z:
        assert not any('/private/' in n for n in z.namelist())
        if not failed_timing:assert any(n.endswith('evaluation-feasibility.json') for n in z.namelist())


def test_timing_request_cannot_replace_reference_control_with_another_primary_profile(tmp_path):
    req=request_fixture(tmp_path)
    changed=copy.deepcopy(req)
    changed['cases'][-1]=seal(dict(changed['cases'][-1],profile=changed['cases'][1]['profile']))
    with pytest.raises(ValueError,match='reference controls'):
        f.validate_request(seal(changed))


@pytest.mark.parametrize('scorer_complete',[True,False])
def test_installed_timing_adapter_requires_complete_scoring_and_retains_its_producer(tmp_path,monkeypatch,scorer_complete):
    from types import SimpleNamespace
    from contextlib import nullcontext
    req=request_fixture(tmp_path)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_qualification.require_hip',lambda:SimpleNamespace(
        cuda=SimpleNamespace(reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:1024)))
    monkeypatch.setattr('oeis_learn.synthesis.client.SynthesisClient',lambda *a:nullcontext(object()))
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_evidence.generation_timing',lambda original:{'explicit_timing_boundary':True})
    def generate(root,name,prompts,profile,**kw):
        assert all(len(p['values'])==20 for p in prompts)
        assert kw['device']=='cuda' and kw['model_control']=='random' and kw['methods']==('correct',)
        assert profile==req['cases'][0]['profile'] and kw['seed']==profile['model']['seed']
        value=seal(dict(records={p['id']:dict(arm_evidence=dict(equal_budget=dict(complete=True))) for p in prompts}))
        path=kw['output_root']/name/'candidates.json';atomic_write(path,value)
        return reference(kw['output_root'],path)
    def score(root,candidate,truth,**kw):
        assert all(len(p['values'])==100 for p in truth)
        assert kw['seconds']>0
        result=seal(dict(outcomes={r['id']:dict(equal_budget=dict(complete=scorer_complete)) for r in truth}))
        atomic_write(root/'scoring/scores.json',result)
        return result
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_evaluation.generate',generate)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_evaluation.score',score)
    if not scorer_complete:
        with pytest.raises(ValueError,match='incomplete portfolio'):f.run_one(tmp_path,0,30)
        assert not (tmp_path/'cases/0/producer.json').exists()
        assert (tmp_path/'cases/0/scoring/scores.json').is_file()
    else:
        result=f.run_one(tmp_path,0,30)
        assert result['scoring']==reference(tmp_path/'cases/0',tmp_path/'cases/0/scoring/scores.json')


def test_roster_cli_is_stdlib_projection_without_device_authority(tmp_path,monkeypatch):
    import sys
    from oeis_learn.pipeline_validation import progressive_research_bindings as bindings
    for name in ('data','design','materials'):atomic_write(tmp_path/(name+'.json'),dict(name=name))
    argv=['research','--roster-only','--root',str(tmp_path),'--output','roster']
    for name in ('data','design','materials'):argv+=['--'+name,name+'.json']
    monkeypatch.setattr(sys,'argv',argv)
    calls=[]
    monkeypatch.setattr(bindings,'publish_roster',lambda *a:calls.append(a))
    monkeypatch.setattr(bindings,'build',lambda *a,**k:pytest.fail('roster cannot construct execution authority'))
    bindings.main()
    assert calls==[(tmp_path,tmp_path/'roster',dict(name='data'),dict(name='design'),dict(name='materials'))]
    monkeypatch.setattr(sys,'argv',argv+['--qualification','foreign'])
    with pytest.raises(SystemExit):bindings.main()


def test_operational_headroom_tolerates_threefold_slowdown_without_shrinking_cases():
    from oeis_learn.pipeline_validation.research_limits import case_allowance, padded_us
    # The old equal-share allowances were 75 s / 33.3 s. A cold but healthy
    # case may take 3x those values without losing its complete-case allowance.
    assert case_allowance(3600)>3*(3600/48)
    assert case_allowance(12*3600)>3*(12*3600/1296)
    assert padded_us(100)>300
    assert 0<case_allowance(10)<10
    assert case_allowance(7200)<=3600
    with pytest.raises(TimeoutError):case_allowance(0)
    with pytest.raises(ValueError):case_allowance(float('inf'))
    with pytest.raises(ValueError):padded_us(True)


def test_readiness_requires_more_than_threefold_measured_time(tmp_path):
    from tests.unit.test_progressive_research_roster import fixture
    from oeis_learn.generalization.progressive.research_roster import construct
    req=request_fixture(tmp_path);roster=construct(*fixture())
    rows=[dict(index=i,case=c['identity'],status='PASS',producer=producer(tmp_path,req,i)) for i,c in enumerate(req['cases'])]
    atomic_write(tmp_path/'evaluation-cases.json',seal(dict(version='progressive-research-evaluation-run/v1',request=req['identity'],status='PASS',cases=rows,actual_elapsed_us=100000)))
    result=f.estimates(tmp_path,req,roster,publish=False)
    by_id={row['id']:row for row in roster['tasks']}
    for task in result['tasks']:
        unpadded=1000*((len(by_id[task['task']]['members'])+1)//2)
        assert task['estimated_us']>3*unpadded


def test_image_build_deadline_and_outer_budget_are_nested(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session
    from oeis_learn.pipeline_validation.research_limits import IMAGE_SECONDS
    from oeis_learn.generalization.progressive.research_budget import SETUP_SECONDS
    s=Session(tmp_path,'run-state/nested/session',{},'prepare-research')
    s.journal['steps']['prepare-research/environment']=dict(status='COMPLETE')
    def launch(argv,output,**kwargs):
        compile(argv[3],'<image builder>','exec')
        assert int(argv[-1]) < kwargs['seconds'] < IMAGE_SECONDS
        assert kwargs['seconds']+kwargs['cleanup_seconds']<=IMAGE_SECONDS
        atomic_write(argv[-3],dict(image='sha256:boundary'))
    monkeypatch.setattr('oeis_learn.pipeline_validation.supervision.launch',launch)
    assert Workflow(s,SimpleNamespace(phase='prepare-research',engine='podman')).images()==dict(qualification='sha256:boundary',gpu='sha256:boundary')
    assert SETUP_SECONDS['images']==2*IMAGE_SECONDS


def test_device_inspection_lease_includes_cleanup(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.generalization_qualification_io import inspect_device
    events=[]
    @contextmanager
    def lock():
        events.append('lock')
        try:yield
        finally:events.append('unlock')
    def launch(*a,**kw):
        events.append('run');assert kw['cleanup_seconds']==120
        atomic_write(tmp_path/'inspect/inspection.json',{'boundary':True})
    def remove(name,timeout):
        assert timeout>100;events.append('remove')
    host=SimpleNamespace(gpu_lock=lock,policy=dict(render_node=None,memory_bytes=1,cpus=1),
        engine=SimpleNamespace(command=lambda **kw:[],remove=remove))
    monkeypatch.setattr('oeis_learn.pipeline_validation.supervision.launch',launch)
    assert inspect_device(host,'image',tmp_path/'inspect')=={'boundary':True}
    assert events==['lock','run','remove','unlock']


@pytest.mark.parametrize('change',['missing-generation','missing-scoring','missing-timing','changed-bytes','foreign-profile','foreign-members','incomplete','wrong-timing'])
def test_timing_closure_rejects_missing_changed_and_foreign_originals(tmp_path,change):
    req=request_fixture(tmp_path);ref=producer(tmp_path,req,0)
    p=load(tmp_path/ref['path']);base=(tmp_path/ref['path']).parent
    f.verify_case(tmp_path,req,req['cases'][0],ref)
    if change.startswith('missing-'):
        key={'generation':'producer','scoring':'scoring','timing':'timing'}[change[8:]]
        (base/p[key]['path']).unlink()
    elif change=='changed-bytes':
        with (base/p['producer']['path']).open('ab') as stream:stream.write(b' ')
    else:
        key='timing' if change=='wrong-timing' else 'producer'
        target=base/p[key]['path'];value=load(target)
        if change=='foreign-profile':value['recipe']=digest('foreign')
        if change=='foreign-members':value['records']['foreign']=value['records'].pop('0')
        if change=='incomplete':value['records']['0']['arm_evidence']['equal_budget']['complete']=False
        if change=='wrong-timing':value['wall_elapsed_us']+=1
        target.write_text(json.dumps(seal(value)))
        p[key]=reference(base,target);(tmp_path/ref['path']).write_text(json.dumps(seal(p)))
        ref=reference(tmp_path,ref['path'])
    with pytest.raises((ValueError,FileNotFoundError)):
        f.verify_case(tmp_path,req,req['cases'][0],ref)
