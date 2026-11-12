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
    # Full tuning tasks carry multiple compatible choices. The slowest, not the
    # first enumerated scope, must determine feasibility.
    other=digest('other exact scope')
    expanded=[cases[0],dict(cases[0],compatibility=other)]
    producers[0]['measurements']['update_us']=100
    second=dict(producers[0],compatibility=other,measurements=dict(update_us=1000000,probe_us=100,checkpoint_us=100))
    monkeypatch.setattr('oeis_learn.generalization.progressive.trial_scopes.variants',
        lambda task:[task,dict(task,initializer=dict(task['initializer'],compatibility=other))])
    result=r.finalize(tmp_path,copy.deepcopy(original),expanded,[*producers,second])
    assert all(row['compatibility']==other for row in result['training'])



@pytest.mark.parametrize('failed_timing',[False,True])
@pytest.mark.parametrize('expanded',[False,True])
def test_registered_handoff_uses_canonical_dependencies_and_stops_before_expensive_calibration(tmp_path,monkeypatch,failed_timing,expanded):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_research_launch as launch
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,write
    from tests.unit.test_progressive_research_roster import fixture
    from oeis_learn.generalization.progressive.research_roster import construct
    session=Session(tmp_path,'run-state/research',{},'prepare-research');base=session.folder/'prepare-research'
    for name in ('role-roster','sensitivity'):
        session.journal['steps']['prepare-research/'+name]=dict(status='COMPLETE')
    design,panels,screening=fixture()
    if expanded:
        from oeis_learn.pipeline_validation.progressive_research_design import scientific_round
        design=scientific_round(seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19])))),expanded=True)
    roster=construct(design,panels,screening);atomic_write(base/'roster/private/role-roster.json',roster)
    (tmp_path/'weights.pt').write_bytes(b'original byte identity; model loading boundary is simulated')
    atomic_write(tmp_path/'manifest.json',dict(retained=dict(checkpoints=dict(D=dict(root='.',descriptor=dict(weights=reference(tmp_path,'weights.pt')))))))
    atomic_write(tmp_path/'support.json',{'explicit_native_producer_boundary':True})
    data=dict(manifest=reference(tmp_path,'manifest.json'));materials=dict(support=reference(tmp_path,'support.json'))
    for name,value in [('registered/registered-data.json',data),('materials/materials.json',materials),('research-round-design.json',design)]:
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
            if operation=='prepare':
                assert kwargs['blueprint'].endswith('/construction/blueprint.json')
                assert seconds==design['operational_reservation']['setup_seconds']['prepare']
            return {'explicit_command_boundary':operation},[]
        return session.step(name,action,dependencies=dependencies)
    w=SimpleNamespace(s=session,root=tmp_path,base=base,host=lambda:object(),run_command=command)
    with session.own():launch.prepare(w,data,design,materials)
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
        value,_=cost_fixture([p['id'] for p in prompts],capped=False)
        path=kw['output_root']/name/'candidates.json';atomic_write(path,value)
        return reference(kw['output_root'],path)
    def score(root,candidate,truth,**kw):
        assert all(len(p['values'])==100 for p in truth)
        assert kw['seconds']>0
        _,result=cost_fixture([r['id'] for r in truth],capped=False)
        for row in result['outcomes'].values():
            row['equal_budget']['scoring']['complete']=scorer_complete
        result=seal(result)
        atomic_write(root/'scoring/scores.json',result)
        return result
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_evaluation.generate',generate)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_evaluation.score',score)
    if not scorer_complete:
        with pytest.raises(ValueError,match='scoring incomplete'):f.run_one(tmp_path,0,30)
        assert not (tmp_path/'cases/0/producer.json').exists()
        assert (tmp_path/'cases/0/scoring/scores.json').is_file()
        assert (tmp_path/'cases/0/timing.json').is_file()
        assert load(tmp_path/'cases/0/evaluation-diagnostics.json')['status']=='FAIL'
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


def cost_fixture(members=('0','1'),*,capped=True):
    """Original-format terminal token cap; missing global methods are unselected."""
    from oeis_learn.pipeline_validation.pilot_clocks import clock_record,DRAIN
    records={};outcomes={};clocks=[]
    for member in members:
        details=[]
        for i in range(8):
            limited=capped and i==0
            details.append(dict(position=i,tokens=[1]*(1024 if limited else 3),complete=not limited,
                generation_seconds=1,generation=dict(complete=not limited,
                    reason='token_cap_without_eos' if limited else None),
                fitting=dict(complete=not limited,outcome='not_attempted' if limited else 'unsat_in_scope')))
        clocks.append(clock_record(prompt=member,mode='correct',snapshot=digest('model'),
            start=0,scope_deadline=3600,finish=8,expected=8,started=8,terminal=8,policy=DRAIN))
        records[member]=dict(costs={'correct':dict(details=details)},
            arm_evidence={m:dict(complete=not capped,reason='token_cap_without_eos' if capped else 'unsat_in_scope')
                          for m in ('raw','rescue','equal_budget')},
            selections={m:None if capped else {'explicit_scoring_boundary':True} for m in ('raw','rescue','equal_budget')})
        outcomes[member]={m:dict(complete=not capped,success=False,outcome='unresolved' if capped else 'failure',
            scoring=dict(complete=not capped,outcome='no_selected_candidate' if capped else 'continuation_mismatch'))
            for m in ('raw','rescue','equal_budget')}
    generated=seal(dict(version='discovery-candidates/v1',methods=['correct'],records=records,
        elapsed_us=8_000_000,clock=dict(policy=DRAIN,expected_portfolios=8,missing_portfolios=6,cost_complete=False,clocks=clocks)))
    scored=seal(dict(outcomes=outcomes))
    return generated,scored


def test_cost_completion_accepts_terminal_caps_without_relabelling_science():
    from oeis_learn.pipeline_validation.progressive_timing_coverage import coverage
    g,s=cost_fixture();before=copy.deepcopy((g,s))
    result=coverage(g,s)
    assert (g,s)==before
    assert result['selected_portfolios']==2 and result['terminal_attempts']==16
    assert result['token_capped_attempts']==2 and result['unresolved_scientific_outcomes']==6
    assert result['unmeasured_native_reserve_us']==16_000_000
    assert result['unmeasured_scoring_reserve_us']==360_000_000


@pytest.mark.parametrize('change',['missing-clock','duplicate-clock','unstarted','missing-detail',
    'wrong-cap','device-error','native-error','scoring-error'])
def test_cost_completion_rejects_unfinished_or_failed_services(change):
    from oeis_learn.pipeline_validation.progressive_timing_coverage import coverage
    g,s=cost_fixture(capped=change!='scoring-error')
    first=g['records']['0']['costs']['correct']['details'][0]
    if change=='missing-clock':g['clock']['clocks'].pop()
    if change=='duplicate-clock':g['clock']['clocks'].append(g['clock']['clocks'][0])
    if change=='unstarted':g['clock']['clocks'][0]['terminal']=7
    if change=='missing-detail':g['records']['0']['costs']['correct']['details'].pop()
    if change=='wrong-cap':first['tokens'].pop()
    if change=='device-error':first['generation']['reason']='device lost'
    if change=='native-error':g['records']['0']['costs']['correct']['details'][1]['fitting']['complete']=False
    if change=='scoring-error':s['outcomes']['0']['equal_budget']['scoring']['complete']=False
    with pytest.raises(ValueError):coverage(g,s)


def test_realistic_hash_member_roster_exceeds_old_limit_and_loads_through_contract(tmp_path,monkeypatch):
    from tests.unit.test_progressive_research_roster import fixture
    from oeis_learn.generalization.progressive.research_roster import construct,ROSTER_DOCUMENT_BYTES
    design,panels,screening=fixture()
    for rows in panels.values():
        for r in rows:r['id']=digest(r['id']);r['group']=digest(r['group'])
    for r in screening:r['id']=digest(r['id']);r['group']=digest(r['group'])
    roster=construct(design,panels,screening)
    atomic_write(tmp_path/'roster.json',roster)
    assert 4<<20 < (tmp_path/'roster.json').stat().st_size < ROSTER_DOCUMENT_BYTES
    atomic_write(tmp_path/'request.json',{'explicit_case_boundary':True})
    report={'status':'READY'};atomic_write(tmp_path/'report.json',report)
    def estimates(root,req,actual,**kw):
        assert actual==roster
        return report
    monkeypatch.setattr(f,'estimates',estimates)
    f.execution_contract(tmp_path,tmp_path/'contract.json',request_path=tmp_path/'request.json',
        roster_path=tmp_path/'roster.json',report_path=tmp_path/'report.json')


def test_v2_timing_proof_replays_caps_and_accounts_for_unmeasured_work(tmp_path):
    from oeis_learn.pipeline_validation.progressive_timing_coverage import coverage
    from oeis_learn.pipeline_validation.progressive_evidence import generation_timing
    from tests.unit.test_progressive_research_roster import fixture
    from oeis_learn.generalization.progressive.research_roster import construct
    req=request_fixture(tmp_path);rows=[]
    for i,case in enumerate(req['cases']):
        ref=producer(tmp_path,req,i);p=load(tmp_path/ref['path']);base=(tmp_path/ref['path']).parent
        if i==0:
            g=load(base/p['producer']['path']);template,scored=cost_fixture()
            g=seal(dict(g,records=template['records'],clock=template['clock']))
            (base/p['producer']['path']).write_text(json.dumps(g))
            p['producer']=reference(base,base/p['producer']['path'])
            s=load(base/p['scoring']['path'])
            s=seal(dict(s,outcomes=scored['outcomes'],candidates=p['producer']['sha256']))
            (base/p['scoring']['path']).write_text(json.dumps(s))
            p['scoring']=reference(base,base/p['scoring']['path'])
            (base/p['timing']['path']).write_text(json.dumps(generation_timing(g)))
            p['timing']=reference(base,base/p['timing']['path'])
            p=seal(dict(p,version='progressive-research-evaluation-case/v2',coverage=coverage(g,s)))
            (tmp_path/ref['path']).write_text(json.dumps(p));ref=reference(tmp_path,ref['path'])
            assert f.verify_case(tmp_path,req,case,ref)==p
        rows.append(dict(index=i,case=case['identity'],status='PASS',producer=ref))
    atomic_write(tmp_path/'evaluation-cases.json',seal(dict(version='progressive-research-evaluation-run/v1',
        request=req['identity'],status='PASS',cases=rows,actual_elapsed_us=100000)))
    report=f.estimates(tmp_path,req,construct(*fixture()))
    assert report['status']=='BLOCKED'  # fast empty scoring cannot establish feasibility
    trial=next(t for t in report['tasks'] if t['task']=='trial-00-1000000-development-generate')
    assert trial['estimated_us']>=16_000_000*64*4
    path=tmp_path/rows[0]['producer']['path'];p=load(path)
    p['coverage']['unmeasured_native_reserve_us']=0
    path.write_text(json.dumps(seal(p)))
    with pytest.raises(ValueError,match='coverage differs'):
        f.verify_case(tmp_path,req,req['cases'][0],reference(tmp_path,path))


def test_failure_bundle_includes_timing_coverage_but_not_private_inputs(tmp_path):
    import zipfile
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session
    s=Session(tmp_path,'run-state/timing',{},'prepare-research')
    base=s.folder/'prepare-research/evaluation-timing/cases/0'
    atomic_write(base/'evaluation-diagnostics.json',seal(dict(status='FAIL',error='timing native fitting incomplete')))
    atomic_write(base/'timing.json',seal(dict(version='progressive-phase-timing/v1')))
    atomic_write(base/'generation/attempts/attempt-index.json',seal(dict(states={'INCOMPLETE':1})))
    atomic_write(base/'generation/attempts/portfolio-clocks.json',seal(dict(cost_complete=False)))
    (base/'generation/attempts/attempts.jsonl').write_text('{"event":"INCOMPLETE"}\n')
    atomic_write(base/'private/training-samples.json',{'secret':True})
    with zipfile.ZipFile(s.package()) as z:
        assert any(n.endswith('evaluation-diagnostics.json') for n in z.namelist())
        assert any(n.endswith('timing.json') for n in z.namelist())
        assert any(n.endswith('attempt-index.json') for n in z.namelist())
        assert any(n.endswith('portfolio-clocks.json') for n in z.namelist())
        assert any(n.endswith('attempts.jsonl') for n in z.namelist())
        assert not any('/private/' in n for n in z.namelist())
