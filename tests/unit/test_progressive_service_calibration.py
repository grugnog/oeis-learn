"""Read-only data reuse and isolated native measurement; no simulated HIP proof."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oeis_learn.pipeline_validation import progressive_service_calibration as c
from oeis_learn.pipeline_validation.progressive_workstation_state import write,read,Session
from oeis_learn.generalization.progressive.contracts import seal,digest,reference


def fixture(root,native=None):
    output=root/'run-state/services/measure-services/services';output.mkdir(parents=True)
    samples=[dict(id=str(i),source=f'(program (return (add $index (int {i+1}))))',
        indices=list(map(str,range(100))),values=[str(n+i+1) for n in range(100)],training_row=digest(i)) for i in range(4)]
    write(output/'private/training-services.json',samples)
    from tests.unit.test_progressive_research_roster import fixture as roster_fixture
    from oeis_learn.generalization.progressive.research_roster import construct
    tasks=construct(*roster_fixture())['tasks']
    measurements=[]
    for methods,control in [(['correct'],'random'),(list(('correct','shuffled','absent','template','polynomial','linear_recurrence')),'random'),
                            (list(('correct','shuffled','absent','template','polynomial','linear_recurrence')),'retained_D')]:
        measurements.append(dict(case=dict(methods=methods,control=control),generation_us=85_000_000,
                                 coverage=dict(unmeasured_native_reserve_us=40_000_000)))
    inputs=seal(dict(version='progressive-service-inputs/v1',source=digest({}),native=native or digest('TEST NATIVE BOUNDARY'),
        original_request_identity=digest('original timing request'),original_request={'path':'original'},
        samples=reference(output,output/'private/training-services.json'),historical_measurements=measurements,
        prior_timing_seconds=1000,tasks=[dict(r,members=len(r.get('members',[]))) for r in tasks]))
    req=c.plan(root,output,inputs,digest('CPU image boundary'),root/'run-state/services')
    return output,inputs,req


def producer(output,req,i,*,failed=False):
    p=seal(dict(version=c.VERSION,request=req['identity'],case=req['cases'][i],status='FAIL' if failed else 'PASS',
        native=req['native'],measurements=dict(fit_us=1000,raw_us=2000,score_us=3000),
        fit=dict(complete=True,outcome='unsat_in_scope'),scoring=dict(complete=True),actual_elapsed_us=10000))
    write(output/'cases'/str(i)/'producer.json',p)
    return p


@pytest.mark.parametrize('failed',[False,True])
def test_independent_cases_are_durable_and_never_reexecuted(tmp_path,monkeypatch,failed):
    output,inputs,req=fixture(tmp_path)
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.source_snapshot',lambda:{})
    monkeypatch.setenv('OEIS_QUALIFICATION_IMAGE',req['image'])
    calls=[]
    def bounded(argv,**kw):
        i=int(argv[-1]);calls.append(i)
        assert kw['seconds']==120
        producer(output,req,i,failed=failed and i==0)
        return int(failed and i==0)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_acceptance._bounded_process',bounded)
    result=c.run_inside(output,c.SECONDS)
    assert result['status']==('FAIL' if failed else 'PASS') and calls==list(range(8))
    originals={p:p.read_bytes() for p in output.glob('cases/*/producer.json')}
    assert c.run_inside(output,c.SECONDS)==result and calls==list(range(8))
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    report=c.report(output,inputs)
    assert report['status']==('INCOMPLETE' if failed else 'MEASURED')
    assert not report['executable'] and not report['research_ready']
    assert report['original_budget_deficits']
    assert any(not r['fits_24h'] for r in report['trials'])
    runtime=report['runtime_estimate']
    assert runtime['expected_total_seconds'] is None and runtime['expected_training_seconds'] is None
    assert runtime['all_eligible_training_tokens']==364_000_000
    assert runtime['stages']['diversity']['all_eligible_training_tokens']==100_000_000
    assert runtime['stages']['tuning']['all_eligible_training_tokens']==264_000_000
    assert [r['incremental_tokens'] for r in runtime['training_tasks'] if r['task'].startswith('trial-00-')]==[1_000_000,1_000_000,2_000_000]
    for stage,row in runtime['stages'].items():
        assert row['observed_generation_projection_seconds']==sum(
            t['measured_extrapolation_us']/1e6 for t in report['tasks']
            if t['stage']==stage and t['operation']=='generate')
    p=read(output/'cases/1/producer.json');write(output/'cases/1/producer.json',seal(dict(p,actual_elapsed_us=99999)))
    with pytest.raises(ValueError,match='identity'):c.run_inside(output,c.SECONDS)


def test_interrupted_case_does_not_prevent_independent_measurements(tmp_path,monkeypatch):
    output,_,req=fixture(tmp_path)
    write(output/'cases/0/started.json',seal(dict(request=req['identity'],case=req['cases'][0])))
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.source_snapshot',lambda:{})
    monkeypatch.setenv('OEIS_QUALIFICATION_IMAGE',req['image'])
    calls=[]
    def process(argv,**kwargs):
        i=int(argv[-1]);calls.append(i);producer(output,req,i);return 0
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_acceptance._bounded_process',process)
    result=c.run_inside(output,c.SECONDS)
    assert calls==list(range(1,8)) and result['cases'][0]['status']=='FAIL'
    assert read(output/'cases/0/producer.json')['error']=='INTERRUPTED'


def test_partial_measurements_still_publish_runtime_review(tmp_path):
    output,inputs,req=fixture(tmp_path)
    first=c.report(output,inputs)
    assert first['status']=='INCOMPLETE' and first['measurement_deficits']==['service_case_batch_missing']
    assert first['runtime_estimate']['expected_total_seconds'] is None
    assert first['service_maxima_us']['score_us'] is None
    producer(output,req,0)
    write(output/'service-cases.json',seal(dict(version=c.VERSION,request=req['identity'],status='RUNNING',
        cases=[dict(index=0,status='PASS',producer=reference(output,output/'cases/0/producer.json'))])))
    partial=c.report(output,inputs)
    assert partial['status']=='INCOMPLETE'
    assert partial['measurement_deficits']==['service_case:'+str(i) for i in range(1,8)]
    assert partial['service_maxima_us']['score_us']==3000
    assert list((output/'recovery-records').glob('service-budget-review-*.json'))


def test_sibling_path_cannot_reset_allowance_and_request_limits_are_closed(tmp_path):
    output,inputs,req=fixture(tmp_path)
    assert c.plan(tmp_path,output,inputs,req['image'],tmp_path/'run-state/services')==req
    with pytest.raises(ValueError,match='sibling'):
        c.plan(tmp_path,tmp_path/'other',inputs,req['image'],tmp_path/'run-state/sibling')
    for change in (dict(seconds=999999),dict(cases=req['cases'][:-1]),dict(limits=dict(req['limits'],fit_seconds=60))):
        with pytest.raises(ValueError,match='bounded'):c.validate_request(seal(dict(req,**change)))


def test_actual_native_known_training_witness_scores_full_horizon(tmp_path,fitting_client):
    # Mandatory Docker CI supplies the actual service; this is a local skip when absent.
    output,_,req=fixture(tmp_path,native=fitting_client.runtime_identity('bigint-body/v1'))
    result=c.run_one(output,0)
    assert result['status']=='PASS' and result['scoring']['complete']
    assert result['scoring']['production']['coverage']==100
    assert result['scoring']['reference']['coverage']==100
    assert all(v>0 for v in result['measurements'].values())


@pytest.mark.parametrize('incomplete',[False,True])
def test_fit_service_error_keeps_independent_raw_and_scoring_measurements(tmp_path,monkeypatch,incomplete):
    output,_,req=fixture(tmp_path);owners=[];called=[]
    class Client:
        # Only the native service boundary is doubled; production orchestration
        # must close each owner and continue the independent diagnostic paths.
        def __init__(self,*args):owners.append(self);self.closed=False
        def __enter__(self):return self
        def __exit__(self,*args):self.closed=True
        def runtime_identity(self,*args):return req['native']
        def fit(self,*args,**kwargs):
            if incomplete:return dict(complete=False,outcome='incomplete',reason='deadline')
            raise RuntimeError('native fitting unavailable')
    monkeypatch.setattr('oeis_learn.synthesis.client.SynthesisClient',Client)
    def raw(*args):called.append('raw');return {'original':True},{'complete':True}
    def score(*args,**kwargs):
        called.append('score')
        return {'outcomes':{'0':{'equal_budget':dict(complete=True,success=True,scoring={'complete':True})}}}
    monkeypatch.setattr('oeis_learn.fresh.bigint_diagnostics._raw',raw)
    monkeypatch.setattr('oeis_learn.fresh.bigint_diagnostics._seal',lambda *a,**kw:{})
    monkeypatch.setattr('oeis_learn.fresh.bigint_diagnostics.score',score)
    result=c.run_one(output,0)
    assert result['status']=='FAIL' and result['errors'][0]['phase']=='fit'
    assert called==['raw','score'] and len(owners)==3 and all(c.closed for c in owners)
    assert set(result['measurements'])=={'fit_us','raw_us','score_us'}
    assert read(output/'cases/0/producer.json')==result


def test_workstation_has_only_cpu_roles_packages_failures_and_reports(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation.progressive_service_workstation import measure
    output,inputs,req=fixture(tmp_path)
    session=Session(tmp_path,'run-state/services',dict(source=req['source']),'measure-services')
    args=SimpleNamespace(phase='measure-services',preparation_session='run-state/old')
    w=Workflow(session,args);calls=[]
    monkeypatch.setattr(c,'gather',lambda *a:inputs)
    write(output/'service-inputs.json',inputs)
    monkeypatch.setattr(w,'host',lambda:SimpleNamespace(engine=SimpleNamespace(facts=lambda:{}),policy={}))
    def images(targets):
        assert targets==('qualification',)
        session.journal['steps']['measure-services/image-qualification']=dict(status='COMPLETE')
        return {'qualification':req['image']}
    monkeypatch.setattr(w,'images',images)
    def stage(host,root,image,folder,argv,**kwargs):
        calls.append('CPU');assert kwargs['gpu'] is False and image==req['image']
        rows=[]
        for i in range(8):
            producer(output,req,i,failed=i==0)
            rows.append(dict(index=i,status='FAIL' if i==0 else 'PASS',producer=reference(output,output/'cases'/str(i)/'producer.json')))
        write(output/'service-cases.json',seal(dict(version=c.VERSION,request=req['identity'],cases=rows,status='FAIL')))
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_handoff.stage',stage)
    with session.own():measure(w)
    assert calls==['CPU'] and session.code()==2
    assert session.journal['steps']['measure-services/budget-review']['status']=='BLOCKED'
    import zipfile
    with zipfile.ZipFile(session.folder/'share/measure-services.zip') as z:
        assert any(n.endswith('service-budget-review.json') for n in z.namelist())
        assert not any('/private/' in n for n in z.namelist())


def test_stdlib_cli_dry_run_and_phase_options(tmp_path):
    import subprocess,sys
    script=Path(__file__).resolve().parents[2]/'scripts/discovery_hip_handoff.py'
    argv=[sys.executable,'-S',str(script),'--engine','podman','--session','run-state/service',
          '--dry-run','measure-services','--preparation-session','run-state/original']
    result=subprocess.run(argv,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    assert not value['launches'] and value['sequence']==['verify-original-training-inputs','CPU-image','eight-native-service-cases','budget-review']
    assert subprocess.run(argv[:-2],capture_output=True).returncode!=0


def donor_fixture(root,monkeypatch):
    from tests.unit.test_progressive_research_feasibility import request_fixture,cost_fixture
    from oeis_learn.generalization.progressive.generator import default_policy,draw
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    from oeis_learn.pipeline_validation.pilot_clocks import clock_record,DRAIN
    folder=root/'run-state/original';base=folder/'prepare-research'
    rows=[]
    for i in range(24):
        row,_=verify_draw(draw(default_policy(seed=13),i))
        if row:rows.append(seal(dict(row,native_verified=True)))  # explicit native producer boundary
    assert len(rows)>=5
    support=seal(dict(version='progressive-prepared-support/v1',diagnostic=False,shards=[rows],
        layout=dict(probes={'held-out-training-probe':{'id':rows[0]['draw_identity']}})))
    write(base/'materials/private/support.json',support)
    write(base/'materials/materials.json',seal(dict(status='COMPLETE',diagnostic=False,
        support=reference(root,base/'materials/private/support.json'))))
    write(base/'census/census.json',seal(dict(native=digest('TEST NATIVE BOUNDARY'))))
    write(base/'roster/private/role-roster.json',seal(dict(tasks=[])))
    timing=base/'evaluation-timing';req=request_fixture(timing)
    samples=[dict(id=r['draw_identity'],indices=r['indices'],values=r['values']) for r in rows[:2]]
    write(timing/'private/training-samples.json',samples)
    previous={'src/oeis_learn/decoder/example.py':digest('unchanged')}
    write(base/'qualification/source.json',previous)
    q=dict(source_closure=reference(base/'qualification',base/'qualification/source.json'))
    write(base/'qualification/qualification.json',q)
    req=seal(dict(req,source=digest(previous),support=support['identity'],
        scope='training-only prospective throughput; no panel exposure',
        samples=reference(timing,timing/'private/training-samples.json'),
        qualification=reference(root,base/'qualification/qualification.json')))
    write(timing/'request.json',req)
    prompts=sorted([dict(id=r['id'],indices=r['indices'][:20],values=r['values'][:20]) for r in samples],key=lambda r:r['id'])
    for i,case in enumerate(req['cases']):
        g,s=cost_fixture([r['id'] for r in samples]);g['methods']=case['methods']
        if len(case['methods'])>1:
            for key,row in g['records'].items():
                for mode in ('shuffled','absent','template'):
                    row['costs'][mode]=copy.deepcopy(row['costs']['correct'])
                    row['selections'][mode]=None;row['arm_evidence'][mode]=dict(complete=False)
                    s['outcomes'][key][mode]=dict(complete=False,scoring=dict(outcome='no_selected_candidate'))
                    g['clock']['clocks'].append(clock_record(prompt=key,mode=mode,snapshot=digest('model'),
                        start=0,scope_deadline=3600,finish=8,expected=8,started=8,terminal=8,policy=DRAIN))
                for mode in ('polynomial','linear_recurrence'):
                    row['selections'][mode]=None;row['arm_evidence'][mode]=dict(complete=True)
                    s['outcomes'][key][mode]=dict(complete=True,scoring=dict(outcome='no_selected_candidate'))
        g=seal(dict(g,recipe=digest(case['profile']),model_control=case['control'],prompts=digest(prompts)))
        path=timing/'cases'/str(i)/'generation/candidates.json';write(path,g)
        write(timing/'cases'/str(i)/'scoring/scores.json',seal(dict(s,candidates=reference(root,path)['sha256'])))
    step_files={
        'native-census':[base/'census/census.json'],
        'materials':[base/'materials/materials.json',base/'materials/private/support.json'],
        'role-roster':[base/'roster/private/role-roster.json'],
        'qualify':[base/'qualification/qualification.json'],
        'evaluation-plan':[timing/'request.json']}
    steps={'prepare-research/'+k:dict(status='COMPLETE',outputs=[reference(root,p) for p in paths]) for k,paths in step_files.items()}
    steps['prepare-research/evaluation-timing']=dict(status='FAILED',elapsed_seconds=1000)
    write(folder/'journal.json',seal(dict(steps=steps)))
    write(folder/'session.json',seal(dict(binding={'source':req['source']})))
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_handoff.original_authority',lambda *a:(q,base/'qualification'))
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.collect_snapshot',lambda root:previous)
    return folder,support,previous


def test_gather_reuses_original_bytes_and_excludes_training_probes(tmp_path,monkeypatch):
    donor,support,_=donor_fixture(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in donor.rglob('*.json')}
    output=tmp_path/'run-state/new/services'
    value=c.gather(tmp_path,'run-state/original',output)
    assert len(value['historical_measurements'])==14
    assert value['prior_timing_seconds']==1000 and value['additional_seconds']==1320
    samples=json.loads((output/'private/training-services.json').read_text())
    assert len(samples)==4 and len({r['id'] for r in samples})==4
    assert support['layout']['probes']['held-out-training-probe']['id'] not in {r['id'] for r in samples}
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert c.gather(tmp_path,'run-state/original',output)==value


@pytest.mark.parametrize('fault',['changed-support','foreign-generator','after-science','changed-source','symlink'])
def test_gather_refuses_incompatible_or_unbound_originals(tmp_path,monkeypatch,fault):
    donor,support,previous=donor_fixture(tmp_path,monkeypatch)
    if fault=='changed-support':
        with (donor/'prepare-research/materials/private/support.json').open('a') as stream:stream.write(' ')
    if fault=='foreign-generator':
        p=donor/'prepare-research/evaluation-timing/cases/0/generation/candidates.json'
        value=read(p);write(p,seal(dict(value,recipe=digest('foreign'))))
    if fault=='after-science':
        p=donor/'journal.json';v=read(p);v['steps']['run-research/run']={'status':'RUNNING'};write(p,seal(v))
    if fault=='changed-source':
        monkeypatch.setattr('oeis_learn.experiments.source_snapshot.collect_snapshot',lambda root:{k:digest('changed') for k in previous})
    if fault=='symlink':
        p=donor/'prepare-research/evaluation-timing/cases/0/generation/candidates.json'
        target=tmp_path/'aliased.json';p.rename(target);p.symlink_to(target)
    with pytest.raises(ValueError):c.gather(tmp_path,'run-state/original',tmp_path/'run-state/new/services')
