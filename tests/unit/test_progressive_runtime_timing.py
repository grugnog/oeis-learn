"""Actual learner timing plus original-owner, failure and projection boundaries."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

from oeis_learn.pipeline_validation import progressive_runtime_timing as t
from oeis_learn.pipeline_validation.progressive_workstation_state import read, write, Session
from oeis_learn.generalization.progressive.contracts import seal, digest, reference, load
from tests.unit.test_progressive_training_timing import package_fixture


def request_fixture(root, package):
    output=root/'run-state/runtime/measure-runtime/timing'
    write(output/'private/training-input.json',package)
    review=dict(runtime_estimate=dict(training_tasks=[
        dict(task='first',stage='capability',domain='baseline',incremental_tokens=1000000),
        dict(task='continuation',stage='tuning',domain='all',incremental_tokens=3000000)],
        stages={s:dict(observed_generation_projection_seconds=100,
                       positive_witness_scoring_projection_seconds=20) for s in ('capability','tuning')}))
    write(output/'service-budget-review.json',seal(review))
    inputs=seal(dict(source=digest({}),original_inputs=digest('original'),
        package=reference(output,'private/training-input.json'),
        service_review=reference(output,'service-budget-review.json')))
    write(output/'runtime-inputs.json',inputs)
    req=t.plan(root,output,inputs,digest('image'),root/'run-state/runtime')
    return output,inputs,req


@pytest.mark.parametrize('index',[0,6])
def test_actual_installed_learner_two_tasks_timing_and_restore(tmp_path,monkeypatch,index):
    monkeypatch.setenv('OEIS_PROGRESSIVE_TIMING','1')
    monkeypatch.delenv('OEIS_PROGRESSIVE_ACCEPTANCE_DEVICE',raising=False)
    monkeypatch.setattr(t,'TARGET',16000)
    package=package_fixture(tmp_path)
    package=seal(dict(package,shards=package['shards']*8))
    output,inputs,req=request_fixture(tmp_path,package)
    value=t.run_case(output,index,device='cpu')
    assert value['status']=='PASS' and value['device']=='cpu'
    a,b=value['segments']
    assert a['result']['metrics']['initialization']=='fresh_random'
    assert b['result']['metrics']['initialization']=='resume'
    for segment in (a,b):
        timing=segment['timing']
        assert timing['tokens']>0 and timing['updates']>8
        assert timing['wall_us']>=timing['loop_us']>0
        assert timing['components']['optimizer_update']['calls']==timing['updates']
        assert timing['components']['checkpoint_save']['calls']<timing['updates']
        assert timing['components']['checkpoint_restore']['calls']>0
        assert timing['components']['probes']['calls']>0
        # Nested categories are exclusive, never double-counted into totals.
        assert sum(c['self_us'] for c in timing['components'].values()) <= timing['wall_us']
    if index==6:assert max(s['timing']['maximum_shard'] for s in (a,b))>0
    assert t.report(output,inputs)['status']=='INCOMPLETE' # CPU cannot become HIP.
    with pytest.raises(ValueError,match='publication differs'):
        t.run_case(output,index,device='cpu')


def test_case_packages_keep_original_rows_probes_and_do_not_rewrite_science(tmp_path):
    package=package_fixture(tmp_path);original=copy.deepcopy(package)
    for case in t.scopes():
        derived=t.case_package(package,case)
        assert derived['shards']==original['shards'] and derived['layout']==original['layout']
        assert derived['checkpoint_policy']=='hourly/v1' and derived['scope']=='DIAGNOSTIC_ONLY'
        assert derived['update_policy']['effective_batch']==case['batch']
        assert derived['profile']['shape']==case['shape']
    assert package==original


def test_owner_and_closed_request(tmp_path):
    output,inputs,req=request_fixture(tmp_path,package_fixture(tmp_path))
    assert t.plan(tmp_path,output,inputs,req['image'],tmp_path/'run-state/runtime')==req
    with pytest.raises(ValueError,match='sibling'):
        t.plan(tmp_path,output,inputs,req['image'],tmp_path/'run-state/another')
    for change in ({'target_tokens':999999},{'scope':'research'},{'cases':req['cases'][:-1]}):
        with pytest.raises(ValueError,match='closed'):t.validate_request(seal(dict(req,**change)))


def fake_producer(req,case):
    if case['kind']=='calibration':
        return seal(dict(request=req['identity'],case=case,status='PASS',device='actual_HIP',
                         elapsed_seconds=10,calibration={'measurements':{'update_us':1000000}}))
    timing=dict(windows=[dict(tokens=10000,elapsed_us=5000000,cold=False)],loop_us=5000000,
        components={'checkpoint_save':dict(calls=2,inclusive_us=2000000,self_us=2000000)},maximum_shard=1)
    return seal(dict(request=req['identity'],case=case,status='PASS',device='actual_HIP',
                     segments=[dict(timing=timing,process={'elapsed_us':7000000})]*2))


def test_unpadded_runtime_arithmetic_and_missing_results(tmp_path):
    output,inputs,req=request_fixture(tmp_path,package_fixture(tmp_path));rows=[]
    for i,case in enumerate(req['cases']):
        path=output/'cases'/str(i)/'producer.json';write(path,fake_producer(req,case))
        rows.append(dict(index=i,status='PASS',producer=reference(output,path)))
    write(output/'runtime-cases.json',seal(dict(request=req['identity'],cases=rows,device_block=None)))
    value=t.report(output,inputs)
    assert value['status']=='MEASURED'
    estimate=value['runtime_estimate']
    assert estimate['all_eligible_incremental_tokens']==4000000
    # 2000s loops + 2s overhead for each of two tasks + 240s evaluation.
    assert estimate['selected_round_low_seconds']==2244
    assert estimate['selected_round_high_seconds']==2244
    assert estimate['calibration_1296_case_sample_projection_seconds']==[12960,12960]
    assert estimate['full_campaign_expected_seconds'] is None
    assert not value['scientific_budgets_changed'] and not value['assessment_exposed']
    rows.pop()
    write(output/'runtime-cases.json',seal(dict(request=req['identity'],cases=rows,device_block=None)))
    assert t.report(output,inputs)['status']=='INCOMPLETE'


def test_installed_case_batch_continues_failure_but_blocks_unsafe_cleanup(tmp_path,monkeypatch):
    output,inputs,req=request_fixture(tmp_path,package_fixture(tmp_path))
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.source_snapshot',lambda:{})
    monkeypatch.setenv('OEIS_QUALIFICATION_IMAGE',req['image'])
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_qualification.hardware_facts',lambda:{})
    calls=[]
    def launch(argv,folder,**kwargs):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        if folder.name=='health':return {}
        index=int(argv[-1]);calls.append(index)
        if index in (0,2):
            write(folder/'process.json',dict(returncode=1,cleanup_failure='still alive' if index==2 else None))
            raise RuntimeError('injected case failure')
        write(folder.parent/'producer.json',fake_producer(req,req['cases'][index]))
        return {}
    monkeypatch.setattr('oeis_learn.pipeline_validation.supervision.launch',launch)
    result=t.run_inside(output,t.HOST_SECONDS)
    assert calls==[0,1,2]
    assert result['status']=='FAIL' and result['device_block']=='UNPROVEN_CASE_CLEANUP'
    assert [c['status'] for c in result['cases']][:4]==['FAIL','PASS','FAIL','BLOCKED']
    assert t.report(output,inputs)['status']=='INCOMPLETE'


def test_workstation_packages_failure_no_restart(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation.progressive_runtime_workstation import measure
    output,inputs,req=request_fixture(tmp_path,package_fixture(tmp_path))
    for name in ('service-inputs.json','prior-training.json'):write(output/name,{})
    session=Session(tmp_path,'run-state/runtime',dict(source=req['source']),'measure-runtime')
    w=Workflow(session,SimpleNamespace(phase='measure-runtime',service_session='run-state/services',training_session='run-state/train'))
    monkeypatch.setattr(t,'gather',lambda *args:inputs)
    monkeypatch.setattr(w,'host',lambda:SimpleNamespace(engine=SimpleNamespace(facts=lambda:{}),policy={}))
    def images(targets):
        session.journal['steps']['measure-runtime/image-gpu']=dict(status='COMPLETE')
        return {'gpu':req['image']}
    monkeypatch.setattr(w,'images',images);calls=[]
    def stage(*args,**kwargs):
        assert kwargs['gpu'];calls.append('stage');raise RuntimeError('injected container failure')
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_handoff.stage',stage)
    with session.own():measure(w)
    with session.own():measure(w)
    assert calls==['stage']
    import zipfile
    with zipfile.ZipFile(session.folder/'share/measure-runtime.zip') as z:
        assert any(n.endswith('runtime-review.json') for n in z.namelist())
        assert not any('/private/' in n for n in z.namelist())


def test_stdlib_cli_requires_both_originals():
    script=Path(__file__).resolve().parents[2]/'scripts/discovery_hip_handoff.py'
    args=[sys.executable,'-S',str(script),'--engine','podman','--session','run-state/runtime','--dry-run',
          'measure-runtime','--service-session','run-state/service','--training-session','run-state/training']
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'seven-installed-learner-cases' in json.loads(result.stdout)['sequence']
    assert subprocess.run(args[:-2],capture_output=True).returncode!=0


def test_gather_original_training_and_service_closure(tmp_path,monkeypatch):
    from tests.unit.test_progressive_training_timing import donor_fixture
    from oeis_learn.pipeline_validation import progressive_training_timing as old
    service,support=donor_fixture(tmp_path)
    service_path=service/'measure-services/services/service-inputs.json'
    original=read(service_path)
    qroot=tmp_path/'run-state/original/qualification'
    write(qroot/'source.json',{})
    q={'source_closure':reference(qroot,'source.json')}
    write(qroot/'qualification.json',q)
    request=seal(dict(source=digest({}),qualification=reference(tmp_path,qroot/'qualification.json')))
    write(qroot/'request.json',request)
    original=seal(dict(original,original_request=reference(tmp_path,qroot/'request.json')))
    write(service_path,original)
    sj=read(service/'journal.json')
    sj['steps']['measure-services/original-inputs']['outputs']=[reference(tmp_path,service_path)]
    sp=service/'measure-services/services/request.json';sv=seal(dict(inputs=original['identity']));write(sp,sv)
    sj['steps']['measure-services/service-plan']['outputs']=[reference(tmp_path,sp)]
    rp=service/'measure-services/services/service-budget-review.json';write(rp,seal(dict(service_request=sv['identity'])))
    sj['steps']['measure-services/budget-review']['outputs']=[reference(tmp_path,rp)]
    write(service/'journal.json',seal(sj))
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.collect_snapshot',lambda root:{})
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_handoff.original_authority',lambda *a:(q,qroot))
    folder=tmp_path/'run-state/training';base=folder/'measure-training/timing'
    inputs=old.gather(tmp_path,'run-state/services',base)
    req=old.plan(tmp_path,base,inputs,digest('old image'),folder)
    p=load(base/'private/training-input.json',maximum=256<<20)
    # Physical producer is the sole hardware double; all original references,
    # journals, source checks and package constructors are real.
    write(base/'measurement/producer.json',seal(dict(status='PASS',device='actual_HIP',package=p['identity'])))
    write(base/'training-runtime-review.json',seal(dict(status='MEASURED')))
    files={'original-inputs':['training-timing-inputs.json','private/training-input.json'],
           'timing-plan':['request.json'],'training-measurements':['measurement/producer.json'],
           'runtime-review':['training-runtime-review.json']}
    write(folder/'journal.json',seal(dict(steps={'measure-training/'+k:dict(status='COMPLETE',
        outputs=[reference(tmp_path,base/n) for n in names]) for k,names in files.items()})))
    originals={p:p.read_bytes() for p in (tmp_path/'run-state').rglob('*.json')}
    output=tmp_path/'run-state/runtime/measure-runtime/timing'
    value=t.gather(tmp_path,'run-state/services','run-state/training',output)
    assert value['original_inputs']==original['identity']
    assert t.gather(tmp_path,'run-state/services','run-state/training',output)==value
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    with (base/'measurement/producer.json').open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError):t.gather(tmp_path,'run-state/services','run-state/training',output)


def test_observer_preserves_real_dropout_weights_and_rng(tmp_path,monkeypatch):
    import torch
    from tests.unit.test_progressive_training_policy import profile, policy, batch
    from oeis_learn.pipeline_validation.progressive_training import PolicyUpdate
    from oeis_learn.pipeline_validation.learner_timing import recording
    from oeis_learn.generalization.budgets import Ledger
    p=profile(.2);cfg=policy(p);x,tokens,pad=batch()
    charge=int(tokens[:,1:].ne(pad).sum());outcomes=[]
    for enabled in ('0','1'):
        monkeypatch.setenv('OEIS_PROGRESSIVE_TIMING',enabled)
        runner=PolicyUpdate(p,cfg,device='cpu');ledger=Ledger(charge*2)
        with recording(tmp_path/enabled):
            for i in range(2):
                _,ledger=runner.update(x,tokens,ledger=ledger.reserve(str(i),charge))
        outcomes.append(([v.detach().clone() for v in [*runner.encoder.parameters(),*runner.decoder.parameters()]],torch.get_rng_state()))
    assert all(torch.equal(a,b) for a,b in zip(outcomes[0][0],outcomes[1][0]))
    assert torch.equal(outcomes[0][1],outcomes[1][1])
    assert not (tmp_path/'0/timing.json').exists()
    assert read(tmp_path/'1/timing.json')['components']['optimizer_update']['calls']==2
