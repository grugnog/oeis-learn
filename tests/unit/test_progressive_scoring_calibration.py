"""Positive training witnesses fill unmeasured scoring paths, never science."""
import pytest
from oeis_learn.generalization.progressive.contracts import load,seal,digest,reference
from oeis_learn.pipeline_validation.progressive_workstation_state import write as atomic_write
from oeis_learn.pipeline_validation import progressive_scoring_calibration as scoring
from oeis_learn.pipeline_validation import progressive_research_feasibility as feasibility


def fixture(root):
    from tests.unit.test_progressive_research_feasibility import request_fixture
    from tests.unit.test_progressive_service_calibration import fixture as service_fixture
    service_root,_,service_req=service_fixture(root)
    service_root=service_root.rename(root/'scoring-services')
    req=request_fixture(root)
    # Explicit native process boundary. All consumers verify its original refs.
    service_req=seal(dict(service_req,source=req['source'],image=req['image'],
        inputs=digest(dict(support=req['support'],scope='positive training service timing'))))
    atomic_write(service_root/'request.json',service_req)
    req=seal(dict(req,version='progressive-research-evaluation-request/v2',
        scoring_services=reference(root,service_root/'request.json')))
    atomic_write(root/'request.json',req)
    samples=load(service_root/service_req['samples']['path'])
    rows=[]
    for i,case in enumerate(service_req['cases']):
        value=seal(dict(version='progressive-service-calibration/v1',request=service_req['identity'],
            case=case,status='PASS',native=service_req['native'],training_row=samples[case['sample']]['training_row'],
            measurements=dict(fit_us=1000,raw_us=2000,score_us=3000),fit=dict(complete=True),errors=[],
            scoring=dict(complete=True,outcome='success',production=dict(outcome='ok',coverage=100),reference=dict(outcome='ok',coverage=100)),
            actual_elapsed_us=10000))
        path=service_root/'cases'/str(i)/'producer.json';atomic_write(path,value)
        rows.append(dict(index=i,status='PASS',producer=reference(service_root,path)))
    atomic_write(service_root/'service-cases.json',seal(dict(version=service_req['version'],request=service_req['identity'],
        cases=rows,status='PASS')))
    return req,service_root


def test_scoring_uses_measured_positive_witnesses_and_the_generating_method_scope(tmp_path):
    from tests.unit.test_progressive_research_feasibility import producer
    from tests.unit.test_progressive_research_roster import fixture as roster_fixture
    from oeis_learn.generalization.progressive.research_roster import construct
    req,service_root=fixture(tmp_path)
    rows=[dict(index=i,case=c['identity'],status='PASS',producer=producer(tmp_path,req,i)) for i,c in enumerate(req['cases'])]
    atomic_write(tmp_path/'evaluation-cases.json',seal(dict(version='progressive-research-evaluation-run/v1',request=req['identity'],
        status='PASS',cases=rows,actual_elapsed_us=100000)))
    roster=construct(*roster_fixture());report=feasibility.estimates(tmp_path,req,roster)
    by_id={r['id']:r for r in roster['tasks']}
    assert report['status']=='READY' and report['positive_scoring']['score_us']==3000
    for row in report['tasks']:
        task=by_id[row['task']]
        if task['operation']=='score':
            arms=8 if by_id[task['generation']]['all_methods'] else 3
            assert row['estimated_us']==4*(4000+len(task['members'])*arms*3000)
    # Positive paths must all exist: empty model outputs never authorize skipping them.
    (service_root/'cases/7/producer.json').unlink()
    with pytest.raises((ValueError,OSError)):feasibility.estimates(tmp_path,req,roster)


@pytest.mark.parametrize('change',['missing','training','native','incomplete','reference','mismatch','time','source','tamper'])
def test_positive_service_closure_is_checked_before_planning(tmp_path,change):
    req,base=fixture(tmp_path)
    if change=='source':
        req=seal(dict(req,source=digest('different')))
    elif change=='missing':(base/'cases/0/producer.json').unlink()
    else:
        path=base/'cases/0/producer.json';p=load(path)
        if change=='training':p['training_row']=digest('another row')
        if change=='native':p['native']=digest('another runtime')
        if change=='incomplete':p['scoring']['complete']=False
        if change=='reference':p['scoring']['reference']['coverage']=20
        if change=='mismatch':p['scoring']['outcome']='continuation_mismatch'
        if change=='time':p['measurements']['score_us']=10**12
        if change=='tamper':p['actual_elapsed_us']+=1
        atomic_write(path,seal(p))
        if change!='tamper':
            result=load(base/'service-cases.json');result['cases'][0]['producer']=reference(base,path)
            atomic_write(base/'service-cases.json',seal(result))
    with pytest.raises((ValueError,OSError)):scoring.measurements(tmp_path,req)


@pytest.mark.parametrize('service_ok',[False,True])
def test_positive_failure_still_runs_independent_model_timing_cases(tmp_path,monkeypatch,service_ok):
    from tests.unit.test_progressive_research_feasibility import producer
    req,base=fixture(tmp_path)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_qualification.require_hip',lambda:None)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_qualification.hardware_facts',lambda:{})
    monkeypatch.setattr('oeis_learn.pipeline_validation.generalization_execution.verify_execution',lambda:req['execution_policy'])
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.source_snapshot',lambda:{})
    monkeypatch.setenv('OEIS_QUALIFICATION_IMAGE',req['image']);monkeypatch.setenv('OEIS_QUALIFICATION_ENGINE','podman')
    def failed(*args):
        if not service_ok:raise ValueError('positive scoring failed')
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_service_calibration.run_inside',failed)
    calls=[]
    def process(argv,**kwargs):
        i=int(argv[4]);calls.append(i);producer(tmp_path,req,i);return 0
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_acceptance._bounded_process',process)
    if service_ok:feasibility.run_inside(tmp_path)
    else:
        with pytest.raises(ValueError,match='incomplete'):feasibility.run_inside(tmp_path)
    assert calls==list(range(14))
    result=load(tmp_path/'evaluation-cases.json')
    assert result['status']==('PASS' if service_ok else 'FAIL') and all(r['status']=='PASS' for r in result['cases'])
    if not service_ok:assert load(tmp_path/'failures.json')[0]['case']=='positive-scoring-services'


def test_original_training_witness_selection_excludes_probes_and_is_deterministic(tmp_path,monkeypatch):
    from tests.unit.test_progressive_service_calibration import donor_fixture
    from oeis_learn.pipeline_validation.progressive_service_calibration import training_samples
    folder=donor_fixture(tmp_path,monkeypatch)[0]
    support=load(folder/'prepare-research/materials/private/support.json')
    chosen=training_samples(support)
    probes={p['id'] for p in support['layout']['probes'].values()}
    assert len(chosen)==4 and not probes&{r['id'] for r in chosen}
    reversed_support=seal(dict(support,shards=[list(reversed(support['shards'][0]))]))
    assert training_samples(reversed_support)==chosen
    scoring.prepare(tmp_path/'positive',support,source=digest('source'),image=digest('image'),native=digest('native'))
    assert load(tmp_path/'positive/private/training-services.json')==chosen


def test_execution_contract_replays_positive_scoring_evidence(tmp_path):
    from tests.unit.test_progressive_research_feasibility import producer
    from tests.unit.test_progressive_research_roster import fixture as roster_fixture
    from oeis_learn.generalization.progressive.research_roster import construct
    req,base=fixture(tmp_path)
    atomic_write(tmp_path/'source.json',{})
    atomic_write(tmp_path/'qualification.json',dict(source_closure=reference(tmp_path,'source.json')))
    req=seal(dict(req,qualification=reference(tmp_path,'qualification.json')))
    atomic_write(tmp_path/'request.json',req)
    rows=[dict(index=i,case=c['identity'],status='PASS',producer=producer(tmp_path,req,i)) for i,c in enumerate(req['cases'])]
    atomic_write(tmp_path/'evaluation-cases.json',seal(dict(version='progressive-research-evaluation-run/v1',
        request=req['identity'],status='PASS',cases=rows,actual_elapsed_us=100000)))
    roster=construct(*roster_fixture());atomic_write(tmp_path/'roster.json',roster)
    feasibility.estimates(tmp_path,req,roster)
    value=feasibility.execution_contract(tmp_path,tmp_path/'contract.json',request_path=tmp_path/'request.json',
        roster_path=tmp_path/'roster.json',report_path=tmp_path/'evaluation-feasibility.json')
    runtime=dict(payload=dict(qualification=[req['qualification']],scope=dict(images=['cpu',req['image']],execution_policy=req['execution_policy'])))
    assert feasibility.verify_contract(tmp_path,reference(tmp_path,'contract.json'),runtime)==value
    (base/'cases/7/producer.json').unlink()
    with pytest.raises((OSError,ValueError)):feasibility.verify_contract(tmp_path,reference(tmp_path,'contract.json'),runtime)


def test_actual_native_service_batch_satisfies_preparation_measurement_contract(tmp_path,fitting_client):
    from oeis_learn.pipeline_validation import progressive_service_calibration as services
    req,base=fixture(tmp_path)
    service_req=seal(dict(load(base/'request.json'),native=fitting_client.runtime_identity('bigint-body/v1')))
    atomic_write(base/'request.json',service_req)
    req=seal(dict(req,scoring_services=reference(tmp_path,base/'request.json')))
    rows=[]
    for case in service_req['cases']:
        path=base/'cases'/str(case['index'])/'producer.json';path.unlink()
        services.run_one(base,case['index'])
        rows.append(dict(index=case['index'],status='PASS',producer=reference(base,path)))
    atomic_write(base/'service-cases.json',seal(dict(version=services.VERSION,request=service_req['identity'],cases=rows,status='PASS')))
    result=scoring.measurements(tmp_path,req)
    assert result['score_us']>0 and result['owner_overhead_us']>=0
