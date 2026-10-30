import pytest
from oeis_learn.generalization.recovery import Workflow, restart
from oeis_learn.generalization.resources import preflight, log_tail, reserve_stage
from oeis_learn.generalization.bundles import bundle_plan, verify_bundle

def test_immutable_terminal_and_restart():
    flow=Workflow('study',('qualify','prepare'))
    flow=flow.start('qualify').finish('qualify','PASS')
    with pytest.raises(ValueError): flow.start('qualify')
    assert flow.status=='PARTIAL'
    flow=flow.start('prepare').finish('prepare','INTERRUPTED')
    assert flow.status=='INTERRUPTED'
    assert restart(flow,old_root='old',new_root='new',reason='interrupted',learner=False)['reason']=='interrupted'
    with pytest.raises(ValueError): restart(flow,old_root='old',new_root='old',reason='interrupted',learner=False)
    with pytest.raises(ValueError): restart(flow,old_root='old',new_root='new',reason='interrupted',learner=True)

def test_reservations_and_log_bounds():
    assert preflight(study_bytes=10,checkpoint_bytes=5,shard_bytes=1,raw_bytes=1,sharing_bytes=1,available_bytes=100)['minutes']==240
    with pytest.raises(ValueError): preflight(study_bytes=100,checkpoint_bytes=5,shard_bytes=1,raw_bytes=1,sharing_bytes=1,available_bytes=90)
    with pytest.raises(ValueError): reserve_stage('cases',elapsed_seconds=200*60,requested_seconds=75*60)
    tail=log_tail(('x'*20000+'\n')*200)
    assert len(tail.encode())<=1024**2 and len(tail.splitlines())<=100

def test_required_results_cannot_be_silently_redacted():
    with pytest.raises(ValueError): bundle_plan([dict(path='weights',bytes=1,private=True,required=True)],mandatory={'weights'})
    with pytest.raises(ValueError): bundle_plan([dict(path='huge',bytes=513*1024**2,private=False,required=True)],mandatory={'huge'})
    assert bundle_plan([dict(path='private',bytes=1,private=True,required=False)],mandatory=set())['omissions']==['private']
    with pytest.raises(ValueError): verify_bundle({'study':'s','stages':[]},expected_study='other',root='.')

@pytest.mark.parametrize("stale_family", [False, True])
def test_offline_complete_bundle_and_stale_summary(tmp_path, stale_family):
    import copy,hashlib,json
    from pathlib import Path
    from oeis_learn.generalization.contracts import freeze
    from oeis_learn.pipeline_validation.identity import seal
    from oeis_learn.generalization.qualification import FACTS
    fixtures=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())
    def write(name,value):
        raw=json.dumps(value,sort_keys=True).encode();(tmp_path/name).write_bytes(raw)
        return dict(path=name,bytes=len(raw),sha256='sha256:'+hashlib.sha256(raw).hexdigest())
    base=write('base.json',{})
    def replace_refs(value):
        if isinstance(value,dict):
            if set(value)=={'path','bytes','sha256'}: return dict(base)
            return {k:replace_refs(v) for k,v in value.items()}
        if isinstance(value,list): return [replace_refs(v) for v in value]
        return value
    study=freeze('study',replace_refs(fixtures['study'])).record()
    study_ref=write('study.json',study)
    device=write('device.json',seal(dict(version='generalization-device-evidence/v1',device='hip',loaded_weights='fixture',finite_forward=True,finite_backward=True,parameters_changed=True)))
    q=replace_refs(fixtures['qualification']);facts={k:q[k] for k in FACTS}
    producer=seal(dict(version='generalization-producer/v1',status='PASS',facts=facts,suites=q['suite_selection'],capabilities=q['covered_capabilities'],artifacts=[device],dependencies=[],feature_acceptance=False))
    producer_ref=write('producer.json',producer)
    q['producer']=dict(status='PASS',receipt=producer_ref,artifacts=[device]);q['artifact_inventory']=[device]
    qref=write('qualification.json',freeze('qualification',q).record())
    ledger=write('ledger.json',dict(study=study['identity'],required=['qualify'],receipts=[dict(stage='qualify',status='PASS',study=study['identity'],original=producer['identity'])]))
    measurements=write('attempts.json',dict(roster={'a':'f'},outcomes={'a':True}))
    summary=write('summary.json',dict(n=1,successes=1,unresolved=0,lower='1',upper='1',completion='1',macro_lower='1',macro_upper='1'))
    record=replace_refs(fixtures['evidence']);record.update(study=study_ref,qualification_producers=[qref],stage_ledger=ledger,attempts=measurements,summary=summary,software_status='PASS',scientific_status='UNMEASURED',omissions=[])
    record['inventory']=[base,study_ref,device,producer_ref,qref,ledger,measurements,summary]
    record=freeze('evidence',record).record()
    if stale_family:
        record['families']=write('false-families.json',{'invented_family': {'successes': 999}})
        record['inventory'].append(record['families'])
        record=freeze('evidence',record).record()
        with pytest.raises(ValueError,match='family'):
            verify_bundle(record,expected_study=study['identity'],root=tmp_path)
        return
    assert verify_bundle(record,expected_study=study['identity'],root=tmp_path)['integrity']=='PASS'
    bad=copy.deepcopy(record);bad['software_status']='PARTIAL';bad=freeze('evidence',bad).record()
    with pytest.raises(ValueError,match='stale'): verify_bundle(bad,expected_study=study['identity'],root=tmp_path)
    (tmp_path/'summary.json').write_text('{}')
    with pytest.raises(ValueError): verify_bundle(record,expected_study=study['identity'],root=tmp_path)
