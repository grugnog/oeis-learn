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
    assert preflight(study_bytes=10,checkpoint_bytes=5,shard_bytes=1,raw_bytes=1,sharing_bytes=1,available_bytes=100)['minutes']==300
    with pytest.raises(ValueError): preflight(study_bytes=100,checkpoint_bytes=5,shard_bytes=1,raw_bytes=1,sharing_bytes=1,available_bytes=90)
    with pytest.raises(ValueError): reserve_stage('cases',elapsed_seconds=200*60,requested_seconds=75*60)
    tail=log_tail(('x'*20000+'\n')*200)
    assert len(tail.encode())<=1024**2 and len(tail.splitlines())<=100


def test_extended_startup_budget_still_protects_packaging_and_margin():
    # Exactly 30 minutes remain: 20 for packaging, 10 emergency margin.
    with pytest.raises(ValueError,match='preserve packaging'):
        reserve_stage('cases',elapsed_seconds=270*60,requested_seconds=1)
    assert reserve_stage('packaging',elapsed_seconds=270*60,requested_seconds=1200)['deadline_seconds']==290*60
    with pytest.raises(ValueError):reserve_stage('packaging',elapsed_seconds=270*60+1,requested_seconds=1200)
    with pytest.raises(ValueError,match='ceiling'):reserve_stage('preflight',elapsed_seconds=0,requested_seconds=2701)

def test_required_results_cannot_be_silently_redacted():
    with pytest.raises(ValueError): bundle_plan([dict(path='weights',bytes=1,private=True,required=True)],mandatory={'weights'})
    with pytest.raises(ValueError): bundle_plan([dict(path='huge',bytes=513*1024**2,private=False,required=True)],mandatory={'huge'})
    assert bundle_plan([dict(path='private',bytes=1,private=True,required=False)],mandatory=set())['omissions']==['private']
    with pytest.raises(ValueError): verify_bundle({'study':'s','stages':[]},expected_study='other',root='.')

@pytest.mark.parametrize("stale_family", [False, True, "seeds", "costs", "bounds", "coverage", "summary", "partial", "archive"])
def test_offline_complete_bundle_and_stale_summary(tmp_path, stale_family, monkeypatch):
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
    measurements=write('attempts.json',dict(version='generalization-attempts/v1',attempts=[dict(row='a',ordinal=0,outcome=True,queue_us=1,service_us=2,cleanup_us=None)]))
    summary=write('summary.json',dict(n=1,successes=1,unresolved=0,lower='1',upper='1',completion='1',macro_lower='1',macro_upper='1'))
    record=replace_refs(fixtures['evidence']);record.update(study=study_ref,qualification_producers=[qref],stage_ledger=ledger,attempts=measurements,summary=summary,software_status='PASS',scientific_status='UNMEASURED',omissions=[])
    record['inventory']=[base,study_ref,device,producer_ref,qref,ledger,measurements,summary]
    golden=dict(n=1,successes=1,unresolved=0,lower='1',upper='1',completion='1',macro_lower='1',macro_upper='1')
    seed=study['seeds'][0]
    members=dict(
        effective_invocation=dict(version='generalization-measurement-plan/v1',study=study['identity'],required_stages=['qualify'],roster={'a':dict(family='f',seed=seed,cell='structural/raw',attempts=1)}),
        families={'structural/raw':{str(seed):{'f':golden}}},
        seeds={'structural/raw':{str(seed):golden}},
        costs=dict(queue=dict(measured_us=1,unavailable=0),service=dict(measured_us=2,unavailable=0),cleanup=dict(measured_us=0,unavailable=1)),
        coverage=dict(expected_rows=1,resolved_rows=1,expected_attempts=1,recorded_attempts=1,missing_attempts=0),
        bounds={k:golden[k] for k in ('lower','upper','completion','macro_lower','macro_upper')})
    for key,value in members.items():
        record[key]=write(key+'.json',value);record['inventory'].append(record[key])
    record=freeze('evidence',record).record()
    if stale_family == 'archive':
        from oeis_learn.pipeline_validation.generalization_reporting import package_evidence
        from oeis_learn.pipeline_validation.generalization_archive import verify_archive
        archive=package_evidence(tmp_path,record,output=tmp_path/'export')
        assert verify_archive(archive)['integrity']=='PASS'
        assert (tmp_path/'export/payloads/summary.json').read_bytes()==(tmp_path/'summary.json').read_bytes()
        assert json.loads((tmp_path/'export/summary.json').read_text())['selected_scope']=='PASS'
        with pytest.raises(ValueError):package_evidence(tmp_path,record,output=tmp_path/'export')
        from types import SimpleNamespace
        from oeis_learn.pipeline_validation import generalization_cli, generalization_roles, generalization_role
        from oeis_learn import container_validation
        write('evidence.json',record)
        image='sha256:'+'1'*64
        host=SimpleNamespace(engine=SimpleNamespace(image=lambda value:value))
        monkeypatch.setattr(container_validation,'inspect_current_image',lambda *a,**kw:None)
        operations=[]
        def invoke(host,image,root,role,artifacts,**kwargs):
            operations.append(kwargs['operation'])
            inputs=kwargs.get('source_roots',{}).get('public_evidence',root)
            target=root/kwargs['output']/'role';target.mkdir(parents=True)
            handler=generalization_role.package if kwargs['operation']=='package' else generalization_role.verify_archive
            return handler({'public_evidence':inputs},target,kwargs['settings'],seconds=kwargs['seconds'])
        monkeypatch.setattr(generalization_roles,'project_and_invoke',invoke)
        args=SimpleNamespace(operation='report',inputs='.',manifest='study.json',output='reported',image=image)
        assert generalization_cli.main(args,tmp_path,lambda args:host)==0
        assert verify_archive(tmp_path/'reported.zip')['integrity']=='PASS'
        args=SimpleNamespace(operation='verify-bundle',archive='reported.zip',output='checked',image=image)
        assert generalization_cli.main(args,tmp_path,lambda args:host)==0
        assert operations==['package','verify-archive']
        return
    if stale_family == 'partial':
        partial_ledger=write('partial-ledger.json',dict(study=study['identity'],required=['qualify'],receipts=[]))
        record['stage_ledger']=partial_ledger;record['inventory'].append(partial_ledger)
        record['software_status']='PARTIAL'
        record=freeze('evidence',record).record()
        verified=verify_bundle(record,expected_study=study['identity'],root=tmp_path)
        assert verified['integrity']=='PASS' and verified['software_status']=='PARTIAL'
        return
    if isinstance(stale_family,str):
        record[stale_family]=write('false-'+stale_family+'.json',{})
        record['inventory'].append(record[stale_family]);record=freeze('evidence',record).record()
        with pytest.raises(ValueError,match=stale_family):
            verify_bundle(record,expected_study=study['identity'],root=tmp_path)
        return
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
