"""Time-only reuse ownership transfer; installed calibration replay tested separately."""
from types import SimpleNamespace
import pytest
from oeis_learn.generalization.progressive.contracts import seal,reference,load,digest
from oeis_learn.pipeline_validation.progressive_workstation_state import write


def fixture(root,monkeypatch):
    from tests.unit.test_progressive_preparation_reuse import construction_owner
    from oeis_learn.pipeline_validation import progressive_reused_preparation_recovery as r,progressive_calibration_reuse as reuse
    from oeis_learn.pipeline_validation.progressive_workstation import command
    from oeis_learn.pipeline_validation.preparation_recovery import Allowance
    folder,receipt=construction_owner(root,monkeypatch);base=folder/'prepare-research'
    qref=reference(root,base/'qualification/qualification.json')
    donor=root/'run-state/original/prepare-research/construction/blueprint.json';write(donor,{})
    reuse_proof=seal(dict(blueprint=reference(root,donor)))
    write(base/'calibration-reuse/calibration-reuse.json',reuse_proof)
    blueprint=seal(dict(TEST_BOUNDARY_ONLY=True));write(base/'calibration-reuse/blueprint.json',blueprint)
    prepared=seal(dict(payload=dict(deficits=['training_time:confirmation-1-17'],references=[dict(name='calibration_reuse',
        artifact=reference(root,base/'calibration-reuse/calibration-reuse.json'))])))
    write(base/'prepared/prepared.json',prepared)
    # Authority/input replays have their own real integration tests. Keep owner,
    # request, path, process, journal and allowance verification real here.
    monkeypatch.setattr(r,'validate',lambda kind,v:v)
    def replay(*a,**kw):
        assert kw=={'historical':True}
        assert a[2]==prepared
    monkeypatch.setattr(reuse,'validate_prepared',replay)
    monkeypatch.setattr(reuse,'source_authority',lambda *a:None)
    trajectory=base/'training-trajectory'
    request=seal(dict(seconds=36300));write(trajectory/'request.json',request)
    with Allowance(trajectory/'trajectory-allowance.json',request['identity'],36300).attempt():pass
    write(trajectory/'container-inspect.json',[dict(State=dict(Running=False,OOMKilled=False,ExitCode=0))])
    write(trajectory/'process.json',dict(returncode=0,failure=None,cleanup_failure=None))
    write(trajectory/'cleanup.json',dict(acknowledged=True,within_bound=True,failure=None))
    session=load(folder/'session.json');session['binding'].update(engine='podman',memory_gib=64,cpus=8,reuse_calibration_session='run-state/original')
    write(folder/'session.json',seal(session))
    result=seal(dict(operation='prepare',mechanical_status='PARTIAL',reason='BOUND_OUTCOME_BLIND_PREPARATION',
        producer_error=None,scientific_status='UNMEASURED',missing_evidence=prepared['payload']['deficits'],
        metrics=dict(prepared=reference(base/'prepared',base/'prepared/prepared.json'))))
    write(base/'prepared/result.json',result)
    write(base/'prepared/owner.json',seal(dict(version='discovery-owner/v1',output=str((base/'prepared').relative_to(root)))))
    write(base/'prepared/operation-binding.json',seal(dict(operation='prepare',blueprint=blueprint['identity'],qualification=qref)))
    logs=base/'logs/prepare/0000'
    argv=command(root,SimpleNamespace(engine='podman',memory_gib=64,cpus=8,render_node=None),'prepare',
        blueprint=str((base/'calibration-reuse/blueprint.json').relative_to(root)),
        qualification=str((base/'qualification').relative_to(root)),output=str((base/'prepared').relative_to(root)))
    write(logs/'command.json',dict(argv=argv));write(logs/'process.json',dict(returncode=2,failure=None,cleanup_failure=None,outcome='COMPLETE'))
    write(logs/'stdout.log',result)
    journal=load(folder/'journal.json')
    for n in ('construction','trajectory-plan','training-trajectory','calibration-reuse-plan'):
        journal['steps']['prepare-research/'+n]=dict(status='COMPLETE',outputs=[])
    journal['steps']['prepare-research/calibration-reuse-plan']['outputs']=[reference(root,base/'calibration-reuse/blueprint.json')]
    journal['steps']['prepare-research/prepare']=dict(status='BLOCKED')
    write(folder/'journal.json',seal(journal))
    return folder,receipt


@pytest.mark.parametrize('fault',[None,'science','freeze','running','history','command','cleanup','stdout','donor','allowance','deficit','blueprint','failed-stage'])
def test_clean_time_only_recovery_is_fenced(tmp_path,monkeypatch,fault):
    from oeis_learn.pipeline_validation.progressive_preparation_owner import failure_proof,claim_owner
    folder,receipt=fixture(tmp_path,monkeypatch);base=folder/'prepare-research'
    journal=load(folder/'journal.json')
    if fault=='science':journal['steps']['run-research/start']=dict(status='COMPLETE')
    if fault=='freeze':(base/'frozen').mkdir()
    if fault=='running':journal['steps']['prepare-research/prepare']['status']='RUNNING'
    if fault=='history':journal['steps']['prepare-research/prepare']['history']=[dict(status='BLOCKED')]
    if fault=='failed-stage':journal['steps']['prepare-research/training-trajectory']['status']='FAILED'
    if fault=='command':write(base/'logs/prepare/0000/command.json',dict(argv=['unrelated']))
    if fault=='cleanup':write(base/'training-trajectory/cleanup.json',dict(acknowledged=False))
    if fault=='stdout':(base/'logs/prepare/0000/stdout.log').write_text('not the result')
    if fault=='donor':
        s=load(folder/'session.json');s['binding']['reuse_calibration_session']='wrong';write(folder/'session.json',seal(s))
    if fault=='allowance':
        p=base/'training-trajectory/trajectory-allowance.json';v=load(p);write(p,seal(dict(v,active={'running':True})))
    if fault=='deficit':
        p=base/'prepared/prepared.json';v=load(p);v['payload']['deficits']=['missing_authority'];write(p,seal(v))
    if fault=='blueprint':write(base/'calibration-reuse/blueprint.json',{})
    write(folder/'journal.json',seal(journal))
    if fault:
        with pytest.raises((ValueError,OSError,AssertionError)):failure_proof(tmp_path,receipt)
    else:
        before=(folder/'journal.json').read_bytes()
        proof=failure_proof(tmp_path,receipt)
        assert proof['version'].endswith('/v6') and proof['scientific_allowance_transferred'] is False
        # Original calibration donor -> reused-calibration time stop -> new owner.
        # The new recipient must keep pointing to that first original donor.
        original=seal(dict(receipt,recipient='run-state/original',current_source=digest('original')))
        v5=seal(dict(version='progressive-preparation-owner-proof/v5',predecessor=original,
            original_costs_retained=True,scientific_allowance_transferred=False,
            blueprint=reference(tmp_path,'run-state/original/prepare-research/construction/blueprint.json')))
        write(tmp_path/'history/original.json',original);write(tmp_path/'history/v5.json',v5)
        previous=seal(dict(version='progressive-preparation-owner/v2',receipt=receipt,
            previous=reference(tmp_path,'history/original.json'),proof=reference(tmp_path,'history/v5.json')))
        claim=tmp_path/'run-state/claim.json';write(claim,previous)
        new=seal(dict(receipt,recipient='run-state/next',current_source=digest('new')))
        claim_owner(tmp_path,claim,new)
        assert load(claim)['receipt']==new and (folder/'journal.json').read_bytes()==before
        from oeis_learn.pipeline_validation.progressive_calibration_reuse_workstation import fenced_original
        w=SimpleNamespace(root=tmp_path,s=SimpleNamespace(folder=tmp_path/'run-state/next'),
            args=SimpleNamespace(reuse_calibration_session='run-state/original'))
        assert fenced_original(w)==tmp_path/'run-state/original/prepare-research'
        with pytest.raises(ValueError,match='superseded'):claim_owner(tmp_path,claim,receipt)
        # Round015 failed CPU qualification after round014's v6 transfer. A
        # subsequent owner must traverse v1 -> v6 -> v5 to the original donor.
        from tests.unit.test_progressive_preparation_reuse import failed_owner
        failed_owner(tmp_path,w.s.folder,new)
        failed_journal=(w.s.folder/'journal.json').read_bytes()
        latest=seal(dict(new,recipient='run-state/after-cpu',current_source=digest('registration fix')))
        claim_owner(tmp_path,claim,latest)
        assert (w.s.folder/'journal.json').read_bytes()==failed_journal
        w.s.folder=tmp_path/latest['recipient']
        assert fenced_original(w)==tmp_path/'run-state/original/prepare-research'
        assert (folder/'journal.json').read_bytes()==before
        w.args.reuse_calibration_session=new['recipient']
        with pytest.raises(ValueError,match='selected calibration donor differs'):fenced_original(w)


def test_reservation_failure_transfers_original_donor_with_no_prepared_publication(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_reused_preparation_recovery as r, progressive_calibration_reuse as reuse
    from oeis_learn.pipeline_validation.progressive_preparation_owner import failure_proof,claim_owner
    from oeis_learn.pipeline_validation.progressive_calibration_reuse_workstation import fenced_original
    folder,receipt=fixture(tmp_path,monkeypatch);base=folder/'prepare-research'
    raw=load(base/'prepared/prepared.json');(base/'prepared/prepared.json').unlink()
    monkeypatch.setattr(reuse,'raw_prepared',lambda *a:raw)
    # Full reservation/authority replay, original traceback and inspection faults
    # are exercised with real preparation inputs by the integration test.
    calls=[]
    def reservation(*a):
        assert a[3]==raw and not (base/'prepared/prepared.json').exists()
        calls.append(1)
        return dict(replayed_resources={'boundary':'integration tested separately'})
    monkeypatch.setattr(r,'reservation_failure',reservation)
    journal=load(folder/'journal.json');journal['steps']['prepare-research/prepare']['status']='FAILED'
    write(folder/'journal.json',seal(journal))
    logs=base/'logs/prepare/0000'
    p=load(logs/'process.json');p['returncode']=3;write(logs/'process.json',p)
    original=seal(dict(receipt,recipient='run-state/original',current_source=digest('original')))
    v5=seal(dict(version='progressive-preparation-owner-proof/v5',predecessor=original,
        original_costs_retained=True,scientific_allowance_transferred=False,
        blueprint=reference(tmp_path,'run-state/original/prepare-research/construction/blueprint.json')))
    write(tmp_path/'history/original.json',original);write(tmp_path/'history/v5.json',v5)
    previous=seal(dict(version='progressive-preparation-owner/v2',receipt=receipt,
        previous=reference(tmp_path,'history/original.json'),proof=reference(tmp_path,'history/v5.json')))
    claim=tmp_path/'run-state/claim.json';write(claim,previous)
    before=(folder/'journal.json').read_bytes()
    proof=failure_proof(tmp_path,receipt)
    assert proof['version'].endswith('/v7') and proof['prepared'] is None
    latest=seal(dict(receipt,recipient='run-state/after-debit',current_source=digest('fixed')))
    claim_owner(tmp_path,claim,latest)
    w=SimpleNamespace(root=tmp_path,s=SimpleNamespace(folder=tmp_path/latest['recipient']),
        args=SimpleNamespace(reuse_calibration_session='run-state/original'))
    assert fenced_original(w)==tmp_path/'run-state/original/prepare-research'
    assert calls and (folder/'journal.json').read_bytes()==before
    assert not (base/'prepared/prepared.json').exists()


def test_pinned_repair_runs_only_unfinished_roles_and_retains_history(tmp_path,monkeypatch):
    import importlib.util,json,zipfile
    from pathlib import Path
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session
    from oeis_learn.pipeline_validation import progressive_workstation as workflow
    path=Path(__file__).resolve().parents[2]/'scripts/discovery_repair_reservation.py'
    spec=importlib.util.spec_from_file_location('repair_script',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    binding=dict(engine='podman',memory_gib=64,cpus=8,source=digest('original'))
    s=Session(tmp_path,m.SESSION,binding,'prepare-research');base=s.folder/'prepare-research'
    script=tmp_path/'run-state/repair.py';script.write_bytes(path.read_bytes())
    for name in ('qualify','evaluation-feasibility','construction','training-trajectory','calibration-reuse-plan','sensitivity'):
        s.journal['steps']['prepare-research/'+name]=dict(status='COMPLETE',outputs=[])
    s.journal['steps']['prepare-research/prepare']=dict(status='FAILED',reason='original debit mismatch',outputs=[])
    for name in ('freeze','acceptance','readiness'):s.journal['steps']['prepare-research/'+name]=dict(status='BLOCKED')
    old=s.journal['steps']['prepare-research/prepare'].copy();s.save()
    write(base/'prepared/result.json',dict(original='untouched'))
    write(base/'qualification/qualification.json',dict(original='qualification boundary'))
    original=(base/'prepared/result.json').read_bytes()
    write(base/'preparation-headroom.json',seal(dict(setup_seconds=dict(freeze=7200,acceptance=7800))))
    monkeypatch.setattr(m,'verify_original',lambda *a:None)
    def repair(*a):
        write(base/'reservation-repair/blueprint.json',dict(original_debit=True))
        write(base/'reservation-repair/reservation-repair.json',dict(remaining_seconds=10000))
        return dict(remaining_seconds=10000,script=reference(tmp_path,script)),[base/'reservation-repair/blueprint.json',base/'reservation-repair/reservation-repair.json']
    monkeypatch.setattr(m,'repair_plan',repair)
    monkeypatch.setattr('oeis_learn.generalization.progressive.campaign.validate_campaign',lambda c:c)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_device_acceptance.consume_receipt',lambda *a:None)
    calls=[]
    def launch(argv,logs,**kw):
        op=argv[argv.index('discovery')+1];calls.append(op)
        output=tmp_path/argv[argv.index('--output')+1]
        if op=='prepare':
            assert '--blueprint' in argv and 'reservation-repair/blueprint.json' in argv[argv.index('--blueprint')+1]
            assert 9800<=kw['seconds']<=10000
            write(output/'resource-estimate.json',dict(status='READY'))
        if op=='freeze':
            write(output/'campaign.json',dict(identity=digest('campaign'),purpose='research',payload=dict(caps=dict(elapsed_us=10000000))))
            write(output/'freeze-publication.json',seal(dict(campaign=digest('campaign'),files=[reference(output,'campaign.json')])))
            write(output/'result.json',seal(dict(operation=op,mechanical_status='COMPLETE',campaign=digest('campaign'))))
        if op=='acceptance':write(output/'acceptance.json',seal(dict(campaign=digest('campaign'))))
        if op=='report':
            output.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(output,'w') as z:z.writestr('result.json','{}')
        elif op!='freeze':write(output/'result.json',dict(mechanical_status='COMPLETE'))
        if op=='verify-bundle':write(output/'verification.json',dict(status='PASS'))
        write(logs/'process.json',dict(returncode=0,outcome='COMPLETE'))
    monkeypatch.setattr('oeis_learn.pipeline_validation.supervision.launch',launch)
    args=SimpleNamespace(phase='prepare-research',engine='podman',memory_gib=64,cpus=8,render_node=None)
    with s.own():m.continue_preparation(tmp_path,s,args,script)
    assert calls==['prepare','freeze','acceptance','run','report','verify-bundle']
    assert s.journal['steps']['prepare-research/prepare']['history']==[old]
    assert (base/'prepared/result.json').read_bytes()==original
    assert (s.folder/'share/run-research.zip').is_file()
    s=Session(tmp_path,m.SESSION,binding,'prepare-research')
    with s.own():m.continue_preparation(tmp_path,s,args,script)
    assert calls==['prepare','freeze','acceptance','run','report','verify-bundle']
    assert (base/'prepared/result.json').read_bytes()==original


@pytest.mark.parametrize('fault',[None,'source','commit','owner','image','active','output'])
def test_pinned_repair_requires_original_source_owner_and_outputs(tmp_path,monkeypatch,fault):
    import importlib.util
    from pathlib import Path
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session
    path=Path(__file__).resolve().parents[2]/'scripts/discovery_repair_reservation.py'
    spec=importlib.util.spec_from_file_location('repair_check',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source={'original':digest('code')};s=Session(tmp_path,m.SESSION,dict(source=digest(source)),'prepare-research')
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'wrong' if fault=='commit' else m.QUALIFIED_COMMIT)
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.collect_snapshot',lambda root:{} if fault=='source' else source)
    receipt=dict(census=digest('census'),recipient=m.SESSION);owner=seal(dict(receipt=receipt))
    write(s.folder/'preparation-owner.json',owner)
    claim=tmp_path/'run-state/discovery-preparation-revisions'/('runtime-'+receipt['census'][7:]+'.json')
    write(claim,seal(dict(receipt={}))) if fault=='owner' else write(claim,owner)
    base=s.folder/'prepare-research';images=[digest('cpu'),digest('gpu')]
    write(base/'qualification/qualification.json',seal(dict(images=images)))
    for name,image in zip(('qualification','gpu'),images):
        # Actual image.json publications are plain JSON, not sealed records.
        write(base/('image-'+name)/'image.json',dict(image=digest('changed') if fault=='image' else image))
    p=base/'result.json';write(p,dict(original=True))
    s.journal['steps']['prepare-research/qualify']=dict(status='COMPLETE',outputs=[reference(tmp_path,p)])
    if fault=='active':s.journal['steps']['prepare-research/prepare']=dict(status='INTERRUPTED')
    if fault=='output':write(p,dict(changed=True))
    if fault:
        with pytest.raises(ValueError):m.verify_original(tmp_path,s)
    else:m.verify_original(tmp_path,s)
