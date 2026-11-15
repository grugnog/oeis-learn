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
