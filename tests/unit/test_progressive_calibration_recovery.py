"""No hardware is simulated as evidence: local tests cover the ownership boundary."""
import pytest
from oeis_learn.generalization.progressive.contracts import load, reference, record, seal, digest
from oeis_learn.pipeline_validation.progressive_workstation_state import write
from oeis_learn.pipeline_validation.progressive_calibration_recovery import ERROR


def failed_preflight(root, monkeypatch):
    from tests.integration.test_progressive_calibration import host_inputs
    from tests.unit.test_progressive_preparation_reuse import construction_owner
    scratch=root/'fixture'
    scratch.mkdir()
    b=host_inputs(scratch,monkeypatch)
    preparation=load(scratch/'preparation-plan.json')
    folder,receipt=construction_owner(root,monkeypatch)
    base=folder/'prepare-research';out=base/'prepared';construction=base/'construction'
    plan=load(scratch/'calibration-plan.json')
    write(construction/'calibration-plan.json',plan)
    b=seal(dict(b,purpose='research',payload=dict(b['payload'],references=[dict(name='iteration_calibration_plan',
        role='custodian',kind='policy',artifact=reference(root,construction/'calibration-plan.json'))])))
    write(construction/'blueprint.json',b)
    write(construction/'preparation-plan.json',preparation)
    summary=seal(dict(status='COMPLETE',assessment_exposed=False,executable=False,exact_calibration_cases=1,
        blueprint=reference(root,construction/'blueprint.json')))
    write(construction/'construction-summary.json',summary)
    journal=load(folder/'journal.json')
    journal['steps']['prepare-research/construction']=dict(status='COMPLETE',outputs=[reference(root,construction/n) for n in
        ('calibration-plan.json','blueprint.json','preparation-plan.json','construction-summary.json')])
    journal['steps']['prepare-research/prepare']=dict(status='FAILED',elapsed_seconds=2)
    write(folder/'journal.json',seal(journal))
    s=load(folder/'session.json');s['binding'].update(engine='podman',cpus=8,memory_gib=64)
    write(folder/'session.json',seal(s))
    write(out/'owner.json',seal(dict(version='discovery-owner/v1',output=str(out.relative_to(root)),invocation='fixture')))
    write(out/'operation-binding.json',seal(dict(operation='prepare',blueprint=b['identity'],
        qualification=reference(root,base/'qualification/qualification.json'))))
    trace=out/'private/host-traceback.txt';trace.parent.mkdir()
    trace.write_text('Traceback (most recent call last):\n'
        f'  File "{root}/src/oeis_learn/pipeline_validation/progressive_calibration.py", line 214, in _prepare\n'
        '    guard.check(required_bytes, 64 * len(cases) + 256)\n'
        f'  File "{root}/src/oeis_learn/generalization/progressive/storage.py", line 246, in check\n'
        f'    raise TimeoutError("{ERROR}")\nTimeoutError: {ERROR}\n')
    result=seal(dict(version='discovery-result/v1',operation='prepare',mechanical_status='FAILED',
        scientific_status='UNMEASURED',reason='HOST_FAILURE',producer_error=dict(type='TimeoutError',message=ERROR,
        producer='host',code='PRODUCER_FAILURE',local_traceback=reference(out,trace))))
    write(out/'result.json',result)
    state=root/'run-state/discovery-calibration'/digest(['iteration-calibration',b['identity']])[7:]
    prepared=record('prepared',dict(preparation['payload'],blueprint=b['identity']),created_at=b['created_at'],purpose=b['purpose'],track=b['track'])
    write(state/'preparation-allowance.json',seal(dict(seconds=plan['seconds'],active=None,spent_seconds=2,
        binding=dict(blueprint=b['identity'],prepared=prepared['identity'],output=str(out.relative_to(root))),
        attempts=[dict(outcome='TimeoutError',error=ERROR,charged_seconds=2)])))
    (state/'preparation-allowance.lock').touch()
    logs=base/'logs/prepare/0000'
    write(logs/'command.json',dict(argv=['python3','-S',str(root/'scripts/foundation_run.py'),'--engine','podman',
        '--memory-gib','64','--cpus','8','--host-free-memory-gib','24','discovery','prepare',
        '--blueprint',str((construction/'blueprint.json').relative_to(root)),
        '--qualification',str((base/'qualification').relative_to(root)),'--output',str(out.relative_to(root))]))
    write(logs/'process.json',dict(returncode=1,cleanup_failure=None,failure=None,outcome='COMPLETE',elapsed_seconds=3))
    write(logs/'stdout.log',result)
    return folder,receipt,state


@pytest.mark.parametrize('fault',[None,'inspection','calibration','recovery','prepared','frozen','science','active',
    'budget','changed-trace','later-frame','other-error','unreaped','other-command','stdout','allowance',
    'construction','history','binding','source','cap','state-alias'])
def test_preflight_recovery_requires_original_pre_dispatch_failure(tmp_path,monkeypatch,fault):
    from oeis_learn.pipeline_validation.progressive_preparation_owner import failure_proof,claim_owner
    folder,receipt,state=failed_preflight(tmp_path,monkeypatch)
    base=folder/'prepare-research';out=base/'prepared';logs=base/'logs/prepare/0000'
    journal=load(folder/'journal.json')
    if fault in {'inspection','calibration','recovery'}:(out/fault).mkdir()
    if fault=='prepared':write(out/'prepared.json',{})
    if fault=='frozen':(base/'frozen').mkdir()
    if fault=='science':journal['steps']['run-research/start']=dict(status='COMPLETE')
    if fault=='active':journal['steps']['prepare-research/prepare']['status']='RUNNING'
    if fault=='history':journal['steps']['prepare-research/prepare']['history']=[dict(status='FAILED')]
    if fault=='budget':write(state/'budget-binding.json',{})
    if fault=='state-alias':
        (state/'preparation-allowance.lock').unlink();(state/'preparation-allowance.lock').symlink_to(tmp_path/'external')
    if fault=='changed-trace':(out/'private/host-traceback.txt').write_text('changed')
    if fault in {'later-frame','other-error'}:
        result=load(out/'result.json')
        if fault=='later-frame':
            path=out/'private/host-traceback.txt';path.write_text(path.read_text().replace('guard.check(required_bytes, 64 * len(cases) + 256)','guard.check()'))
            result['producer_error']['local_traceback']=reference(out,path)
        else:result['producer_error']['message']='free storage byte/inode reserve exhausted'
        write(out/'result.json',seal(result));write(logs/'stdout.log',seal(result))
    if fault=='unreaped':
        p=load(logs/'process.json');p['cleanup_failure']='live group';write(logs/'process.json',p)
    if fault=='other-command':
        p=load(logs/'command.json');p['argv'][-1]='other';write(logs/'command.json',p)
    if fault=='stdout':write(logs/'stdout.log',{})
    if fault=='allowance':
        p=load(state/'preparation-allowance.json');p['spent_seconds']=0;write(state/'preparation-allowance.json',seal(p))
    if fault=='construction':write(base/'construction/construction-summary.json',{})
    if fault=='binding':write(out/'operation-binding.json',seal({}))
    if fault=='source':
        s=load(folder/'session.json');s['binding']['source']=digest('other');write(folder/'session.json',seal(s))
    if fault=='cap':
        p=load(base/'construction/calibration-plan.json');p['maximum_inodes']+=1;write(base/'construction/calibration-plan.json',seal(p))
    write(folder/'journal.json',seal(journal))
    if fault:
        with pytest.raises((ValueError,OSError)):failure_proof(tmp_path,receipt)
    else:
        original={p:p.read_bytes() for path in (folder,state) for p in path.rglob('*') if p.is_file()}
        proof=failure_proof(tmp_path,receipt)
        assert proof['allowance']['value']['spent_seconds']==2
        assert proof['scientific_allowance_transferred'] is False
        assert proof['current_source_requalification_required'] and proof['historical_qualification_only']
        claim=tmp_path/'run-state/discovery-preparation-revisions/test.json';write(claim,receipt)
        next_receipt=seal(dict(receipt,recipient='run-state/discovery/next',current_source=digest('fixed')))
        claim_owner(tmp_path,claim,next_receipt)
        assert load(claim)['receipt']==next_receipt
        assert all(p.read_bytes()==raw for p,raw in original.items())
        with pytest.raises(ValueError,match='superseded'):claim_owner(tmp_path,claim,receipt)
