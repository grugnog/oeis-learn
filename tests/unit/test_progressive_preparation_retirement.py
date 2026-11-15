"""Exact diagnostic tensor retirement never removes scientific state or evidence."""
from pathlib import Path
import pytest
from oeis_learn.generalization.progressive.contracts import seal,digest,reference,record
from oeis_learn.pipeline_validation.progressive_workstation_state import write,read
from oeis_learn.pipeline_validation import discovery_cleanup as cleanup


def fixture(root,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_handoff as h,progressive_runtime_timing as rt
    from tests.helpers.progressive_fixture import task
    session='run-state/discovery/preparation';folder=root/session;base=folder/'prepare-research';qroot=base/'qualification'
    source={'test_boundary_only':True};write(qroot/'source.json',source)
    write(folder/'session.json',seal(dict(binding=dict(source=digest(source)))))
    q=dict(source_closure=reference(qroot,'source.json'),images=[digest('cpu'),digest('gpu')],engine_security=dict(engine='podman'))
    write(qroot/'qualification.json',q)
    write(qroot/'result.json',seal(dict(mechanical_status='COMPLETE',qualification_status='QUALIFIED',metrics=dict(qualification=reference(qroot,'qualification.json')))))
    qcase=qroot/'device/cases/scope';write(qcase/'producer.json',seal(dict(status='PASS',scope='TEST BOUNDARY ONLY')))
    # Only original hardware authority is doubled; file allowlisting, exact
    # manifests, path confinement, locks, byte hashing and retirement are real.
    monkeypatch.setattr(h,'original_authority',lambda *a:(q,qroot))
    monkeypatch.setattr(h,'validate_device_proofs',lambda *a,**kw:({},[reference(qroot,qcase/'producer.json')]))
    def terminated(p):
        write(p/'container-inspect.json',[dict(State=dict(Running=False,OOMKilled=False,ExitCode=0))])
        write(p/'process.json',dict(returncode=0,failure=None,cleanup_failure=None))
        write(p/'cleanup.json',dict(acknowledged=True,within_bound=True,failure=None))
    terminated(qroot/'device')
    all_tensors=[]
    def checkpoint(cp):
        gen=cp/('generalization-'+'a'*32);gen.mkdir(parents=True)
        for name in ('weights','optimizer','rng-torch'):
            path=gen/(name+'.pt');path.write_bytes(name.encode())
            if name!='rng-torch':all_tensors.append(path)
        write(gen/'state.json',seal(dict(version='generalization-learner-state/v1',states={n:reference(cp,gen/(n+'.pt')) for n in ('weights','optimizer','rng-torch')})))
    checkpoint(qcase/'checkpoints')
    tr=base/'training-trajectory';terminated(tr)
    req=seal(dict(version=rt.FEASIBILITY_VERSION,source=digest(source),image=q['images'][1],inputs=digest('i'),original_inputs=digest('o'),
        package=dict(path='private/input.json',bytes=1,sha256=digest('x')),cases=rt.feasibility_scopes(),target_tokens=rt.TARGET,
        learner_seconds=rt.LEARN_SECONDS,calibration_seconds=rt.CALIBRATION_SECONDS,seconds=rt.FEASIBILITY_SECONDS,
        scope='DIAGNOSTIC_ONLY; no scientific selection or qualification'))
    write(tr/'request.json',req);refs=[];items=[]
    for i,case in enumerate(req['cases']):
        out=tr/'cases'/str(i);out.mkdir(parents=True)
        (out/'diagnostics.zip').write_bytes(b'original diagnostics boundary')
        p=seal(dict(status='PASS',scope='DIAGNOSTIC_ONLY',request=req['identity'],case=case,diagnostics=reference(out,'diagnostics.zip')))
        write(out/'producer.json',p);refs.append(reference(root,out/'producer.json'))
        items.append(dict(index=i,status='PASS',producer=reference(tr,out/'producer.json')))
        t=task(id='first')['payload'];t.pop('campaign')
        campaign=seal(dict(version='discovery-campaign/v1',purpose='acceptance',track='synthetic_strict',created_at='2026-11-12T00:00:00Z',payload=dict(tasks=[t])))
        write(out/'private/learner/campaign.json',campaign)
        bound=record('task',dict(t,campaign=campaign['identity']),created_at=campaign['created_at'],purpose='acceptance',track=campaign['track'])
        checkpoint(out/'private/learner/learning'/bound['identity'][7:]/'checkpoints')
    write(tr/'runtime-cases.json',seal(dict(request=req['identity'],status='PASS',device_block=None,cases=items)))
    write(base/'prepared/prepared.json',seal(dict(payload=dict(deficits=['training_time:confirmation']))))
    write(folder/'journal.json',seal(dict(steps={
        'prepare-research/qualify':dict(status='COMPLETE',outputs=[reference(root,qroot/'result.json')]),
        'prepare-research/training-trajectory':dict(status='COMPLETE',outputs=refs),
        'prepare-research/prepare':dict(status='BLOCKED')})))
    return session,folder,all_tensors


def test_preparation_diagnostic_retirement_keeps_all_original_evidence(tmp_path,monkeypatch):
    session,folder,tensors=fixture(tmp_path,monkeypatch)
    originals={p:p.read_bytes() for p in folder.rglob('*') if p.is_file() and p not in tensors}
    output=tmp_path/'run-state/maintenance/diagnostic-retirement'
    plan=cleanup.plan(tmp_path,[session],output)
    assert len(plan['files'])==28 and all(p.exists() for p in tensors)
    assert cleanup.apply(tmp_path,output)['files']==28
    assert all(not p.exists() for p in tensors)
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    assert cleanup.apply(tmp_path,output)['files']==28


@pytest.mark.parametrize('fault',['frozen','science','active','failed','unbound','source','checkpoint-alias','tensor','campaign'])
def test_retirement_rejects_unproven_or_scientific_state(tmp_path,monkeypatch,fault):
    session,folder,tensors=fixture(tmp_path,monkeypatch);base=folder/'prepare-research'
    j=read(folder/'journal.json')
    if fault=='frozen':(base/'frozen').mkdir()
    if fault=='science':j['steps']['run-research/start']=dict(status='COMPLETE')
    if fault=='active':j['steps']['prepare-research/prepare']['status']='RUNNING'
    if fault=='failed':j['steps']['prepare-research/training-trajectory']['status']='FAILED'
    if fault=='unbound':j['steps']['prepare-research/qualify']['outputs']=[]
    if fault=='source':write(folder/'session.json',seal(dict(binding=dict(source=digest('changed')))))
    if fault=='checkpoint-alias':tensors[0].unlink();tensors[0].symlink_to(tensors[1])
    if fault=='tensor':tensors[0].write_bytes(b'changed')
    if fault=='campaign':
        p=base/'training-trajectory/cases/0/private/learner/campaign.json';v=read(p);write(p,seal(dict(v,purpose='research')))
    write(folder/'journal.json',seal(j))
    with pytest.raises((ValueError,OSError)):
        output=tmp_path/'run-state/maintenance/no';cleanup.plan(tmp_path,[session],output);cleanup.apply(tmp_path,output)
    assert tensors[1].exists()
