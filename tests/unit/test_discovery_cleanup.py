"""Cleanup is confined to completed diagnostics; original evidence survives."""
import fcntl
import os
from pathlib import Path
import pytest
from oeis_learn.generalization.progressive.contracts import seal,digest,reference
from oeis_learn.pipeline_validation.progressive_workstation_state import write,read
from oeis_learn.pipeline_validation import discovery_cleanup as c


@pytest.fixture
def timing(tmp_path):
    from oeis_learn.pipeline_validation.progressive_training_timing import VERSION,SECONDS,UPDATES,CHECKPOINT_BYTES
    root=tmp_path;session='run-state/discovery/training-timing';folder=root/session;base=folder/'measure-training/timing'
    write(folder/'session.json',seal(dict(binding=dict(source=digest('original')))))
    req=seal(dict(version=VERSION,source=digest('source'),image=digest('image'),inputs=digest('inputs'),
        package=dict(path='private/training-input.json',bytes=1,sha256=digest('input')),seconds=SECONDS,
        updates=UPDATES,checkpoint_bytes=CHECKPOINT_BYTES,
        scope='DIAGNOSTIC_ONLY; reference recipe; no scientific checkpoint or learning claim'))
    write(base/'request.json',req)
    generation=base/'measurement/checkpoints'/('generalization-'+'a'*32);generation.mkdir(parents=True)
    for name in ('weights','optimizer','rng-torch'):(generation/(name+'.pt')).write_bytes(name.encode()*100)
    states={n:reference(generation.parent,generation/(n+'.pt')) for n in ('weights','optimizer','rng-torch')}
    write(generation/'state.json',seal(dict(version='generalization-learner-state/v1',states=states)))
    write(base/'measurement/producer.json',seal(dict(status='PASS',scope='DIAGNOSTIC_ONLY; original timing')))
    write(base/'training-runtime-review.json',seal(dict(status='MEASURED')))
    write(folder/'journal.json',seal(dict(steps={
        'measure-training/timing-plan':dict(status='COMPLETE',outputs=[reference(root,base/'request.json')]),
        'measure-training/training-measurements':dict(status='COMPLETE',outputs=[reference(root,base/'measurement/producer.json')]),
        'measure-training/runtime-review':dict(status='COMPLETE',outputs=[reference(root,base/'training-runtime-review.json')])})))
    return root,session,generation,root/'run-state/maintenance/cleanup'


def test_plan_is_read_only_and_apply_preserves_history(timing):
    root,session,generation,output=timing
    originals={p:p.read_bytes() for p in (root/session).rglob('*') if p.is_file()}
    plan=c.plan(root,[session],output)
    assert len(plan['files'])==2 and (generation/'weights.pt').exists()
    receipt=c.apply(root,output)
    assert receipt['status']=='RETIRED' and not (generation/'weights.pt').exists()
    for p,raw in originals.items():
        if p.name not in {'weights.pt','optimizer.pt'}:assert p.read_bytes()==raw
    assert read(output/'retirement-intent.json')['files'][0]['sha256']
    assert c.apply(root,output)==receipt
    assert (output/'cleanup-report.zip').is_file()


@pytest.mark.parametrize('change',['content','symlink','producer','extra','outside'])
def test_changed_or_unapproved_files_abort_before_any_deletion(timing,change):
    root,session,generation,output=timing;c.plan(root,[session],output)
    target=generation/'weights.pt'
    if change=='content':target.write_bytes(b'changed')
    if change=='symlink':
        target.unlink();target.symlink_to(generation/'rng-torch.pt')
    if change=='producer':write(root/session/'measure-training/timing/measurement/producer.json',seal(dict(status='FAIL')))
    if change=='extra':
        new=generation.with_name('generalization-'+'b'*32);new.mkdir()
        for name in ('weights','optimizer'):(new/(name+'.pt')).write_bytes(b'extra')
        write(new/'state.json',seal(dict(version='generalization-learner-state/v1',states={n:reference(new.parent,new/(n+'.pt')) for n in ('weights','optimizer')})))
    if change=='outside':
        path=root/'important.pt';path.write_bytes(b'keep')
        value=read(output/'plan.json');value['files'].append(dict(path='important.pt',original_sha256=digest('keep'),**c.stamp(path)))
        write(output/'plan.json',seal(value))
    with pytest.raises((ValueError,OSError)):c.apply(root,output)
    assert (generation/'optimizer.pt').exists()


def test_locked_or_unreconciled_owner_blocks_plan(timing):
    root,session,generation,output=timing
    with (root/'run-state/gpu.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):c.plan(root,[session],output)
    write(root/'run-state/gpu-pending.json',dict(name='pending'))
    with pytest.raises(ValueError,match='GPU owner'):c.plan(root,[session],output)
    assert (generation/'weights.pt').exists()


def test_interrupted_retirement_resumes_without_losing_evidence(timing,monkeypatch):
    root,session,generation,output=timing;c.plan(root,[session],output)
    unlink=Path.unlink;count=0
    def interrupted(path,*a,**kw):
        nonlocal count
        if path.name in {'weights.pt','optimizer.pt'}:
            count+=1
            if count==2:raise OSError('simulated interruption')
        return unlink(path,*a,**kw)
    monkeypatch.setattr(Path,'unlink',interrupted)
    with pytest.raises(OSError,match='interruption'):c.apply(root,output)
    assert (output/'retirement-intent.json').exists()
    monkeypatch.setattr(Path,'unlink',unlink)
    assert c.apply(root,output)['status']=='RETIRED'
    assert (generation/'state.json').exists()


def test_runtime_cleanup_follows_task_identities_not_display_names(tmp_path):
    from oeis_learn.pipeline_validation import progressive_runtime_timing as rt
    from oeis_learn.generalization.progressive.contracts import record
    from tests.helpers.progressive_fixture import task
    root=tmp_path;session='run-state/discovery/runtime';base=root/session/'measure-runtime/timing'
    write(root/session/'session.json',seal(dict(binding={})))
    req=seal(dict(version=rt.VERSION,source=digest('s'),image=digest('i'),inputs=digest('in'),original_inputs=digest('original'),
        package=dict(path='private/input.json',bytes=1,sha256=digest('x')),cases=rt.scopes(),target_tokens=rt.TARGET,
        learner_seconds=rt.LEARN_SECONDS,calibration_seconds=rt.CALIBRATION_SECONDS,seconds=rt.HOST_SECONDS,
        scope='DIAGNOSTIC_ONLY; no scientific selection or qualification'))
    write(base/'request.json',req);refs=[];all_tensors=[]
    for i,case in enumerate(rt.scopes()):
        folder=base/'cases'/str(i);folder.mkdir(parents=True)
        (folder/'diagnostics.zip').write_bytes(b'original diagnostic archive boundary')
        roots=[]
        if case['kind']=='learner':
            tasks=[]
            for name in ('first','second'):
                t=task(id=name)['payload'];t.pop('campaign');tasks.append(t)
            campaign=seal(dict(version='discovery-campaign/v1',purpose='acceptance',track='synthetic_strict',
                created_at='2026-11-12T00:00:00Z',payload=dict(tasks=tasks)))
            write(folder/'private/learner/campaign.json',campaign)
            for t in tasks:
                bound=record('task',dict(t,campaign=campaign['identity']),created_at=campaign['created_at'],purpose='acceptance',track='synthetic_strict')
                roots.append(folder/'private/learner/learning'/bound['identity'][7:]/'checkpoints')
        else:roots.append(folder/'private/calibration/checkpoints')
        for checkpoint_root in roots:
            gen=checkpoint_root/('generalization-'+'a'*32);gen.mkdir(parents=True)
            for name in ('weights','optimizer'):
                path=gen/(name+'.pt');path.write_bytes(name.encode());all_tensors.append(path)
            write(gen/'state.json',seal(dict(version='generalization-learner-state/v1',states={n:reference(checkpoint_root,gen/(n+'.pt')) for n in ('weights','optimizer')})))
        write(folder/'producer.json',seal(dict(status='PASS',scope='DIAGNOSTIC_ONLY',request=req['identity'],case=case,diagnostics=reference(folder,'diagnostics.zip'))))
        refs.append(reference(root,folder/'producer.json'))
    write(base/'runtime-review.json',seal(dict(status='MEASURED')))
    write(root/session/'journal.json',seal(dict(steps={
        'measure-runtime/timing-plan':dict(status='COMPLETE',outputs=[reference(root,base/'request.json')]),
        'measure-runtime/runtime-measurements':dict(status='COMPLETE',outputs=refs),
        'measure-runtime/runtime-review':dict(status='COMPLETE',outputs=[reference(root,base/'runtime-review.json')])})))
    out=root/'run-state/maintenance/runtime'
    assert len(c.plan(root,[session],out)['files'])==36
    assert c.apply(root,out)['files']==36
    assert all(not p.exists() for p in all_tensors)
    assert len(list(base.glob('cases/*/producer.json')))==11
