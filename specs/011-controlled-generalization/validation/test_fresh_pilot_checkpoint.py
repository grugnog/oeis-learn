import io,json,pickle,sys,zipfile
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fresh_pilot_checkpoint as m
from oeis_learn.pipeline_validation.identity import write,seal,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.budgets import Ledger


def fixture(root,pending=False):
    command=root/'execution/train-static';cp=command/'training/run/execution/role/private/learner/checkpoints'
    cp.mkdir(parents=True)
    folder=cp/'generalization-complete';folder.mkdir()
    (folder/'weights.pt').write_bytes(b'PRIVATE WEIGHTS')
    with zipfile.ZipFile(folder/'scheduler.pt','w') as z:z.writestr('archive/data.pkl',pickle.dumps(dict(learner_seconds=21560.)))
    ledger=Ledger(25000000,consumed=1000,updates=2)
    if pending:ledger=ledger.reserve('next-batch',100)
    state=seal(dict(version='generalization-learner-state/v1',kind='full_state',contract=digest('config'),boundary='before_or_after_complete_update',
        ledger=ledger.snapshot(),states=dict(weights=reference(folder/'weights.pt',cp),scheduler=reference(folder/'scheduler.pt',cp),
            optimizer={},history={},pins={},replay={},selection={},rng=dict(python={},numpy={},torch={},device={}),data=dict(cursor=3,pending_batch=[dict(source='PRIVATE SOURCE',values=['PRIVATE TERMS'])] if pending else None))))
    write(folder/'state.json',state);ref=reference(folder/'state.json',cp)
    write(cp/'generalization-latest.json',ref);write(cp/'generalization-pins.json',{ref['sha256']:ref})
    (cp/'generalization-interrupted').mkdir();(cp/'generalization-interrupted/optimizer.pt').write_bytes(b'PARTIAL')
    inventory=[reference(p,command) for p in command.rglob('*') if p.is_file()]
    receipt=seal(dict(version='generalization-command-result/v2',stage='train',status='FAIL',study=digest('study'),
        invocation=seal(dict(selected=dict(arms=['static'],seeds=[20261016]))),inventory=inventory,result=dict(reason='TimeoutError')))
    write(command/'result.json',receipt)
    return command,cp,folder,receipt


@pytest.mark.parametrize('pending',[False,True])
def test_read_only_metadata_distinguishes_saved_weights_from_reserved_update(tmp_path,pending):
    command,cp,folder,receipt=fixture(tmp_path,pending)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=m.inspect(command)
    assert result['committed_tokens']==1000 and result['updates']==2
    assert result['pending_tokens']==(100 if pending else 0)
    assert result['learner_seconds']==21560 and result['state']==reference(folder/'state.json',cp)
    assert result['command']==receipt['identity'] and result['original_status']=='FAIL'
    assert result['unreferenced_generation_count']==1 and result['resume_authorized'] is False
    assert not any(v in json.dumps(result) for v in ['PRIVATE','PARTIAL','source','values'])
    assert before=={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change',['blob','state','pointer','symlink','inventory'])
def test_corrupt_originals_fail_closed(tmp_path,change):
    command,cp,folder,receipt=fixture(tmp_path)
    if change=='blob':(folder/'weights.pt').write_bytes(b'changed')
    if change=='state':(folder/'state.json').write_text('{}')
    if change=='pointer':write(cp/'generalization-latest.json',dict(path='../escape'),exclusive=False)
    if change=='symlink':
        p=folder/'weights.pt';p.rename(folder/'saved.pt');p.symlink_to('saved.pt')
    if change=='inventory':write(command/'result.json',seal(dict(receipt,inventory=[])),exclusive=False)
    with pytest.raises((ValueError,KeyError)):m.inspect(command)


def test_scheduler_rejects_executable_pickle_and_nonfinite_values(tmp_path):
    for value in [eval,dict(learner_seconds=float('nan')),dict(learner_seconds=True),dict(other=1)]:
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:z.writestr('archive/data.pkl',pickle.dumps(value))
        with pytest.raises(ValueError):m.scheduler_seconds(buf.getvalue())


def test_cli_packages_only_aggregates_and_preserves_original(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.bundle import verify
    run=tmp_path/'pilot';fixture(run,True)
    monkeypatch.setattr(m,'ROOT',tmp_path)
    assert m.main(['--run','pilot','--output','metadata'])==0
    archive=tmp_path/'metadata.zip'
    assert verify(archive)['summary_state']=='PASS'
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={'summary.json','checkpoints.json','manifest.json'}
        assert not any(word in b''.join(z.read(n) for n in z.namelist()) for word in [b'PRIVATE',b'pickle',b'PARTIAL'])
    assert m.collect(run)['runs'][0]['checkpoint']['committed_tokens']==1000
    assert m.main(['--run','pilot','--output','pilot/metadata'])!=0


def test_automatic_failure_export_includes_checkpoint_accounting(tmp_path):
    import fresh_pilot_report as reports
    run=tmp_path/'pilot';fixture(run)
    files=reports.diagnostics(run,run/'diagnostics')
    checkpoint=[p for p,name in files if name.endswith('checkpoint-summary.json')]
    assert len(checkpoint)==1
    summary=json.loads(checkpoint[0].read_text())
    assert summary['runs'][0]['checkpoint']['original_status']=='FAIL'
    assert summary['runs'][0]['checkpoint']['updates']==2


def test_no_checkpoint_or_corruption_cannot_report_verified(tmp_path,monkeypatch):
    run=tmp_path/'pilot';run.mkdir()
    assert m.collect(run)['status']=='BLOCKED'
    _,_,folder,_=fixture(run);(folder/'weights.pt').write_bytes(b'bad')
    result=m.collect(run)
    assert result['status']=='BLOCKED' and result['runs'][0]['status']=='UNVERIFIED'
    assert set(result['runs'][0])=={'arm','command_path','status','error_type'}
    monkeypatch.setattr(m,'ROOT',tmp_path)
    assert m.main(['--run','pilot','--output','invalid'])!=0
