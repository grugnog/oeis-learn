"""Real tiny-tensor tests of periodic persistence, not hardware throughput evidence."""
import json
from pathlib import Path
import pytest
from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
from oeis_learn.pipeline_validation import generalization_training as training


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');pytest.importorskip('numpy')
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    source='(program (return (int 1)))';tokens=encode_body(source)
    row=dict(id='one',role='training',source=source,tokens=tokens,indices=list(map(str,range(100))),values=['1']*100)
    (tmp_path/'rows.json').write_text(json.dumps([row]))
    def models(recipe,device):
        torch.manual_seed(recipe['model']['seed'])
        return torch.nn.Linear(1,1),torch.nn.Linear(1,1),{}
    calls=[];failure=[None]
    def loss(encoder,decoder,batch,device,**kw):
        optimizer=kw['optimizer'];optimizer.zero_grad()
        target=torch.tensor([[1.]])
        loss=(decoder(encoder(target))-target).square().mean();loss.backward();optimizer.step()
        calls.append(len(calls)+1)
        if len(calls)==failure[0]:raise InterruptedError('injected after parameter mutation')
    monkeypatch.setattr(bigint_worker,'models',models);monkeypatch.setattr(bigint_worker,'_batch_loss',loss)
    cfg=dict(study='study',arm='static',seed=7,recipe=dict(model={},learning=dict(learning_rate=.001)),
        shards=[reference(tmp_path/'rows.json',tmp_path)],data_mode='static',batch_size=1,target_tokens=32*len(tokens),
        milestones=[],seconds=60,device='cpu',checkpoint_policy='hourly/v1')
    return cfg,len(tokens),calls,failure


def loaded(root,result,cfg):
    return GeneralizationCheckpointStore(root/'checkpoints',result['compatibility'],cfg['target_tokens']).load(result['checkpoint'])


def same_weights(a,b):
    import torch
    for component in ('encoder','decoder'):
        for key,value in a['states']['weights'][component].items():assert torch.equal(value,b['states']['weights'][component][key])


def test_hourly_avoids_intermediate_tensor_saves_but_keeps_progress(tmp_path,fixture):
    cfg,cost,_,_=fixture
    hourly=training.run_torch(tmp_path,tmp_path/'hourly',cfg)
    dense=training.run_torch(tmp_path,tmp_path/'dense',dict(cfg,checkpoint_policy='per-update/v1'))
    assert hourly['status']==dense['status']=='PASS'
    assert hourly['tokens']==hourly['charged_tokens']==32*cost and hourly['updates']==32
    assert hourly['timing']['checkpoint_calls']==2 and dense['timing']['checkpoint_calls']==65
    assert hourly['timing']['journal_writes']>=65 and len(hourly['timing']['windows'])==3
    assert hourly['timing']['progress']['recoverable_tokens']==32*cost
    same_weights(loaded(tmp_path/'hourly',hourly,cfg),loaded(tmp_path/'dense',dense,cfg))
    journal=json.loads((tmp_path/'hourly/checkpoints/generalization-progress.json').read_text())
    assert journal['closed'] and journal['checkpoint']==hourly['checkpoint']


def test_crash_rolls_back_weights_but_never_refunds_spent_exposure(tmp_path,fixture):
    cfg,cost,calls,failure=fixture;failure[0]=5
    with pytest.raises(InterruptedError):training.run_torch(tmp_path,tmp_path/'crash',cfg)
    prior=tmp_path/'crash/checkpoints';pointer=json.loads((prior/'generalization-latest.json').read_text())
    record=json.loads((prior/'generalization-progress.json').read_text())
    assert record['ledger']['consumed']==5*cost and record['ledger']['updates']==4 and not record['closed']
    progress=json.loads((tmp_path/'crash/timing.json').read_text())['progress']
    assert progress['tokens']==4*cost and progress['recoverable_tokens']==0
    originals={str(p):p.read_bytes() for p in prior.rglob('*') if p.is_file()}
    failure[0]=None
    resumed=training.run_torch(tmp_path,tmp_path/'resumed',cfg,resume=(prior,pointer),resume_elapsed=1)
    assert resumed['status']=='PARTIAL' and resumed['reason']=='RECOVERED_WITH_DISCARDED_EXPOSURE' and resumed['discarded_tokens']==5*cost
    assert resumed['tokens']==27*cost and resumed['charged_tokens']==32*cost and resumed['updates']==27
    expected=training.run_torch(tmp_path,tmp_path/'expected',cfg,stop_after_updates=27)
    same_weights(loaded(tmp_path/'resumed',resumed,cfg),loaded(tmp_path/'expected',expected,cfg))
    assert originals=={str(p):p.read_bytes() for p in prior.rglob('*') if p.is_file()}


def test_unclean_without_supervisor_does_not_guess_free_wall_time(tmp_path,fixture):
    cfg,cost,calls,failure=fixture;failure[0]=3
    with pytest.raises(InterruptedError):training.run_torch(tmp_path,tmp_path/'crash',cfg)
    prior=tmp_path/'crash/checkpoints';pointer=json.loads((prior/'generalization-latest.json').read_text())
    count=len(calls);failure[0]=None
    result=training.run_torch(tmp_path,tmp_path/'expired',cfg,resume=(prior,pointer))
    assert result['status']=='PARTIAL' and len(calls)==count
    assert result['learner_seconds']>=cfg['seconds'] and result['charged_tokens']==3*cost and result['tokens']==0


def test_clean_stop_resumes_exactly_and_preserves_milestone_pins(tmp_path,fixture):
    cfg,cost,_,_=fixture;cfg['milestones']=[3*cost+1,16*cost]
    full=training.run_torch(tmp_path,tmp_path/'full',cfg)
    part=training.run_torch(tmp_path,tmp_path/'part',cfg,stop_after_updates=5)
    resumed=training.run_torch(tmp_path,tmp_path/'resumed',cfg,resume=(tmp_path/'part/checkpoints',part['checkpoint']))
    assert resumed['discarded_tokens']==0 and full['milestones']==resumed['milestones']
    assert full['milestones']=={str(3*cost+1):3*cost,str(16*cost):16*cost}
    same_weights(loaded(tmp_path/'full',full,cfg),loaded(tmp_path/'resumed',resumed,cfg))


def test_missing_journal_or_older_point_fails_closed(tmp_path,fixture):
    cfg,_,_,_=fixture;r=training.run_torch(tmp_path,tmp_path/'part',cfg,stop_after_updates=3)
    prior=tmp_path/'part/checkpoints';roster=json.loads((prior/'retained.json').read_text())
    old=next(c['state'] for c in roster['checkpoints'] if 'initial' in c['pins'])
    with pytest.raises(ValueError,match='latest'):training.run_torch(tmp_path,tmp_path/'old',cfg,resume=(prior,old))
    (prior/'generalization-progress.json').unlink()
    with pytest.raises(ValueError,match='journal'):training.run_torch(tmp_path,tmp_path/'missing',cfg,resume=(prior,r['checkpoint']))


def test_periodic_deadline_forces_an_intermediate_full_snapshot(tmp_path,fixture,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_persistence
    cfg,_,_,_=fixture
    # Zero test interval exercises the same due boundary without waiting an hour.
    monkeypatch.setattr(generalization_persistence,'INTERVAL',0)
    result=training.run_torch(tmp_path,tmp_path/'due',cfg)
    assert result['timing']['checkpoint_calls']==33


def test_interrupted_journal_after_new_snapshot_does_not_discard_saved_work(tmp_path,fixture,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_persistence
    from oeis_learn.fresh import common
    cfg,_,_,_=fixture
    monkeypatch.setattr(generalization_persistence,'INTERVAL',0)
    original=common.write
    def fail_journal(path,value,**kwargs):
        if Path(path).name=='generalization-progress.json' and value['ledger']['updates']==2:
            raise OSError('journal publication interrupted')
        return original(path,value,**kwargs)
    monkeypatch.setattr(common,'write',fail_journal)
    with pytest.raises(OSError):training.run_torch(tmp_path,tmp_path/'crash',cfg)
    monkeypatch.setattr(common,'write',original)
    prior=tmp_path/'crash/checkpoints';pointer=json.loads((prior/'generalization-latest.json').read_text())
    resumed=training.run_torch(tmp_path,tmp_path/'resumed',cfg,resume=(prior,pointer),resume_elapsed=1)
    full=training.run_torch(tmp_path,tmp_path/'full',cfg)
    assert resumed['status']=='PASS' and resumed['discarded_tokens']==0 and resumed['updates']==32
    same_weights(loaded(tmp_path/'resumed',resumed,cfg),loaded(tmp_path/'full',full,cfg))


def test_selection_metadata_copy_preserves_closed_hourly_recovery_accounting(tmp_path,fixture):
    from oeis_learn.pipeline_validation.identity import seal,digest
    cfg,_,_,_=fixture;r=training.run_torch(tmp_path,tmp_path/'part',cfg,stop_after_updates=3)
    prior=tmp_path/'part/checkpoints';roster=json.loads((prior/'retained.json').read_text())
    decision=seal(dict(version='generalization-selected-pins/v1',roster=roster['identity'],decision=digest('selection'),
        selected=roster['latest'],best_loss=roster['latest'],reason='selected'))
    store=GeneralizationCheckpointStore(prior,r['compatibility'],cfg['target_tokens'])
    pointer=store.selected_copy(tmp_path/'selected',roster,decision)
    resumed=training.run_torch(tmp_path,tmp_path/'resumed',cfg,resume=(tmp_path/'selected',pointer))
    assert resumed['status']=='PASS' and resumed['discarded_tokens']==0
    journal=json.loads((prior/'generalization-progress.json').read_text());journal=seal(dict(journal,closed=False))
    (prior/'generalization-progress.json').write_text(json.dumps(journal))
    with pytest.raises(ValueError,match='closed'):store.selected_copy(tmp_path/'unsafe',roster,decision)
