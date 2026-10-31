import json
import pytest
from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
from oeis_learn.generalization.budgets import Ledger
from oeis_learn.generalization.state import STATE_FIELDS
from oeis_learn.pipeline_validation.generalization_training import train_updates


def states():
    s={key:{} for key in STATE_FIELDS}
    s['rng']={key:[] for key in ('python','numpy','torch','device')}
    s['data']=dict(cursor=0,pending_batch=None)
    return s


def test_atomic_full_state_and_tampering(tmp_path):
    store=GeneralizationCheckpointStore(tmp_path,'contract',100)
    s=states();s['weights']={'fixture':[1,2]}
    ref=store.save(s,Ledger(100),encode=lambda x:json.dumps(x).encode())
    loaded=store.load(ref,decode=json.loads)
    assert loaded['states']==s and loaded['ledger']==Ledger(100)
    (tmp_path/ref['path']).write_text('{}')
    with pytest.raises(ValueError):store.load(ref,decode=json.loads)


def test_pending_batch_survives_interruption_and_budget_boundary(tmp_path):
    batches=[[dict(id='a',tokens=[9,2],values=['0']*20)],[dict(id='b',tokens=[8,7,2],values=['1']*20)]]
    s=states();updates=[];saved=[]
    def save(state,ledger): saved.append((json.loads(json.dumps(state)),ledger))
    def update(batch): updates.append(batch[0]['id']);raise InterruptedError()
    with pytest.raises(InterruptedError):train_updates(s,Ledger(4),batches,update=update,save=save)
    state,ledger=saved[-1]
    assert ledger.pending and ledger.consumed==2
    result=train_updates(state,ledger,batches,update=lambda batch:updates.append(batch[0]['id']),save=save)
    assert result['ledger'].consumed==2 and result['ledger'].updates==1
    assert result['ledger'].pending[1]==3 and result['status']=='COMPLETE'
    assert updates==['a','a']


def test_prepared_static_and_refresh_cursor_and_targets(tmp_path):
    from oeis_learn.pipeline_validation.generalization_training import PreparedBatches
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.decoder.bigint_codec import encode_body
    rows=[]
    for i in range(4):
        source=f'(program (return (int {i})))'
        rows.append(dict(id=str(i),role='training',source=source,tokens=encode_body(source),
                         indices=list(map(str,range(100))),values=[str(i)]*100))
    (tmp_path/'rows.json').write_text(json.dumps(rows))
    ref=reference(tmp_path/'rows.json',tmp_path)
    static=PreparedBatches(tmp_path,[ref],batch_size=2,mode='static',updates=4)
    refresh=PreparedBatches(tmp_path,[ref],batch_size=2,mode='refresh',updates=4)
    assert [r['id'] for r in static[2]]==['0','1']
    assert len(refresh)==2 and len(static)==4
    assert all(len(r['values'])==20 for r in refresh[1])
    assert static[0]==PreparedBatches(tmp_path,[ref],batch_size=2,mode='static',updates=4)[0]
    with pytest.raises(IndexError):refresh[2]
    rows[0]['role']='assessment';(tmp_path/'rows.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError):PreparedBatches(tmp_path,[reference(tmp_path/'rows.json',tmp_path)],batch_size=2,mode='static',updates=4)


def test_checkpoint_pins_survive_pruning_and_failed_save(tmp_path):
    store=GeneralizationCheckpointStore(tmp_path,'contract',100)
    encode=lambda x:json.dumps(x).encode()
    first=store.save(states(),Ledger(100),encode=encode)
    second=store.save(states(),Ledger(100),encode=encode)
    store.retain({'initial':first,'latest':second},('initial','latest'))
    third=store.save(states(),Ledger(100),encode=encode)
    store.retain({'initial':first,'old':second,'latest':third},('initial','latest'))
    assert (tmp_path/first['path']).is_file() and (tmp_path/third['path']).is_file()
    assert not (tmp_path/second['path']).exists()
    def fail(value):raise OSError('injected disk failure')
    with pytest.raises(OSError):store.save(states(),Ledger(100),encode=fail)
    assert json.loads((tmp_path/'generalization-latest.json').read_text())==third


def test_descriptor_uses_committed_exposure_without_deserialization(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.identity import digest
    store=GeneralizationCheckpointStore(tmp_path,digest('contract'),100)
    state=states();state['data']['pending_batch']=[{'tokens':[3,2]}]
    ref=store.save(state,Ledger(100).reserve('batch',2),encode=lambda x:json.dumps(x).encode())
    monkeypatch.setattr(store,'_decode',lambda _:pytest.fail('descriptor must not deserialize private state'))
    descriptor=store.describe(ref,initialization=digest('init'),pins=['latest'])
    assert descriptor['tokens']==0 and descriptor['updates']==0
    assert descriptor['weights']==json.loads((tmp_path/ref['path']).read_text())['states']['weights']
    (tmp_path/descriptor['weights']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):store.describe(ref,initialization=digest('init'),pins=['latest'])


@pytest.mark.parametrize("data_mode",["static","refresh"])
@pytest.mark.parametrize("target_format",["literal","parameterized"])
def test_torch_worker_exact_resume_and_retained_milestones(tmp_path,monkeypatch,target_format,data_mode,replay_mode=False):
    torch=pytest.importorskip('torch')
    pytest.importorskip('numpy')
    from oeis_learn.pipeline_validation.generalization_training import run_torch
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.pipeline_validation.identity import digest
    rows=[]
    for i in range(8 if replay_mode else 4 if data_mode=="refresh" else 2):
        src=f'(program (return (int {i})))' if target_format=='literal' else '(program (return (param.get p0)))'
        if replay_mode and i==0:src='(program (return (add (int 0) (int 0))))'
        rows.append(dict(id='task' if replay_mode and i==0 else str(i),role='training',source=src,tokens=encode_body(src),indices=list(map(str,range(100))),values=[str(i)]*100))
    (tmp_path/'rows.json').write_text(json.dumps(rows));shards=[reference(tmp_path/'rows.json',tmp_path)]
    # Tiny real tensor modules isolate persistence equivalence from model scale.
    def models(recipe,device):
        torch.manual_seed(recipe['model']['seed'])
        return torch.nn.Linear(1,1).to(device),torch.nn.Linear(1,1).to(device),{}
    def loss(encoder,decoder,batch,device,**kw):
        optimizer=kw['optimizer'];optimizer.zero_grad()
        value=torch.tensor([[float(r['values'][0])] for r in batch],device=device)
        result=(decoder(encoder(value))-value).square().mean();result.backward();optimizer.step()
    monkeypatch.setattr(bigint_worker,'models',models);monkeypatch.setattr(bigint_worker,'_batch_loss',loss)
    recipe=dict(model=dict(seed=7),learning=dict(learning_rate=.001),inputs={})
    config=dict(study='study',arm='static',seed=7,recipe=recipe,shards=shards,data_mode=data_mode,batch_size=1,
                target_tokens=sum(len(r['tokens']) for r in rows)*(1 if data_mode=='refresh' else 2),milestones=[len(rows[0]['tokens'])],seconds=30,device='cpu')
    extra={}
    if replay_mode:
        import time
        from tests.integration.test_generalization_replay import NativeFixture,task
        from oeis_learn.pipeline_validation.generalization_replay import search_round
        source='(program (return (int 0)))'
        pool,_=search_round([task()],propose=lambda *a:dict(original_source=source,final_source=source,assignment={},tokens=encode_body(source)),
            client=NativeFixture(),checkpoint='checkpoint',seed=7)
        config['batch_size']=4
        first,second,_=models(recipe,'cpu')
        torch.save(dict(encoder=first.state_dict(),decoder=second.state_dict()),tmp_path/'initializer.pt')
        extra=dict(initializer=reference(tmp_path/'initializer.pt',tmp_path),replay=dict(pool=pool,tasks=[task()],client=NativeFixture()))
    full=run_torch(tmp_path,tmp_path/'full',config,**extra)
    part=run_torch(tmp_path,tmp_path/'part',config,stop_after_updates=1,**extra)
    resumed=run_torch(tmp_path,tmp_path/'resumed',config,resume=(tmp_path/'part/checkpoints',part['checkpoint']),**extra)
    a=GeneralizationCheckpointStore(tmp_path/'full/checkpoints',full['compatibility'],config['target_tokens']).load(full['checkpoint'])
    b=GeneralizationCheckpointStore(tmp_path/'resumed/checkpoints',resumed['compatibility'],config['target_tokens']).load(resumed['checkpoint'])
    assert a['ledger']==b['ledger']
    for component in ('encoder','decoder'):
        for name,value in a['states']['weights'][component].items():
            assert torch.equal(value,b['states']['weights'][component][name])
    assert full['milestones']==resumed['milestones']
    if not replay_mode and data_mode=='static' and target_format=='literal':
        from oeis_learn.pipeline_validation.identity import seal
        roster=json.loads((tmp_path/'part/checkpoints/retained.json').read_text())
        candidates=roster['checkpoints'];selected=roster['latest']
        best=next(c['identity'] for c in candidates if 'initial' in c['pins'])
        decision=seal(dict(version='generalization-selected-pins/v1',roster=roster['identity'],
            decision=digest('selection decision'),selected=selected,best_loss=best,reason='selected'))
        pointer=GeneralizationCheckpointStore(tmp_path/'part/checkpoints',part['compatibility'],config['target_tokens']).selected_copy(tmp_path/'selected',roster,decision)
        continued=run_torch(tmp_path,tmp_path/'continued',config,resume=(tmp_path/'selected',pointer))
        restored=GeneralizationCheckpointStore(tmp_path/'continued/checkpoints',continued['compatibility'],config['target_tokens']).load(continued['checkpoint'])
        assert restored['states']['selection']['checkpoint']==selected
        for member in ('encoder','decoder'):
            for key,value in a['states']['weights'][member].items():
                assert torch.equal(value,restored['states']['weights'][member][key])
        kept=json.loads((tmp_path/'continued/checkpoints/retained.json').read_text())
        assert {best,selected}<={c['identity'] for c in kept['checkpoints']}
        assert kept['best_loss']==best and kept['selected']==selected
    for name,result in [('full',full),('resumed',resumed)]:
        roster=json.loads((tmp_path/name/'checkpoints/retained.json').read_text())
        assert result['retained']==roster['identity']
        assert roster['best_loss'] is None and roster['selected'] is None
        assert len([c for c in roster['checkpoints'] if 'latest' in c['pins']])==1
        for candidate in roster['checkpoints']:
            weights=torch.load(tmp_path/name/'checkpoints'/candidate['weights']['path'],weights_only=True)
            assert set(weights)=={'encoder','decoder'}
        assert any('initial' in c['pins'] for c in roster['checkpoints'])


def test_replay_initialized_torch_resume(tmp_path,monkeypatch):
    test_torch_worker_exact_resume_and_retained_milestones(tmp_path,monkeypatch,'literal','static',replay_mode=True)


def test_selection_pins_are_persisted_without_mounting_supervision(tmp_path):
    from oeis_learn.pipeline_validation.identity import digest,seal
    store=GeneralizationCheckpointStore(tmp_path/'original',digest('contract'),100)
    encode=lambda value:json.dumps(value).encode()
    state=states();state['history']={'exposures':{},'milestones':{}}
    first=store.save(state,Ledger(100),encode=encode)
    second=store.save(state,Ledger(100,consumed=2,updates=1),encode=encode)
    descriptors=[store.describe(ref,initialization=digest('random'),pins=[label]) for ref,label in [(first,'initial'),(second,'latest')]]
    roster=seal(dict(version='generalization-retained/v1',checkpoints=descriptors))
    decision=seal(dict(version='generalization-selected-pins/v1',roster=roster['identity'],decision=digest('selection'),
        best_loss=descriptors[0]['identity'],selected=descriptors[1]['identity'],reason='selected'))
    pointer=store.selected_copy(tmp_path/'selected',roster,decision,decode=json.loads,encode=encode)
    restored=GeneralizationCheckpointStore(tmp_path/'selected',digest('contract'),100).load(pointer,decode=json.loads)
    assert restored['states']['selection']['checkpoint']==descriptors[1]['identity']
    assert restored['states']['selection']['best_loss']==descriptors[0]['identity']
    assert set(restored['states']['pins'])=={first['sha256'],second['sha256']}
    assert restored['ledger']==Ledger(100,consumed=2,updates=1)
    changed=seal(dict(decision,selected=digest('unknown')))
    with pytest.raises(ValueError,match='outside'):
        store.selected_copy(tmp_path/'forbidden',roster,changed,decode=json.loads,encode=encode)
    assert not (tmp_path/'forbidden').exists()


def test_model_storage_reservation_blocks_before_any_update(tmp_path,monkeypatch):
    import torch
    from oeis_learn.pipeline_validation.generalization_training import storage_preflight
    from oeis_learn.fresh import bigint_worker
    calls=[]
    def models(recipe,device):
        calls.append(device);return torch.nn.Linear(2,2),torch.nn.Linear(2,1),{}
    monkeypatch.setattr(bigint_worker,'models',models)
    study=dict(checkpoints=dict(milestones=[10,20]),seeds=[7])
    runtime=dict(training={'a':{},'b':{}},replay=None,storage=dict(checkpoint_bytes=32<<20))
    result=storage_preflight({},study,runtime)
    assert result['parameter_bytes']==9*4 and result['pinned_generations_reserved']==12
    assert calls==['cpu']
    runtime['storage']['checkpoint_bytes']=result['required_checkpoint_bytes']-1
    with pytest.raises(ValueError,match='required pins'):storage_preflight({},study,runtime)
