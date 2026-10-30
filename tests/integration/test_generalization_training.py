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


@pytest.mark.parametrize("target_format",["literal","parameterized"])
def test_torch_worker_exact_resume_and_retained_milestones(tmp_path,monkeypatch,target_format):
    torch=pytest.importorskip('torch')
    pytest.importorskip('numpy')
    from oeis_learn.pipeline_validation.generalization_training import run_torch
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.pipeline_validation.identity import digest
    rows=[]
    for i in range(2):
        src=f'(program (return (int {i})))' if target_format=='literal' else '(program (return (param.get p0)))'
        rows.append(dict(id=str(i),role='training',source=src,tokens=encode_body(src),indices=list(map(str,range(100))),values=[str(i)]*100))
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
    config=dict(study='study',arm='static',seed=7,recipe=recipe,shards=shards,data_mode='static',batch_size=1,
                target_tokens=sum(len(r['tokens']) for r in rows)*2,milestones=[len(rows[0]['tokens'])],seconds=30,device='cpu')
    full=run_torch(tmp_path,tmp_path/'full',config)
    part=run_torch(tmp_path,tmp_path/'part',config,stop_after_updates=1)
    resumed=run_torch(tmp_path,tmp_path/'resumed',config,resume=(tmp_path/'part/checkpoints',part['checkpoint']))
    a=GeneralizationCheckpointStore(tmp_path/'full/checkpoints',full['compatibility'],config['target_tokens']).load(full['checkpoint'])
    b=GeneralizationCheckpointStore(tmp_path/'resumed/checkpoints',resumed['compatibility'],config['target_tokens']).load(resumed['checkpoint'])
    assert a['ledger']==b['ledger']
    for component in ('encoder','decoder'):
        for name,value in a['states']['weights'][component].items():
            assert torch.equal(value,b['states']['weights'][component][name])
    assert full['milestones']==resumed['milestones']
