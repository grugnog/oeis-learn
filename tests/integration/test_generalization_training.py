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
