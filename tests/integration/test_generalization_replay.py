import time
from oeis_learn.pipeline_validation.generalization_replay import search_round,restore_pool
from oeis_learn.decoder.bigint_codec import encode_body

class NativeFixture:
    def runtime_identity(self,profile):return 'fixture-runtime'
    def execute(self,source,indices,*,deadline):
        assert deadline>time.monotonic()
        return dict(outcome='ok',values=['0']*len(indices))


def task():
    return dict(id='task',role='training',ancestry=['mechanical','training'],source='(program (return (add (int 0) (int 0))))',indices=list(map(str,range(100))),values=['0']*100)


def test_native_replay_admission_and_restore():
    src='(program (return (int 0)))'
    def propose(visible,attempt,deadline):
        assert set(visible)=={'id','indices','values'} and len(visible['values'])==20
        return dict(original_source=src,final_source=src,assignment={},tokens=encode_body(src))
    pool,report=search_round([task()],propose=propose,client=NativeFixture(),checkpoint='checkpoint',seed=3)
    assert len(pool.entries)==1 and report['search_ledger'][0][1]==8
    restored=restore_pool(pool.snapshot(),tasks=[task()],client=NativeFixture(),deadline=time.monotonic()+60)
    assert restored.snapshot()==pool.snapshot()
    assert report['plan']['arms']==['continued_sft','verified_replay']


def test_empty_yield_is_terminal_without_weaker_verification():
    def propose(*args):raise ValueError('no_candidate')
    pool,report=search_round([task()],propose=propose,client=NativeFixture(),checkpoint='checkpoint',seed=3)
    assert not pool.entries and report['plan']['status']=='empty_yield'


def test_mixed_batch_persists_replay_cursor_before_interrupted_update():
    import copy
    import pytest
    from oeis_learn.pipeline_validation.generalization_replay import ReplayBatches
    from oeis_learn.pipeline_validation.generalization_training import train_updates
    from oeis_learn.generalization.budgets import Ledger
    src='(program (return (int 0)))'
    propose=lambda *args:dict(original_source=src,final_source=src,assignment={},tokens=encode_body(src))
    pool,_=search_round([task()],propose=propose,client=NativeFixture(),checkpoint='checkpoint',seed=3)
    base=[[dict(id=str(i),tokens=encode_body(src),values=['0']*20,indices=list(map(str,range(20)))) for i in range(8)]]
    state={'data':dict(cursor=0,pending_batch=None),'replay':pool.snapshot()}
    batches=ReplayBatches(base,pool,[task()],state)
    saved=[]
    def save(s,l):saved.append((copy.deepcopy(s),l))
    def interrupt(batch):raise InterruptedError()
    with pytest.raises(InterruptedError):train_updates(state,Ledger(1000),batches,update=interrupt,save=save)
    persisted,ledger=saved[-1]
    assert persisted['replay']['cursor']==2
    assert len(persisted['data']['pending_batch'])==8
    updated=[]
    result=train_updates(persisted,ledger,batches,update=lambda b:updated.append(b),save=save)
    assert len(updated)==1 and result['state']['replay']['cursor']==2
    assert sum(r['id'].startswith('replay:') for r in updated[0])==2

