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


def test_search_overrun_retains_charged_attempt_evidence(monkeypatch):
    from oeis_learn.pipeline_validation import generalization_replay as replay
    clock=[0.]
    monkeypatch.setattr(replay.time,'monotonic',lambda:clock[0])
    def overrun(*args):
        clock[0]=61.
        raise ValueError('native deadline cleanup overrun')
    _,report=replay.search_round([task()],propose=overrun,client=NativeFixture(),checkpoint='checkpoint',seed=3)
    assert sum(row[1] for row in report['search_ledger'])==1


def test_interruption_persists_attempt_before_work_and_preserves_measured_time(monkeypatch):
    import pytest
    from oeis_learn.pipeline_validation import generalization_replay as replay
    clock=[0.];saved=[]
    monkeypatch.setattr(replay.time,'monotonic',lambda:clock[0])
    def interrupt(*args):
        assert saved[-1]['search_state']['pending']==['task',60.]
        assert saved[-1]['search_ledger']==[['task',1,0]]
        clock[0]=3.
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        replay.search_round([task()],propose=interrupt,client=NativeFixture(),checkpoint='checkpoint',seed=3,
                            save=lambda pool,report:saved.append(report))
    assert saved[-1]['search_ledger']==[['task',1,3.]]
    assert saved[-1]['status']=='INTERRUPTED'
    assert caught.value.replay_evidence==saved[-1]


def test_late_candidate_is_not_audited_or_admitted(monkeypatch):
    from oeis_learn.pipeline_validation import generalization_replay as replay
    clock=[0.];calls=[]
    monkeypatch.setattr(replay.time,'monotonic',lambda:clock[0])
    def late(*args):clock[0]=2.;return {}
    monkeypatch.setattr(replay,'admission',lambda *a,**kw:calls.append('audit'))
    pool,report=replay.search_round([task()],propose=late,client=NativeFixture(),checkpoint='checkpoint',seed=3,seconds=1)
    assert not calls and not pool.entries
    assert report['search_ledger']==[['task',1,2.]]
    assert report['search_state']['overruns']==[['task',2.,1.]]
    assert report['status']=='PARTIAL'


def test_failed_reservation_persistence_prevents_work():
    import pytest
    calls=[]
    def save(*args):raise OSError('disk full')
    with pytest.raises(OSError,match='disk full'):
        search_round([task()],propose=lambda *a:calls.append('work'),client=NativeFixture(),checkpoint='c',seed=1,save=save)
    assert calls==[]


def test_reservation_persistence_deadline_prevents_callback(monkeypatch):
    from oeis_learn.pipeline_validation import generalization_replay as replay
    clock=[0.];calls=[]
    monkeypatch.setattr(replay.time,'monotonic',lambda:clock[0])
    def save(*args):clock[0]=2.
    _,report=replay.search_round([task()],propose=lambda *a:calls.append('work'),client=NativeFixture(),checkpoint='c',seed=1,seconds=1,save=save)
    assert calls==[] and report['status']=='PARTIAL'
    assert report['search_ledger']==[['task',1,2.]]


def test_unexpected_failure_propagates_with_accounting(monkeypatch):
    import pytest
    from oeis_learn.pipeline_validation import generalization_replay as replay
    clock=[0.]
    monkeypatch.setattr(replay.time,'monotonic',lambda:clock[0])
    def fail(*args):clock[0]=5.;raise RuntimeError('native crash')
    with pytest.raises(RuntimeError,match='native crash') as caught:
        replay.search_round([task()],propose=fail,client=NativeFixture(),checkpoint='c',seed=1)
    assert caught.value.replay_evidence['status']=='FAIL'
    assert caught.value.replay_evidence['search_ledger']==[['task',1,5.]]


def test_fitted_literal_round_restores_original_and_transformed_sources():
    src='(program (return (int 0)))'
    candidate=dict(original_source='(program (return (int 9)))',
        transformed_source='(program (return (param.get p0)))',final_source=src,
        assignment={'p0':'0'},tokens=encode_body(src))
    pool,report=search_round([task()],propose=lambda *a:candidate,client=NativeFixture(),checkpoint='checkpoint',seed=7)
    restored=restore_pool(pool.snapshot(),tasks=[task()],client=NativeFixture(),deadline=time.monotonic()+60)
    assert restored.snapshot()==pool.snapshot() and report['admitted']==1
    assert restored.entries[0].record()['original_source']==candidate['original_source']
