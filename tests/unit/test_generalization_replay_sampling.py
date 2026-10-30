import pytest
from fractions import Fraction as F
from oeis_learn.generalization.replay_policy import ReplayPool
from oeis_learn.generalization.replay_round import SearchLedger, round_plan
from oeis_learn.generalization.replay_admission import admit
from oeis_learn.decoder.bigint_codec import encode_body


def entry(task_id,source,order):
    task=dict(id=task_id,role='training',ancestry=('mechanical','training'),source='(program (return (add (add (int 0) (int 0)) (int 0))))',indices=tuple(map(str,range(100))),values=('0',)*100)
    return admit(task,original_source=source,final_source=source,assignment={},tokens=encode_body(source),origin_checkpoint='c',production=lambda s,i:('0',)*100,production_identity='runtime',order=order)

def test_task_first_cap_duplicate_and_restore():
    a=entry('a','(program (return (int 0)))',0)
    b=entry('b','(program (return (int 0)))',1)
    c=entry('a','(program (return (sub (int 0) (int 0))))',2)
    pool=ReplayPool(seed=3).add(a).add(b)
    assert pool.task_weights()=={'a':F(1,2),'b':F(1,2)}
    pool=pool.add(c)
    assert pool.task_weights()=={'a':F(1,2),'b':F(1,2)}
    assert len(pool.add(a).entries)==3
    samples,next_pool=pool.sample(8)
    assert len(samples)==2
    restored=ReplayPool.restore(next_pool.snapshot(),verify=lambda e:e)
    assert restored.sample(8)[0]==next_pool.sample(8)[0]
    assert ReplayPool(seed=3).sample(8)[0]==()
    with pytest.raises(ValueError): ReplayPool.restore(dict(next_pool.snapshot(),version='synthesis-replay/v1'),verify=lambda e:e)

def test_search_round_limits_and_empty_yield():
    ledger=SearchLedger()
    for i in range(8): ledger=ledger.charge('task',1)
    with pytest.raises(ValueError): ledger.charge('task',1)
    assert round_plan(0)['status']=='empty_yield'
    assert round_plan(1)['tokens_per_arm']==4_000_000
    assert round_plan(1)['milestones']==(1_000_000,2_000_000,4_000_000)
