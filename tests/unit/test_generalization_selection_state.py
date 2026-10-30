from fractions import Fraction as F
import pytest
from oeis_learn.generalization.selection import select
from oeis_learn.generalization.checkpoints import pin_plan
from oeis_learn.generalization.experiments import promotion, next_stage
from oeis_learn.generalization.state import snapshot, restore, STATE_FIELDS
from oeis_learn.generalization.budgets import Ledger

def candidate(id,tokens,unresolved=0):
    roster={str(i):'f' for i in range(20)}
    outcomes={str(i):True for i in range(20-unresolved)}
    return dict(id=id,tokens=tokens,role='selection',structural=(roster,outcomes),parameter=(roster,outcomes))

def test_selection_ties_and_ineligibility():
    assert select([candidate('b',8),candidate('a',4)]).checkpoint_identity=='a'
    assert select([candidate('b',8,2)]).checkpoint_identity is None
    with pytest.raises(ValueError): select([dict(candidate('a',4),role='assessment')])
    assert pin_plan({'a':4,'b':8,'c':9},milestones=(4,8),best='a',latest='c',selected='b')==('a','b','c')
    with pytest.raises(ValueError): pin_plan({'a':4},milestones=(4,8),best='a',latest='a',selected=None,through=8)

def test_complete_state_and_nonreplenished_pending_batch():
    parts={k:{'fixture':k} for k in STATE_FIELDS}
    parts['rng']={k:[] for k in ('python','numpy','torch','device')}
    parts['data']={'cursor':3,'pending_batch':['row3']}
    ledger=Ledger(10).reserve('b',4)
    s=snapshot(parts,contract='contract',ledger=ledger)
    a=restore(s,contract='contract',cap=10)
    assert a['ledger']==ledger and a['states']['data']['pending_batch']==['row3']
    with pytest.raises(ValueError): restore(s,contract='other',cap=10)
    with pytest.raises(ValueError): snapshot({'weights':{}},contract='contract',ledger=ledger)
    with pytest.raises(ValueError): restore(dict(s,kind='weights_only'),contract='contract',cap=10)

def test_promotion_bounds_and_assessment_firewall():
    rows=[dict(seed=i,structural=(F(6,100),F(8,100)),parameter=(F(-2,100),F(0)),completion=F(1)) for i in range(3)]
    assert promotion(rows)=='promote'
    rows[0]['structural']=(F(-1,100),F(1,100))
    assert promotion(rows)=='inconclusive'
    with pytest.raises(ValueError): next_stage('P1-screen',valid=True,selection_completion=F(1),feedback='assessment')
    assert next_stage('P1-screen',valid=True,selection_completion=F(1),feedback='selection')=='P1-confirm'
