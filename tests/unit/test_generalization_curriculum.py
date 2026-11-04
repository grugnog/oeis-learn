"""Independent Graves equation and deterministic-stream counterexamples."""
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import pytest

FIX=json.loads(Path('tests/fixtures/generalization/pac-policy.json').read_text())['kernel']

@pytest.mark.parametrize('case',['positive','negative'])
def test_graves_first_share_fraction(case):
    from oeis_learn.generalization.curriculum import Exp3S
    k=Exp3S(['a','b','c'],seed=1,eta=math.log(2)/3)
    chosen=k.choose();assert chosen['family']=='a' and chosen['probability']==pytest.approx(1/3)
    k.observe(chosen,FIX[case]['reward'],event_id=f"feedback/{chosen['action']}",role='training')
    assert k.valid_updates==1
    assert k.probabilities()==pytest.approx([float(Fraction(x)) for x in FIX[case]['probabilities']],abs=1e-14)


def test_observed_zero_shares_missing_does_not_and_denominator_is_selected():
    from oeis_learn.generalization.curriculum import Exp3S
    k=Exp3S(['a','b','c'],seed=1)
    k.logweights=list(map(math.log,(.6,.3,.1)));k.valid_updates=2
    chosen=k.choose();k.observe(chosen,0,event_id=f"feedback/{chosen['action']}",role='training')
    assert k.weights()==pytest.approx((.5,.3125,.1875))
    old=copy.deepcopy(k.snapshot());chosen=k.choose()
    k.observe(chosen,None,event_id=f"feedback/{chosen['action']}",role='training')
    assert k.logweights==old['logweights'] and k.valid_updates==3
    chosen=k.choose();forged=dict(chosen,probability=.99)
    before=copy.deepcopy(k.snapshot())
    with pytest.raises(ValueError):k.observe(forged,1,event_id='bad',role='training')
    assert k.snapshot()==before


def test_kernel_extremes_roles_duplicates_and_restore():
    from oeis_learn.generalization.curriculum import Exp3S
    k=Exp3S([str(i) for i in range(32)],seed=4)
    k.logweights=[-1e300]*31+[0.]
    p=k.probabilities();assert min(p)==1/128 and max(p)==97/128
    assert sum(p)==pytest.approx(1)
    chosen=k.choose()
    for role in ('selection','assessment','monitoring','review'):
        with pytest.raises(ValueError):k.observe(chosen,1,event_id='bad',role=role)
    for value in (float('nan'),float('inf'),2,-2):
        with pytest.raises(ValueError):k.observe(chosen,value,event_id='bad',role='training')
    k.observe(chosen,1,event_id=f"feedback/{chosen['action']}",role='training')
    with pytest.raises(ValueError):k.observe(chosen,1,event_id=f"feedback/{chosen['action']}",role='training')
    restored=Exp3S.restore(json.loads(json.dumps(k.snapshot())))
    assert restored.choose()==k.choose()
    with pytest.raises(ValueError):k.observe(k.pending,1,event_id='feedback/1',role='training')
    bad=k.snapshot();bad['version']='visit-scheduler/v1'
    with pytest.raises(ValueError):Exp3S.restore(bad)
    with pytest.raises(ValueError):Exp3S(['only'],seed=1)
    with pytest.raises(ValueError):Exp3S(['a','b'],seed=1,eta=float('nan'))
    k.logweights[0]=float('inf')
    with pytest.raises(ValueError):k.probabilities()


def test_past_only_signed_scaler_quantiles_and_reservoir_continuation():
    from oeis_learn.generalization.curriculum import RewardScaler
    s=RewardScaler(seed=7)
    for x in (-2,-1,0,1,2):
        r=s.observe(x,1)
        assert r['reward']==0 and r['status']=='warmup' and r['raw_gain']==x
    r=s.observe(-100,2)
    assert r['raw_gain']==-100 and r['work_gain']==-50
    assert r['q20']==pytest.approx(-1.2) and r['q80']==pytest.approx(1.2) and r['reward']==-1
    tied=RewardScaler(seed=2)
    for _ in range(6):r=tied.observe(1,10)
    assert r['status']=='degenerate' and r['reward']==0
    for i in range(5000):s.observe(i-2500,17)
    assert len(s.values)==4096 and s.seen==5006
    restored=RewardScaler.restore(json.loads(json.dumps(s.snapshot())))
    for i in range(40):assert restored.observe(i,3)==s.observe(i,3)
    assert restored.snapshot()==s.snapshot()
    old=s.snapshot()
    for gain,tokens in [(1,0),(float('nan'),1),(float('inf'),1),(1,True)]:
        with pytest.raises(ValueError):s.observe(gain,tokens)
        assert s.snapshot()==old
    bad=s.snapshot();bad['seen']=0
    with pytest.raises(ValueError):RewardScaler.restore(bad)


def test_independent_family_streams_wrap_skip_and_probe_rotation():
    from oeis_learn.generalization.curriculum import FamilyStream,ProbeRotation
    a=FamilyStream('a',list('abcde'),seed=17)
    b=FamilyStream('b',list('vwxyz'),seed=17)
    a0=FamilyStream('a',list('abcde'),seed=17)
    assert set(a.take(5))==set('abcde')
    assert a0.take(5)==FamilyStream('a',list('abcde'),seed=17).take(5)
    restored=FamilyStream.restore(json.loads(json.dumps(a.snapshot())),family='a',rows=list('abcde'))
    for _ in range(5):
        b.take(4)
        assert restored.take(3,exclude={'a'})==a.take(3,exclude={'a'})
    assert a.cursor>0 or a.epoch>0
    assert len(set(a.take(4,exclude={'a'})))==4
    with pytest.raises(ValueError):a.take(5,exclude={'a'})
    with pytest.raises(ValueError):FamilyStream.restore(a.snapshot(),family='b',rows=list('abcde'))
    p=ProbeRotation(list('abcdefgh'))
    assert [p.take() for _ in range(5)]==[['a','b'],['c','d'],['e','f'],['g','h'],['a','b']]
    assert ProbeRotation.restore(p.snapshot(),probes=list('abcdefgh')).take()==['c','d']
