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
    k.observe(chosen,FIX[case]['reward'],event_id='feedback-0',role='training')
    assert k.valid_updates==1
    assert k.probabilities()==pytest.approx([float(Fraction(x)) for x in FIX[case]['probabilities']],abs=1e-14)


def test_observed_zero_shares_missing_does_not_and_denominator_is_selected():
    from oeis_learn.generalization.curriculum import Exp3S
    k=Exp3S(['a','b','c'],seed=1)
    k.logweights=list(map(math.log,(.6,.3,.1)));k.valid_updates=2
    chosen=k.choose();k.observe(chosen,0,event_id='zero',role='training')
    assert k.weights()==pytest.approx((.5,.3125,.1875))
    old=copy.deepcopy(k.snapshot());chosen=k.choose()
    k.observe(chosen,None,event_id='missing',role='training')
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
    k.observe(chosen,1,event_id='ok',role='training')
    with pytest.raises(ValueError):k.observe(chosen,1,event_id='ok',role='training')
    restored=Exp3S.restore(json.loads(json.dumps(k.snapshot())))
    assert restored.choose()==k.choose()
    bad=k.snapshot();bad['version']='visit-scheduler/v1'
    with pytest.raises(ValueError):Exp3S.restore(bad)
    with pytest.raises(ValueError):Exp3S(['only'],seed=1)
    with pytest.raises(ValueError):Exp3S(['a','b'],seed=1,eta=float('nan'))
    k.logweights[0]=float('inf')
    with pytest.raises(ValueError):k.probabilities()
