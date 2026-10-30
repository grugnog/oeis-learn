from fractions import Fraction as F
import pytest
from oeis_learn.generalization.metrics import summarize, paired_difference, oracle_union
from oeis_learn.generalization.evidence import reconcile, checkpoint_roster

def test_fixed_denominators_macro_and_pairing():
    roster = {'a':'small','b':'large','c':'large','d':'large'}
    outcomes = {'a':True,'b':True,'c':False}
    s = summarize(roster, outcomes)
    assert (s.lower,s.upper,s.completion) == (F(1,2),F(3,4),F(3,4))
    assert (s.macro_lower,s.macro_upper) == (F(2,3),F(5,6))
    assert paired_difference(roster, outcomes, {'a':False,'b':True,'c':False,'d':True}) == (F(-1,4),F(0))
    union = oracle_union(roster, [outcomes, {'d':True}])
    assert union['scope'] == 'scorer-only coverage' and union['summary'].lower == F(3,4)
    with pytest.raises(ValueError): summarize(roster, {'alien':True})

def test_offline_identity_and_unavailable_weights():
    record = dict(study='s',source='code',roster={'a':'family'},outcomes={'a':True})
    assert reconcile([record], study='s', source='code', roster={'a':'family'}).lower == 1
    with pytest.raises(ValueError): reconcile([dict(record, source='other')], study='s',source='code',roster={'a':'family'})
    assert checkpoint_roster({'best':'hash1','latest':'hash2'}, {'hash1'}) == {'best':'available','latest':'unavailable'}

def test_seed_uncertainty_and_all_missing_family():
    from oeis_learn.generalization.metrics import seed_uncertainty
    s=summarize({'a':'small','b':'large','c':'large'}, {'a':True})
    assert s.macro_lower==F(1,2) and s.macro_upper==1
    ci=seed_uncertainty({1:(0,F(1,10)),2:(0,F(1,10)),3:(0,F(1,10))},draws=100)
    assert ci['lower']==ci['upper']==F(1,10) and ci['unit']=='paired_seed'
    with pytest.raises(ValueError): seed_uncertainty({1:(0,1)})
