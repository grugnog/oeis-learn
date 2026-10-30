import pytest
from oeis_learn.data.bigint_programs import BigintSampler
from oeis_learn.generalization.refresh import RefreshPlan
from oeis_learn.generalization.targets import pair

def test_fixed_support_and_numeric_law():
    sampler=BigintSampler(7)
    descriptor=sampler.descriptor(0)
    plan=RefreshPlan((descriptor,),seed=11,capacity=3)
    rows,cursor=plan.take(0,3)
    assert {r['family'] for r in rows}=={descriptor['family']} and cursor==3
    assert all(-16<=int(v)<=16 for r in rows for v in r['assignment'].values())
    assert plan.take(0,3)==(rows,3)
    with pytest.raises(ValueError): plan.take(3,1)
    with pytest.raises(ValueError): RefreshPlan((dict(descriptor,numeric_law='changed'),),seed=11,capacity=3)
    assert sampler.sample(0)==BigintSampler(7).sample(0)

def test_exact_target_pair_and_sharing():
    literal='(program (return (add (int 2) (int 2))))'
    explicit='(program (return (add (param.get p0) (param.get p0))))'
    p=pair(literal,explicit,{'p0':'2'},indices=('0','1'),sharing='explicit')
    assert p['disposition']=='supported' and p['parameter_count']==1
    with pytest.raises(ValueError): pair(literal,explicit,{'p0':'3'},indices=('0',),sharing='explicit')
    protected='(program (local $b0 bigint) (set $b0 (param.get p0)) (if (lt $index (get $b0)) (then (set $b0 (int 1))) (else (set $b0 (int 0)))) (return (get $b0)))'
    concrete=protected.replace('(param.get p0)','(int 2)')
    assert pair(concrete,protected,{'p0':'2'},indices=('0','1'),sharing='explicit')['disposition']=='unsupported_control_dependency'
    nonlinear='(program (return (mul (param.get p0) (param.get p0))))'
    assert pair('(program (return (mul (int 2) (int 2))))',nonlinear,{'p0':'2'},indices=('0',),sharing='explicit')['fit_class']=='nonlinear_bounded_smt'

def test_loop_carried_control_and_parameter_limit():
    source='(program (local $b0 bigint) (local $c0 control) (set $b0 (int 1)) (loop $c0 (control 2) (set $b0 (add (get $b0) (param.get p0))) (if (lt $index (get $b0)) (then (set $b0 (int 1))) (else (set $b0 (int 0))))) (return (get $b0)))'
    concrete=source.replace('(param.get p0)','(int 2)')
    assert pair(concrete,source,{'p0':'2'},indices=('0',),sharing='explicit')['disposition']=='unsupported_control_dependency'
    with pytest.raises(ValueError):
        pair('(program (return (int 1)))','(program (return (param.get p8)))',{'p8':'1'},indices=('0',),sharing='explicit')
