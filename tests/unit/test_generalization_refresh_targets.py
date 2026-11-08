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

def test_paired_grounding_differential_against_integer_formulae():
    for a in (-16,-1,0,2,16):
        for b in (-16,0,16):
            parameter='(program (return (add (mul (param.get p0) $index) (param.get p1))))'
            concrete=f'(program (return (add (mul (int {a}) $index) (int {b}))))'
            result=pair(concrete,parameter,{'p0':str(a),'p1':str(b)},indices=tuple(map(str,range(100))),sharing='explicit')
            assert result['verification']['values']==[str(a*i+b) for i in range(100)]


def test_pac_whole_row_fallback_does_not_drop_or_partially_parameterize():
    from oeis_learn.generalization.targets import paired_learning_view
    guard='(program (return (select (lt $index (int 2)) (int 5) (int 7))))'
    parameter='(program (return (select (lt $index (param.get p0)) (param.get p1) (param.get p2))))'
    result=paired_learning_view(guard,parameter,{'p0':'2','p1':'5','p2':'7'},indices=('0','2'),sharing='explicit')
    assert result['target_source']==guard and result['disposition']=='unsupported_control_dependency'
    assert result['verification']['values']==['5','7']
    # Nine distinct occurrences exceed the fitter/codec limit; retain the entire
    # independently valid original row, never silently drop it or free eight only.
    expr='(param.get p8)'
    for i in reversed(range(8)):expr=f'(add (param.get p{i}) {expr})'
    source=f'(program (return {expr}))';literal=source
    for i in range(9):literal=literal.replace(f'(param.get p{i})',f'(int {i+1})')
    result=paired_learning_view(literal,source,{f'p{i}':str(i+1) for i in range(9)},indices=('0','1'),sharing='distinct-occurrences')
    assert result['target_source']==literal and result['disposition']=='unsupported_parameter_count'
    assert result['verification']['values']==['45','45']
    with pytest.raises(ValueError):paired_learning_view(literal,source,{'p0':'1'},indices=('0',),sharing='explicit')


def test_pac_repeated_sharing_and_grounding_are_not_silently_fallback():
    from oeis_learn.generalization.targets import paired_learning_view
    literal='(program (return (add (int 2) (int 2))))'
    parameter='(program (return (add (param.get p0) (param.get p0))))'
    got=paired_learning_view(literal,parameter,{'p0':'2'},indices=('0','1'),sharing='explicit')
    assert got['target_source']==parameter and got['disposition']=='supported'
    with pytest.raises(ValueError):paired_learning_view(literal,parameter,{'p0':'3'},indices=('0',),sharing='explicit')
    with pytest.raises(ValueError):paired_learning_view(literal,parameter,{'p0':'2'},indices=('0',),sharing='distinct-occurrences')
