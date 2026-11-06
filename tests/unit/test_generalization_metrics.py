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


def pac_roster():
    roster={}
    # Unequal certified family sizes: 1 + 3 + 16*4 = 68. Ambiguous 6+6+7+7;
    # unresolved 6+7+7+7+7. This is an independent denominator fixture.
    for category,sizes in [('certified',[1,3]+[4]*16),('ambiguous',[6,6,7,7]),('unresolved',[6,7,7,7,7])]:
        for f,n in enumerate(sizes):
            for i in range(n):roster[f'{category}/{f}/{i}']=dict(family=f'{category}/{f}',category=category,stratum='structural')
    return roster


def test_pac_equal_task_totals_have_different_fixed_family_macro_rates():
    from oeis_learn.generalization.metrics import pac_structural_metrics,pac_paired_counts
    roster=pac_roster();a=dict.fromkeys(roster,False);b=dict(a)
    a['certified/0/0']=True;b['certified/2/0']=True
    sa=pac_structural_metrics(roster,a);sb=pac_structural_metrics(roster,b)
    assert sa['certified']['successes']==sb['certified']['successes']==1
    assert sa['certified']['family_macro_lower']=='1/18'
    assert sb['certified']['family_macro_lower']=='1/72'
    counts=pac_paired_counts(roster,a,b)
    assert counts==dict(tasks=68,joint_complete=68,both=0,baseline_only=1,treatment_only=1,neither=66,censored_pairs=0)
    b['certified/0/0']=None
    assert pac_paired_counts(roster,a,b)['censored_pairs']==1
    assert pac_structural_metrics(roster,b)['unresolved']['tasks']==34
    assert pac_structural_metrics(roster,b)['certified']['censored']==1


def test_pac_completion_boundary_missing_arms_and_matched_attribution():
    from oeis_learn.generalization.reconciliation import pac_comparison_report
    from tests.unit.test_generalization_execution import pac_setup
    roster=pac_roster();complete=dict.fromkeys(roster,False)
    ids=[k for k,r in roster.items() if r['category']=='certified']
    exp,control,profiles,_=pac_setup()
    for missing,interpretable in [(4,False),(3,True)]:
        b=dict(complete)
        for key in ids[:missing]:b[key]=None
        report=pac_comparison_report(roster,{'B':b,'C':complete},experiment=exp,control=control,profiles=profiles)
        assert report['comparisons']['C-B']['interpretable']==interpretable
        assert 'B-A' not in report['comparisons'] and report['automatic_winner'] is False
        assert report['comparisons']['C-B']['attribution']=='allocation_plus_review'
        assert report['seed_uncertainty']=='unavailable_single_seed'
    report=pac_comparison_report(roster,{'B':complete},experiment=exp,control=control,profiles=profiles)
    assert report['missing_arms']==['C'] and not report['comparisons']['C-B']['interpretable']
    exp,control,profiles,_=pac_setup('matched',True)
    report=pac_comparison_report(roster,{a:complete for a in 'ABCD'},experiment=exp,control=control,profiles=profiles)
    assert set(report['comparisons'])=={'B-A','C-B','D-B','C-D'}
    assert report['comparisons']['C-D']['attribution']=='review_substitution'
    assert report['parameter_endpoint']=='DEFERRED' and report['assessment']=='NOT_EXECUTED'


def test_pac_costs_equal_allowance_not_useful_exposure_or_wall_time():
    from oeis_learn.generalization.reconciliation import pac_costs
    b=dict(allowed_tokens=100,useful_tokens=70,reserved_tokens=0,discarded_tokens=20,unspent_tokens=10,examples=7,updates=3,seconds=12.)
    c=dict(b,useful_tokens=90,discarded_tokens=0,seconds=20.)
    report=pac_costs({'B':b,'C':c,'A':None})
    assert report['arms']['A'] is None and report['equal_allowance']
    assert not report['equal_useful_exposure'] and not report['matched_wall_time']
    with pytest.raises(ValueError):pac_costs({'B':dict(b,discarded_tokens=0)})
