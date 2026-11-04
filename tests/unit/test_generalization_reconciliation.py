import copy
import pytest
from oeis_learn.generalization.reconciliation import reconcile_measurements


def fixture():
    study=dict(identity='study',seeds=[1,2],inference=dict(attempts=8))
    plan=dict(version='generalization-measurement-plan/v1',study='study',required_stages=['evaluate'],roster={
        'a':dict(family='small',seed=1,cell='raw',attempts=2),
        'b':dict(family='large',seed=1,cell='raw',attempts=2),
        'c':dict(family='large',seed=2,cell='raw',attempts=2)})
    def attempt(row,n,outcome): return dict(row=row,ordinal=n,outcome=outcome,queue_us=1,service_us=None,cleanup_us=2)
    measured=dict(version='generalization-attempts/v1',attempts=[attempt('a',0,True),attempt('b',0,False),attempt('b',1,False)])
    return plan,measured,study


def test_missing_seed_rows_and_unequal_families_have_fixed_bounds():
    plan,measurements,study=fixture()
    result=reconcile_measurements(plan,measurements,study)
    assert result['summary']==dict(n=3,successes=1,unresolved=1,lower='1/3',upper='2/3',completion='2/3',macro_lower='1/2',macro_upper='3/4')
    assert result['seeds']['raw']['2']['completion']=='0'
    assert result['families']['raw']['2']['large']['upper']=='1'
    assert result['coverage']['missing_attempts']==3
    assert result['costs']['service']==dict(measured_us=0,unavailable=3)


@pytest.mark.parametrize('change',['duplicate','extra','bad_cost','bool_cost','hidden_truth','wrong_seed','missing_seed','bad_outcome'])
def test_invalid_source_records_rejected(change):
    plan,m,study=fixture()
    if change=='duplicate': m['attempts'].append(copy.deepcopy(m['attempts'][0]))
    if change=='extra': m['attempts'][0]['row']='unknown'
    if change=='bad_cost': m['attempts'][0]['queue_us']=-1
    if change=='bool_cost': m['attempts'][0]['queue_us']=True
    if change=='hidden_truth': m['attempts'][0]['target_source']='secret'
    if change=='wrong_seed': plan['roster']['c']['seed']=3
    if change=='missing_seed': del plan['roster']['c']
    if change=='bad_outcome': m['attempts'][0]['outcome']=1
    with pytest.raises(ValueError): reconcile_measurements(plan,m,study)
