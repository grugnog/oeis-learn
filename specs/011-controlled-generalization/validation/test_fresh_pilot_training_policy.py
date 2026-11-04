import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fresh_pilot_training_policy import assess,budget


def results(seconds=2):
    return {a:dict(status='PASS',updates=32,tokens=32000,charged_tokens=32000,
        timing=dict(invocation_seconds=60,windows=[dict(updates=8,tokens=8000,seconds=seconds,through_updates=u) for u in (12,20,28)])) for a in ('static','refreshed')}


def test_budget_and_slowest_window_conservative_forecast():
    assert budget([[1000]*16,[1100]*16])==32000
    result=assess(results(),16000)
    assert result['projected_seconds_per_arm']==14300 and result['status']=='PASS'
    slow=results();slow['refreshed']['timing']['windows'][1]['seconds']=4
    assert assess(slow,16000)['status']=='BLOCKED'
    assert not result['automatic_research']
    with pytest.raises(ValueError):budget([[1]*16,[10000]*16])


@pytest.mark.parametrize('defect',['missing','partial','pending','few_updates','few_windows','nan','overlap','tokens','seconds'])
def test_forecast_rejects_incomplete_or_malformed_evidence(defect):
    rows=results();r=rows['refreshed'];w=r['timing']['windows']
    if defect=='missing':rows.pop('static')
    if defect=='partial':r['status']='PARTIAL'
    if defect=='pending':r['charged_tokens']+=1
    if defect=='few_updates':r['updates']=20
    if defect=='few_windows':r['timing']['windows']=w[:1]
    if defect=='nan':w[0]['seconds']=float('nan')
    if defect=='overlap':w[1]['through_updates']=12
    if defect=='tokens':w[0]['tokens']=40000
    if defect=='seconds':w[0]['seconds']=100
    with pytest.raises(ValueError):assess(rows,16000)
