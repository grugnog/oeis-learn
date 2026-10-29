"""Prospective calibration and conservative five-pair decisions; no adaptive samples."""
import math
import statistics
from .identity import seal, digest, number

ORDER=['AB','BA','AB','BA','AB']


def calibrate(rows, *, minimum, cap, parent, workload):
    number(minimum,positive=True);number(cap,positive=True)
    if len(rows)!=3 or not all(r['complete'] is True for r in rows):
        raise ValueError('exactly three complete baseline-only calibration receipts required')
    duration=math.ceil(max(minimum,2*max(number(r['seconds']) for r in rows)))
    if duration>cap:
        raise ValueError('calibration cannot fit parent reservation; scope revision required')
    return seal(dict(version='pipeline-calibration/v1',parent=parent,workload=workload,
        baseline_receipts=[r['identity'] for r in rows],reservation_seconds=duration,
        rule=digest('ceil(max(2 * max_elapsed, declared_minimum))'),cold_runs=1,warmups=1,
        calibration_count=3,confirmation_order=ORDER))


def decide(pairs, *, required_strata=None):
    if len(pairs)!=5 or [p.get('order') for p in pairs]!=ORDER:
        return dict(adoption='INCONCLUSIVE',reason='exactly five fresh AB/BA/AB/BA/AB pairs required')
    gains=[];failures=[]
    scopes={p.get('scope') for p in pairs}
    strata=set(required_strata if required_strata is not None else pairs[0].get('strata',{}))
    if len(scopes)!=1 or not scopes<={'cold','warm'} or not strata:
        failures.append('cache scope/mandatory strata missing or inconsistent')
    for i,p in enumerate(pairs):
        if not all(p.get(k) is True for k in ['complete','correct','resources','reset']) or p.get('kind')!='confirmation':
            failures.append(f'pair {i}: incomplete, unqualified, unreset or exploratory')
        if not p.get('work') or p.get('work')!=p.get('candidate_work'):
            failures.append(f'pair {i}: completed work mismatch')
        if set(p.get('strata',{}))!=strata or any(number(v,positive=True)>1.05 for v in p.get('strata',{}).values()):
            failures.append(f'pair {i}: stratum coverage/regression')
        base=number(p['baseline'],positive=True);candidate=number(p['candidate'],positive=True)
        gains.append(1-candidate/base)
        for metric in ['success','complete']:
            if p.get('candidate_'+metric) is None or p.get('baseline_'+metric) is None:
                failures.append(f'pair {i}: fixed-budget {metric} denominator missing')
    for metric in ['success','complete']:
        if all(p.get(a+'_'+metric) is not None for p in pairs for a in ['baseline','candidate']):
            if statistics.median(p['candidate_'+metric] for p in pairs)<statistics.median(p['baseline_'+metric] for p in pairs):
                failures.append('fixed-budget median coverage loss: '+metric)
    if failures:
        outcome='NO_ADOPTION'
    elif min(gains)>=0.10-1e-12:
        outcome='ADOPT'
    elif max(gains)<0.10:
        outcome='NO_ADOPTION'
    else:
        outcome='INCONCLUSIVE'
    return dict(adoption=outcome,paired_gains=gains,median=statistics.median(gains),minimum=min(gains),maximum=max(gains),failures=failures,
        uncertainty='all five pairs; no population confidence claim')
