"""Read-only full-denominator learning summaries; no claim of an asymptotic bound."""
import math
import statistics
from .identity import number,seal


def indexed(rows):
    result={}
    for r in rows:
        if r['id'] in result:raise ValueError('duplicate prompt outcome')
        if type(r['success']) is not bool or type(r['complete']) is not bool or r['success'] and not r['complete']:
            raise ValueError('success requires complete independent scoring')
        result[r['id']]=r
    return result


def coverage(ids,rows):
    if not ids or len(set(ids))!=len(ids):raise ValueError('unique frozen denominator required')
    values=indexed(rows)
    if set(values)-set(ids):raise ValueError('outcome outside frozen population')
    complete=sum(r['complete'] for r in values.values());success=sum(r['success'] for r in values.values())
    n=len(ids);unknown=n-complete
    return dict(denominator=n,complete=complete,success=success,unresolved=unknown,accuracy=success/n,
        coverage=complete/n,lower=success/n,upper=(success+unknown)/n,
        families=len({r['family'] for r in values.values()}),interpretation='fixed-checkpoint measurement bounds, not future-learning bounds')


def paired(before,after):
    a=indexed(before);b=indexed(after)
    if not a or set(a)!=set(b):raise ValueError('identical complete prompt roster required for pairing')
    if any(a[k]['family']!=b[k]['family'] for k in a):raise ValueError('family identity changed')
    known={k for k in a if a[k]['complete'] and b[k]['complete']}
    gained=sum(not a[k]['success'] and b[k]['success'] for k in known)
    lost=sum(a[k]['success'] and not b[k]['success'] for k in known)
    from oeis_learn.fresh.comparison import family_bootstrap
    families={}
    for k in sorted(a):families.setdefault(a[k]['family'],[]).append(int(b[k]['success'])-int(a[k]['success']))
    uncertainty=family_bootstrap(families,seed=20261016) if len(known)==len(a) else None
    return dict(denominator=len(a),known_pairs=len(known),unresolved_pairs=len(a)-len(known),gained=gained,lost=lost,
        net_known=gained-lost,family_uncertainty=uncertainty,
        limitation='Descriptive paired family resampling only; one training seed and repeated development inspection.')


def extrapolate(points):
    unknown=lambda why:dict(status='UNKNOWN',reason=why,guaranteed_capacity_upper_bound=None)
    rows=[r for r in points if r['tokens']>0]
    if len(rows)<4:return unknown('at least four comparable nonzero-exposure checkpoints required')
    for r in rows:
        number(r['tokens'],positive=True);number(r['accuracy']);number(r['coverage'])
    if any(r['coverage']<.95 or r['coverage']>1 or r['accuracy']>1 for r in rows):return unknown('inadequate complete coverage')
    if any(b['tokens']<=a['tokens'] for a,b in zip(rows,rows[1:])):return unknown('exposure did not strictly advance')
    slopes=[(b['accuracy']-a['accuracy'])/math.log2(b['tokens']/a['tokens']) for a,b in zip(rows,rows[1:])]
    predicted=rows[-2]['accuracy']+statistics.median(slopes[:-1])*math.log2(rows[-1]['tokens']/rows[-2]['tokens'])
    error=abs(predicted-rows[-1]['accuracy'])
    if any(s<0 for s in slopes) or error>.05:return unknown('regression or held-out later-checkpoint backtest failure')
    last=rows[-1]['accuracy'];recent=statistics.median(slopes[-3:]);lo=min(slopes[-3:]);hi=max(slopes[-3:])
    clip=lambda x:max(0.,min(1.,x))
    scenarios={}
    for multiple,doublings in [('2x',1),('4x',2)]:
        scenarios[multiple]=dict(plateau=last,recent_trend=clip(last+recent*doublings),
            diminishing_gain=clip(last+sum(recent/(2**i) for i in range(1,doublings+1))),
            observed_slope_sensitivity=[clip(last+lo*doublings),clip(last+hi*doublings)])
    return dict(status='CONDITIONAL',scenarios=scenarios,backtest_absolute_error=error,
        guaranteed_capacity_upper_bound=None,interpretation='Same-regime token-exposure scenarios; not confidence bounds or an asymptote.')


def compute_estimates(points,projection,denominator):
    """Observed learner-role rates only; future evaluation/total costs stay unknown."""
    unknown=lambda why:dict(status='UNKNOWN',reason=why,future_total_seconds=None)
    rows=[r for r in points if r['tokens']>0]
    if projection['status']!='CONDITIONAL':return unknown('learning scenario unavailable')
    if len(rows)<4 or any(type(r.get('learner_role_seconds')) not in (int,float) for r in rows):
        return unknown('comparable observed learner-role wall costs unavailable')
    rates=[]
    for a,b in zip(rows,rows[1:]):
        seconds=b['learner_role_seconds']-a['learner_role_seconds'];tokens=b['tokens']-a['tokens']
        if seconds<=0 or tokens<=0:return unknown('role time/exposure did not strictly advance')
        rates.append(seconds/tokens)
    rates=rates[-3:];last=rows[-1];scenarios={}
    for key,multiple in [('2x',2),('4x',4)]:
        tokens=last['tokens']*(multiple-1);bounds=[tokens*min(rates),tokens*max(rates)]
        gain=max(0,projection['scenarios'][key]['recent_trend']-last['accuracy'])*denominator
        scenarios[key]=dict(additional_tokens=tokens,additional_learner_seconds_range=bounds,
            recent_trend_additional_verified_sequences=gain,
            recent_trend_seconds_per_additional_sequence_range=[x/gain for x in bounds] if gain>0 else None,
            plateau_additional_verified_sequences=0,plateau_seconds_per_additional_sequence=None,
            future_evaluation_seconds=None,future_total_seconds=None)
    return dict(status='CONDITIONAL',scenarios=scenarios,
        interpretation='Observed last three learner-role seconds/token range; no guaranteed throughput, learning gain or total future-study cost.')
