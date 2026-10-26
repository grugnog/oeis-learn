"""Exact full-roster observed-envelope forecast; incomplete work is UNKNOWN."""
from .identity import seal,number,validate
from .pilot_contract import SPEC,PROFILE


def forecast(observations,*,source,work):
    valid=(len(observations)==3 and len({r.get('identity') for r in observations})==3)
    valid=valid and all(r.get('complete') is True and r.get('rows')==296 and r.get('source')==source
        and r.get('work')==work and type(r.get('seconds')) in (int,float) and r['seconds']>0 for r in observations)
    if valid:
        for r in observations:number(r['seconds'],positive=True)
    evaluation=12*max(r['seconds'] for r in observations) if valid else None
    fixed=sum(SPEC['reservations'][k] for k in ('learner','qualification','setup','cleanup','archive'))
    projected=fixed+evaluation if valid else None
    maximum=sum(SPEC['reservations'].values())
    fits=valid and evaluation<=SPEC['reservations']['evaluation'] and projected<=.8*maximum
    return seal(dict(version='pipeline-pilot-cost/v1',profile=PROFILE['identity'],source=source,work=work,
        observations=observations,evaluation_seconds=evaluation,projected_seconds=projected,budget_seconds=maximum,
        within_budget=bool(fits),variability='twice largest of three exact-scope observations; not population p95',
        status='ADMISSIBLE_COST_ONLY' if fits else 'UNKNOWN' if not valid else 'OVER_BUDGET',
        limitation='Early-checkpoint costs do not guarantee later-model lengths/coverage; frozen stage caps still apply.'))


def validate_forecast(record,*,source):
    if not isinstance(record,dict):return False
    expected=forecast(record.get('observations',[]),source=source,work=record.get('work'))
    return record==expected and record['within_budget'] is True
