"""Exact full-roster observed-envelope forecast; incomplete work is UNKNOWN."""
from .identity import seal,number,validate
from .pilot_contract import SPEC,PROFILE


def forecast(observations,*,source,work,archives=()):
    valid=(len(observations)==3 and len({r.get('identity') for r in observations})==3 and all(seal(r)==r for r in observations))
    valid=valid and all(r.get('complete') is True and r.get('rows')==296 and r.get('source')==source
        and r.get('work')==work and all(type(r.get(k)) in (int,float) and r[k]>0
            for k in ('seconds','generation_seconds','scoring_seconds','projected_all_selected_scoring_seconds')) for r in observations)
    if valid:
        for r in observations:
            for k in ('seconds','generation_seconds','scoring_seconds','projected_all_selected_scoring_seconds'):number(r[k],positive=True)
            if r['seconds']+1e-6<r['generation_seconds']+r['scoring_seconds']:valid=False
    import math
    generation=max(30,math.ceil(2*max(r['generation_seconds'] for r in observations))) if valid else None
    scoring=max(30,math.ceil(2*max(r['projected_all_selected_scoring_seconds'] for r in observations))) if valid else None
    evaluation=6*(generation+scoring) if valid else None
    archive_valid=(len(archives)==3 and len({r.get('identity') for r in archives})==3 and
        all(seal(r)==r and r.get('complete') is True and r.get('source')==source and r.get('members')==4096 and
            r.get('bytes',0)>=16<<20 and type(r.get('filesystem_device')) is int and type(r.get('seconds')) in (int,float) and r['seconds']>0 for r in archives))
    archive_valid=archive_valid and len({r['filesystem_device'] for r in archives})==1
    if archive_valid:
        for r in archives:number(r['seconds'],positive=True)
    archive_envelope=2*max(r['seconds']*((512<<20)/r['bytes']) for r in archives) if archive_valid else None
    fixed=sum(SPEC['reservations'][k] for k in ('learner','qualification','setup','cleanup','archive'))
    projected=fixed+evaluation if valid else None
    maximum=sum(SPEC['reservations'].values())
    fits=valid and archive_valid and archive_envelope<=900 and evaluation<=SPEC['reservations']['evaluation'] and projected<=.8*maximum
    return seal(dict(version='pipeline-pilot-cost/v1',profile=PROFILE['identity'],source=source,work=work,
        observations=observations,archives=list(archives),archive_envelope_seconds=archive_envelope,
        generator_role_seconds=generation,scorer_role_seconds=scoring,
        evaluation_seconds=evaluation,projected_seconds=projected,budget_seconds=maximum,
        within_budget=bool(fits),variability='componentwise twice largest of three full finite-attempt observations; includes public all-selected scoring estimate; not population p95',
        status='ADMISSIBLE_COST_ONLY' if fits else 'UNKNOWN' if not valid or not archive_valid else 'OVER_BUDGET',
        limitation='Early-checkpoint costs do not guarantee later-model lengths/coverage; frozen stage caps still apply.'))


def validate_forecast(record,*,source):
    if not isinstance(record,dict):return False
    expected=forecast(record.get('observations',[]),source=source,work=record.get('work'),archives=record.get('archives',[]))
    return record==expected and record['within_budget'] is True
