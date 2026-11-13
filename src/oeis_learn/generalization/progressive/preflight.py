"""Preparation and sustained protocol checks that confer no execution authority."""
import json
from .contracts import digest, seal, verify_refs


def prepare(root, *, cohorts, exposure, sensitivity_design):
    from .cohorts import census
    from .sensitivity import validate_design
    [exposed]=verify_refs(root,[exposure],max_bytes=32<<20)
    history=json.loads(exposed)
    panels=[json.loads(raw) for raw in verify_refs(root,cohorts,max_bytes=64<<20)]
    reports=[census(c,exposure=history,other_cohorts=[other for other in panels if other!=c]) for c in panels]
    validate_design(sensitivity_design)
    sensitivity=dict(design=sensitivity_design,status='VALIDATED_UNEXECUTED',
                     reason='execution requires its separately frozen preparation allowance')
    return seal(dict(version='progressive-outcome-blind-preflight/v1',executable=False,
                     panels=reports,sensitivity=sensitivity,
                     inputs=dict(cohorts=cohorts,exposure=exposure,design=digest(sensitivity_design)),
                     status='PENDING_QUALIFICATION_AND_FREEZE',scientific_status='UNMEASURED'))


def sustained(plan, *, allowance_us):
    fields={'version','tasks','interval_us','interrupt_task','interrupt_after_us',
            'finalization_us','sensors','adoption'}
    if (not isinstance(plan,dict) or set(plan)!=fields or plan['version']!='progressive-sustained-plan/v1'
        or plan['adoption']!='DIAGNOSTIC_ONLY' or type(allowance_us) is not int or allowance_us<=0):
        raise ValueError('closed prospectively reserved sustained protocol required')
    tasks=plan['tasks']
    if (not isinstance(tasks,list) or len(tasks)!=2 or any(set(t)!={'slot','active_us'} for t in tasks)
        or len({t['slot'] for t in tasks})!=2 or any(type(t['slot']) is not str or not t['slot'] for t in tasks)
        or any(type(t['active_us']) is not int or not 0<t['active_us']<=86400000000 for t in tasks)):
        raise ValueError('two finite linked task durations required')
    if (type(plan['interval_us']) is not int or not 0<plan['interval_us']<=60000000
        or type(plan['finalization_us']) is not int or plan['finalization_us']<1800000000
        or any(t['active_us']<=plan['finalization_us'] for t in tasks)
        or sum(t['active_us'] for t in tasks)>allowance_us):
        raise ValueError('observation/finalization/selected allowance violated')
    target=next((t for t in tasks if t['slot']==plan['interrupt_task']),None)
    if (target is None or type(plan['interrupt_after_us']) is not int or
        not 0<plan['interrupt_after_us']<target['active_us']-plan['finalization_us']):
        raise ValueError('predeclared useful-work interruption required')
    if not isinstance(plan['sensors'],dict) or set(plan['sensors'])!={'memory','disk','thermal','throughput'}:
        raise ValueError('explicit telemetry and missing-sensor reasons required')
    for sensor in plan['sensors'].values():
        if (not isinstance(sensor,dict) or set(sensor)!={'available','reason'} or type(sensor['available']) is not bool
            or (sensor['reason'] is not None if sensor['available'] else not isinstance(sensor['reason'],str) or not sensor['reason'].strip())):
            raise ValueError('explicit sensor availability/reason required')
    return seal(dict(version='progressive-sustained-preflight/v1',plan=plan,allowance_us=allowance_us,
                     executable=False,status='PENDING_ACTUAL_DURATION',actual_elapsed_us=None))
