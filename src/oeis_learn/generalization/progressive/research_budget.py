"""Prospective outer reservation for the selected registered research round.

Screening's search/evaluation/finalization is inside its sixteen two-hour trials.
All other roles are charged once to the execution contract's ancillary envelopes.
The ceilings are not expected durations and cannot establish measured feasibility.
"""
from .contracts import seal
from oeis_learn.pipeline_validation.research_limits import IMAGE_SECONDS, ACCEPTANCE_SECONDS

LIMITS_SECONDS=dict(acquisition=24*3600,preparation=24*3600,
    qualification_acceptance=12*3600,evaluation_analysis=72*3600,packaging=4*3600)
# Supervisor envelopes include bounded startup and cleanup. Inner calibration
# retains its original twelve hours; its enclosing command has five minutes
# for dispatch and reporting. No scientific work borrows these extra seconds.
SETUP_SECONDS=dict(images=2*IMAGE_SECONDS,base_qualification=7200,matrix=120,qualify=3700,
    census=21720,sensitivity=7200,materials=3600,roster=1800,construction=7200,
    prepare=43500,freeze=7200,acceptance=ACCEPTANCE_SECONDS+100,evaluation_timing=14430,
    preparation_host=1200,packaging=14400)
ROLE_SECONDS=dict(generate=4200,score=1980,audit_language=3600)


def reservation(roster):
    if seal(roster)!=roster or roster.get('version')!='progressive-research-role-roster/v1':
        raise ValueError('original finite research roster required')
    trials={i:[] for i in range(16)};training=[];outside=[]
    for row in roster['tasks']:
        meta=row['trial']
        if meta and meta['phase'] not in {'selection','confirmation'}:
            if meta['ordinal'] not in trials:raise ValueError('registered screening trial required')
            trials[meta['ordinal']].append(row)
        elif row['operation']=='train':training.append(row)
        else:outside.append(row)
    if any(len(rows)!=18 or sum(r['seconds'] for r in rows)+1800>7200 for rows in trials.values()):
        raise ValueError('complete screening children and protected finalization must fit their shared trial')
    if (len(training)!=12 or any(r['seconds']>43200 for r in training)
            or any(r['seconds']<=0 for r in roster['tasks'])):
        raise ValueError('selected paired diversity/confirmation reservations differ')
    training_seconds=sum(r['seconds'] for r in training)+16*7200
    setup=SETUP_SECONDS
    ancillary=dict(acquisition=0,
        preparation=sum(setup[k] for k in ('census','sensitivity','materials','roster','construction','prepare','preparation_host')),
        qualification_acceptance=sum(setup[k] for k in ('images','base_qualification','matrix','qualify','freeze','acceptance')),
        evaluation_analysis=sum(r['seconds'] for r in outside)+setup['evaluation_timing'],
        packaging=setup['packaging'])
    deficits=['ancillary:'+k for k,v in ancillary.items() if v>LIMITS_SECONDS[k]]
    if training_seconds>176*3600:deficits.append('selected_training')
    total=training_seconds+sum(ancillary.values())
    if total>312*3600:deficits.append('selected_aggregate')
    return seal(dict(version='progressive-research-reservation/v1',roster=roster['identity'],
        status='BLOCKED' if deficits else 'READY',deficits=deficits,
        training_seconds=training_seconds,ancillary_seconds=ancillary,total_seconds=total,
        limits_seconds=LIMITS_SECONDS,selected_maximum_seconds=312*3600,
        setup_seconds=SETUP_SECONDS,role_seconds=ROLE_SECONDS,
        original_input_costs='retained separately in the registered-data receipt and original journals; no new acquisition',
        screening_accounting='all children and shared finalization counted once inside sixteen two-hour trials',
        scientific_budgets_unchanged=True,assessment_exposed=False))


def verify_reservation(root,refs,plans,contract):
    import json
    from .research_roster import ROSTER_DOCUMENT_BYTES
    from .contracts import verify_refs
    selected=[r['artifact'] for r in refs if r['name']=='research-reservation']
    if len(selected)!=1:raise ValueError('one original prospective research reservation required')
    roster=json.loads(verify_refs(root,[contract['roster']],max_bytes=ROSTER_DOCUMENT_BYTES)[0])
    expected=reservation(roster)
    if expected['status']!='READY' or json.loads(verify_refs(root,selected)[0])!=expected:
        raise ValueError('original research ancillary/aggregate reservation differs')
    ceilings={row['id']:row['seconds']*1000000 for row in roster['tasks']}
    if (set(ceilings)!={p['id'] for p in plans} or len(plans)!=len(ceilings)
            or any(p['caps']['elapsed_us']!=ceilings[p['id']] for p in plans)):
        raise ValueError('research task ceilings differ from the original outer reservation')
    return expected
