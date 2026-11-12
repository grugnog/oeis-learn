"""Prospective November runtime amendment; never applied to historical freezes.

Ceilings include headroom, not expected runtime. The measured initial round is
367–401 hours plus preparation. Token horizons, portfolio limits and panel
membership do not change. Training remains bounded by 24 hours per task.
"""
from .contracts import seal

REFERENCE_NAME='research-reservation-v2'
CALIBRATION_SECONDS=72*3600
TRIAL_SECONDS=96*3600
TRAINING_SECONDS=24*3600
SCREEN_CHILD_SECONDS=4*3600
EVALUATION_SECONDS=72*3600
TOTAL_SECONDS=3600*3600
LIMITS_SECONDS=dict(acquisition=24*3600,preparation=96*3600,
    qualification_acceptance=24*3600,evaluation_analysis=3000*3600,packaging=4*3600)


def setup_seconds():
    from .research_budget import SETUP_SECONDS
    return dict(SETUP_SECONDS,prepare=CALIBRATION_SECONDS+1800)


def role_seconds(row):
    shared=row['trial'] and row['trial']['phase'] not in {'selection','confirmation'}
    if row['operation']=='train':return SCREEN_CHILD_SECONDS if shared else TRAINING_SECONDS
    if row['operation'] in {'generate','score'}:
        per_member=30 if row['operation']=='score' else 900 if row.get('all_methods') else 300
        seconds=len(row['members'])*per_member+(2 if shared else 1801)
        if seconds>EVALUATION_SECONDS:raise ValueError('evaluation must be prospectively partitioned')
        return seconds
    return row['seconds']


def screening(roster):
    """Count the seven evaluation pairs exactly once across the three looks."""
    rows=[r for r in roster['tasks'] if r['trial'] and r['trial']['ordinal']==0
          and r['trial']['phase'] not in {'selection','confirmation'}]
    evals=[r for r in rows if r['operation'] in {'generate','score'}]
    return dict(startup_us=20*1000000,
        evaluation_us=[sum(r['seconds'] for r in evals[:6])*1000000,
                       sum(r['seconds'] for r in evals[6:10])*1000000,
                       sum(r['seconds'] for r in evals[10:])*1000000],
        training_us=sum(r['seconds'] for r in rows if r['operation']=='train')*1000000,
        finalization_us=1800*1000000)


def reservation(roster):
    if seal(roster)!=roster or roster.get('version')!='progressive-research-role-roster/v2':
        raise ValueError('original revised roster required')
    if any(r['seconds']!=role_seconds(r) for r in roster['tasks']):
        raise ValueError('revised role reservations differ')
    for i in range(16):
        rows=[r for r in roster['tasks'] if r['trial'] and r['trial']['ordinal']==i
              and r['trial']['phase'] not in {'selection','confirmation'}]
        if len(rows)!=18 or sum(r['seconds'] for r in rows)+1800>TRIAL_SECONDS:
            raise ValueError('revised screening children exceed shared trial')
    training=[r for r in roster['tasks'] if r['operation']=='train']
    if len(training)!=60:raise ValueError('complete original learning roster required')
    setup=setup_seconds()
    ancillary=dict(acquisition=0,
        preparation=sum(setup[k] for k in ('census','sensitivity','materials','roster','construction','prepare','preparation_host')),
        qualification_acceptance=sum(setup[k] for k in ('images','base_qualification','matrix','qualify','freeze','acceptance')),
        evaluation_analysis=sum(r['seconds'] for r in roster['tasks'] if r['operation']!='train')+16*1800+setup['evaluation_timing'],
        packaging=setup['packaging'])
    training_seconds=sum(r['seconds'] for r in training)
    total=training_seconds+sum(ancillary.values())
    deficits=['ancillary:'+k for k,v in ancillary.items() if v>LIMITS_SECONDS[k]]
    if training_seconds>536*3600:deficits.append('selected_training')
    if total>TOTAL_SECONDS:deficits.append('selected_aggregate')
    return seal(dict(version='progressive-research-reservation/v2',roster=roster['identity'],
        status='BLOCKED' if deficits else 'READY',deficits=deficits,
        training_seconds=training_seconds,ancillary_seconds=ancillary,total_seconds=total,
        limits_seconds=LIMITS_SECONDS,selected_maximum_seconds=TOTAL_SECONDS,
        setup_seconds=setup,role_seconds=dict(primary_member=300,all_methods_member=900,score_member=30),
        original_input_costs='retained original journals; no refund or new acquisition',
        screening_accounting='each child counted once by operation, plus sixteen shared finalization reserves',
        scientific_budgets_unchanged=True,assessment_exposed=False))
