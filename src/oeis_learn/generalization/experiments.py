"""Frozen progression; no adaptive assessment feedback or combined winner recipe."""
from fractions import Fraction as F

STAGES=('E0','P0','P1-screen','P1-confirm','P2','P3','P4')
SEEDS=(20261016,20261017,20261018)


def promotion(paired,*,expected_seeds=None):
    if len(paired)!=3 or len({r['seed'] for r in paired})!=3 or expected_seeds is not None and {r['seed'] for r in paired}!=set(expected_seeds):
        return 'inconclusive'
    for row in paired:
        if set(row)!={'seed','structural','parameter','completion'}: raise ValueError('selection-only confirmation fields')
        if row['completion']<F(95,100): return 'inconclusive'
        for key in ('structural','parameter'):
            lo,hi=row[key]
            if not -1<=lo<=hi<=1: raise ValueError('invalid paired bounds')
    slo=sum((r['structural'][0] for r in paired),F())/3
    shi=sum((r['structural'][1] for r in paired),F())/3
    plo=sum((r['parameter'][0] for r in paired),F())/3
    phi=sum((r['parameter'][1] for r in paired),F())/3
    if slo>=F(5,100) and all(r['structural'][0]>=0 for r in paired) and plo>=F(-5,100): return 'promote'
    if shi<F(5,100) or any(r['structural'][1]<0 for r in paired) or phi<F(-5,100): return 'retain_baseline'
    return 'inconclusive'


def next_stage(stage,*,valid,selection_completion,feedback):
    if stage not in STAGES or feedback!='selection': raise ValueError('assessment cannot choose progression')
    if not valid or selection_completion<F(95,100): return 'stop_inconclusive'
    return {'E0':'P0','P0':'P1-screen','P1-screen':'P1-confirm'}.get(stage,'stop_requires_registered_next_recipe')


def confirmation_assessment(previous,current,*,exposed_to_decision):
    if exposed_to_decision and previous==current: raise ValueError('fresh assessment required')
    return current


def recipe(stage):
    if stage not in STAGES: raise ValueError('unknown recipe')
    return dict(stage=stage,seeds=SEEDS if stage=='P1-confirm' else SEEDS[1:] if stage=='P0' else SEEDS[:1],
                tokens=0 if stage=='E0' else 4_000_000 if stage=='P4' else 25_000_000,
                learner_seconds=0 if stage=='E0' else 7200 if stage=='P4' else 21600,
                milestones=() if stage=='E0' else (1_000_000,2_000_000,4_000_000) if stage=='P4' else (4_000_000,8_000_000,16_000_000,25_000_000),
                changing_factor={'E0':'evaluation','P0':'seed','P1-screen':'refresh','P1-confirm':'refresh','P2':'support','P3':'target_format','P4':'verified_replay'}[stage])
