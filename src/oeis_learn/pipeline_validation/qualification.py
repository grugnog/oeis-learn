"""Current-source engineering and historical carryover ledger; no implicit promotion."""
from .identity import seal

CARRYOVERS={
 'T093':'user deferred; no weight acquisition/conversion',
 'T094':'current whole-pipeline performance, parity and retained arithmetic gap review',
 'T097':'supported commands, interruption/resume, isolation and exports on both engines',
 'T098':'all applicable current-source qualification identities and gate decisions',
 'T099':'matched prospective contracts and fresh sealed cohort',
 'T100':'all six matched representation arms and full controls',
 'T101':'single registered transfer endpoint decision',
 'T102':'separately admitted curriculum then replay',
 'T103':'reviewed assistance inputs and matched comparison or explicit deferral',
 'T104':'component sensitivity and separately selected matched retraining',
 'T105':'immutable final plan and explicit authorized execution',
 'T106':'evidence-backed discovery review after final results',
 'T107':'evidence-backed longer-run go/no-go report',
}


def qualification(gates,*,source):
    complete=bool(gates) and all(g.get('state') in {'PASS','REUSED'} for g in gates.values())
    return seal(dict(version='pipeline-qualification/v1',source=source,gates=gates,
        engineering='PASS' if complete else 'BLOCKED',learning='HISTORICAL_NEGATIVE',
        research='NOT_EXECUTED',long_run_authorized=False,
        historical={'008:T093':dict(disposition='DEFERRED'),
                    '008:T096':dict(disposition='NEGATIVE',consumed_seconds=7730,budget_seconds=14400)},
        carryovers={f'008:{key}':dict(disposition='DEFERRED' if key=='T093' else 'PENDING',closure=value) for key,value in CARRYOVERS.items()},
        retained_arithmetic=dict(small_value_gap='13.29–13.40x historical observation',review_trigger='>5x; review rather than weaken semantics')))
