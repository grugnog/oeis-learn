"""Public runtime cost sentinels, never actual-model learning evidence."""
import math
import time
from .identity import digest,seal


def sentinels(client,*,deadline,policies):
    from oeis_learn.fresh.bigint_diagnostics import _raw,_seal,score
    from oeis_learn.synthesis.admission import VisiblePrompt
    from oeis_learn.synthesis.contracts import decimal_integer
    small=list(range(100));large=[(1<<4095)+i for i in range(100)];medium=[(1<<1023)+i for i in range(100)]
    cancellation='(program (return (sub (mul (mul $index $index) (mul $index $index)) (mul (mul $index $index) (mul $index $index)))))'
    cases=[('identity','(program (return $index))',small,small),
        ('quadratic','(program (return (mul $index $index)))',small,[i*i for i in small]),
        ('branch','(program (local $b0 bigint) (if (lt $index (int 50)) (then (set $b0 $index)) (else (set $b0 (mul $index $index)))) (return (get $b0)))',small,[i if i<50 else i*i for i in small]),
        ('factorial','(program (local $b0 bigint) (local $c0 control) (set $b0 (int 1)) (loop $c0 (to_control $index) (set $b0 (mul (get $b0) (to_bigint (increment (get $c0)))))) (return (get $b0)))',small,[math.factorial(i) for i in small]),
        ('large-identity-4096','(program (return $index))',large,large),
        ('large-cancel-1024',cancellation,medium,[0]*100),
        ('large-cancel-4096-stress',cancellation,large,[0]*100)]
    records={};truth={}
    for key,source,indices,values in cases:
        ns=list(map(decimal_integer,indices));vs=list(map(decimal_integer,values))
        prompt=VisiblePrompt(tuple(ns[:20]),tuple(vs[:20]))
        found,status=_raw(client,source,prompt,min(deadline,time.monotonic()+4))
        if found is None or not status['complete']:raise ValueError('public scoring-cost sentinel lacks exact independent visible admission: '+key)
        found.update(_attempt_ordinal=0,_job_id=digest(['public-cost',key]))
        selection=_seal([found],prompt,cost=0,selection='visible-only/v2',snapshot=digest('public-cost'),policies=policies)
        arms=('raw','rescue','equal_budget','shuffled','absent','template')
        records[key]=dict(selections={a:selection for a in arms},arm_evidence={a:dict(complete=True,reason=None) for a in arms},family=key,strata={})
        truth[key]=dict(indices=ns,values=vs)
    result=score(records,truth,client,deadline=deadline)
    # Positive cost calibration must exercise the full100-index production AND
    # independent scoring paths. The original4096-bit degree4 case is retained
    # separately:20-index admission need not imply100-index aggregate fuel fit.
    stress='large-cancel-4096-stress';required=[x[0] for x in cases if x[0]!=stress]
    positive_complete=all(x['complete'] and x['success'] for key in required for x in result['outcomes'][key].values())
    def bounded_stress(row):
        if row['complete'] and row['success']:return True
        scoring=row['scoring'];p=scoring.get('production',{});r=scoring.get('reference',{})
        return (not row['complete'] and not row['success'] and r.get('outcome')=='ok' and r.get('coverage')==100
            and p.get('outcome') in ('failure','resource_exhausted') and p.get('reason') in ('aggregate_fuel','term_fuel'))
    stress_rows=list(result['outcomes'][stress].values())
    stress_known=all(bounded_stress(row) for row in stress_rows)
    disposition='UNRESOLVED' if not stress_known else 'FULL_HORIZON_SUCCESS' if all(x['success'] for x in stress_rows) else 'BOUNDED_RESOURCE_EXHAUSTION'
    complete=positive_complete and stress_known
    maximum=max(x['scoring']['elapsed_seconds'] for r in result['outcomes'].values() for x in r.values())
    return seal(dict(version='pipeline-pilot-scoring-cost/v2',fixture=True,learned_success=False,
        cases=[x[0] for x in cases],source_identity=digest([(x[0],x[1]) for x in cases]),
        work_identity=digest(cases),resources=client.limits.identity,
        required_cases=required,stress_cases=[stress],positive_complete=positive_complete,stress_disposition=disposition,
        complete=complete,outcomes=result['outcomes'],maximum_selected_program_seconds=maximum,
        interpretation='Six required full100-index positive cost fixtures plus retained4096-bit aggregate-fuel stress; maximum includes paid stress failures. Never sequence-learning scores or universal source bounds.'))
