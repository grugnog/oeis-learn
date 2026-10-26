"""Public runtime cost sentinels, never actual-model learning evidence."""
import math
import time
from .identity import digest,seal


def sentinels(client,*,deadline,policies):
    from oeis_learn.fresh.bigint_diagnostics import _raw,_seal,score
    from oeis_learn.synthesis.admission import VisiblePrompt
    from oeis_learn.synthesis.contracts import decimal_integer
    small=list(range(100));large=[(1<<4095)+i for i in range(100)]
    cases=[('identity','(program (return $index))',small,small),
        ('quadratic','(program (return (mul $index $index)))',small,[i*i for i in small]),
        ('branch','(program (local $b0 bigint) (if (lt $index (int 50)) (then (set $b0 $index)) (else (set $b0 (mul $index $index)))) (return (get $b0)))',small,[i if i<50 else i*i for i in small]),
        ('factorial','(program (local $b0 bigint) (local $c0 control) (set $b0 (int 1)) (loop $c0 (to_control $index) (set $b0 (mul (get $b0) (to_bigint (increment (get $c0)))))) (return (get $b0)))',small,[math.factorial(i) for i in small]),
        ('large-cancel','(program (return (sub (mul (mul $index $index) (mul $index $index)) (mul (mul $index $index) (mul $index $index)))))',large,[0]*100)]
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
    complete=all(x['complete'] and x['success'] for r in result['outcomes'].values() for x in r.values())
    maximum=max(x['scoring']['elapsed_seconds'] for r in result['outcomes'].values() for x in r.values())
    return seal(dict(version='pipeline-pilot-scoring-cost/v1',fixture=True,learned_success=False,
        cases=[x[0] for x in cases],source_identity=digest([(x[0],x[1]) for x in cases]),
        complete=complete,outcomes=result['outcomes'],maximum_selected_program_seconds=maximum,
        interpretation='Public cold/warm exact100-index runtime costs, not sequence-learning scores or universal source bounds.'))
