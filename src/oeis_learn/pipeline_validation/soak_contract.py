"""Closed engineering work roster and cumulative clocks; standard library only."""
import copy
import math
import statistics
import time
from .identity import seal,digest,validate,number

SPEC=seal(dict(version='pipeline-soak-work/v1',profile='m100',batches=[8,16,32],
    trials=['warmup','calibration-0','calibration-1','calibration-2'],epochs_per_trial=2,
    work_windows_seconds=[3600,3600],restart_reserve_seconds=600,parent_seconds=7800,
    max_updates=49152,max_cycles=128,max_jobs=32768,window_seconds=300,
    checkpoint_generations=3,cadence_training_passes=8,native_pulse_seconds=600,
    native_pulses=12,native_cases=18,service_seconds=4,body_tokens=1024,
    selection='fastest complete median; smallest batch within five percent',
    drift_limit=.10,execution='pipeline-sizing-execution/v1'))


def manifest():
    from .identity import seal
    from pathlib import Path
    import json
    base=Path(__file__).resolve().parents[3]/'configs/foundation/pipeline_model_sizing.json'
    value=json.loads(base.read_text())
    value.update(version='pipeline-soak-validation/v1',scope='engineering-soak',
        stage_registry_version='pipeline-soak-registry/v1',phases=[1,2,3,4,5],engines='both',
        workloads=['full-partition-soak/v1'],comparisons=['training-batch8-16-32/v1'])
    return seal(value)


def data_contract(examples):
    names={'training','parameter_development','structural_development'}
    if set(examples)!=names:raise ValueError('only complete approved visible partitions required')
    counts={};tokens={};ids={}
    for name,minimum in [('training',512),('parameter_development',128),('structural_development',128)]:
        rows=examples[name]
        if not isinstance(rows,list) or not minimum<=len(rows)<=4096:raise ValueError('full approved bounded partition required: '+name)
        ids[name]=[r['id'] for r in rows]
        if len(set(ids[name]))!=len(rows):raise ValueError('unique full partition IDs required')
        for row in rows:
            if len(row['values'])!=20 or len(row['indices'])!=20 or not 0<len(row['tokens'])<=1024:
                raise ValueError('visible20 and bounded complete training tokens required')
        counts[name]=len(rows);tokens[name]=sum(len(r['tokens']) for r in rows)
    if len(set(sum(ids.values(),[])))!=sum(counts.values()):raise ValueError('split IDs must be distinct')
    return seal(dict(version='pipeline-soak-data/v1',counts=counts,tokens=tokens,
        ids=ids,payload=digest(examples),cadence_tokens=8*tokens['training'],
        lengths={k:dict(min=min(len(r['tokens']) for r in rows),max=max(len(r['tokens']) for r in rows)) for k,rows in examples.items()}))


def select_batch(rows,*,expected_tokens=None):
    eligible={}
    for batch in (8,16,32):
        trials=rows.get(str(batch),[])
        try:
            if len(trials)!=3 or not all(t['complete'] is True for t in trials):continue
            if len({t['tokens'] for t in trials})!=1:continue
            if expected_tokens is not None and any(t['tokens']!=expected_tokens for t in trials):continue
            rates=[number(t['tokens'],positive=True)/number(t['seconds'],positive=True) for t in trials]
            eligible[batch]=statistics.median(rates)
        except (ValueError,KeyError,TypeError):continue
    if not eligible:raise ValueError('no complete resource-qualified training batch')
    return min(b for b,r in eligible.items() if r>=max(eligible.values())*.95)


def lease(source,work,choice,*,now=None):
    from oeis_learn.synthesis.state import require_hash
    for value in (source,work,choice):require_hash(value)
    now=time.monotonic() if now is None else number(now)
    return seal(dict(version='pipeline-soak-lease/v1',source=source,work=work,choice=choice,
        started=now,expires=now+SPEC['parent_seconds'],windows=SPEC['work_windows_seconds'],
        parent_seconds=SPEC['parent_seconds']))


def remaining(record,*,now=None):
    validate(record,{'version','source','work','choice','started','expires','windows','parent_seconds'},'pipeline-soak-lease/v1')
    if (record['parent_seconds']!=7800 or record['windows']!=[3600,3600]
        or record['expires']!=record['started']+7800):raise ValueError('immutable soak parent required')
    from oeis_learn.synthesis.state import require_hash
    for k in ('source','work','choice'):require_hash(record[k])
    number(record['started']);number(record['expires'],positive=True)
    return max(0,record['expires']-(time.monotonic() if now is None else number(now)))
