"""Independent admission and process-boundary counterexamples for T139/T140."""
import json
import time
import pytest
from oeis_learn.data.bigint_programs import BigintSampler, materialize
from oeis_learn.generalization.refresh import RefreshPlan
from oeis_learn.pipeline_validation.identity import seal, digest


def test_wide_stream_is_counter_keyed_and_old_stream_unchanged():
    import random
    d=BigintSampler().descriptor(0)
    for law,bounds in [('independent-uniform-integers[-16,16]/v1',(-16,16)),
                       ('independent-uniform-integers[-1024,1024]/v1',(-1024,1024))]:
        descriptor=seal(dict(d,numeric_law=law))
        plan=RefreshPlan([descriptor],seed=17,capacity=100)
        rows,_=plan.take(0,20)
        assert rows==plan.take(0,9)[0]+plan.take(9,11)[0]
        for ordinal,row in enumerate(rows):
            rng=random.Random(int(digest([plan.identity,ordinal])[7:],16))
            assert row['assignment']=={p:str(rng.randint(*bounds)) for p in descriptor['parameter_names']}
    with pytest.raises(ValueError):RefreshPlan([d,seal(dict(BigintSampler().descriptor(1),numeric_law=law))],seed=17,capacity=100)


def collision_rows():
    from tests.integration.test_generalization_preparation import source
    return [source(1),source(2),source(3)]


def test_exclusion_precedes_native_and_parallel_keeps_exact_selection():
    from oeis_learn.generalization.preparation import scan
    from oeis_learn.generalization.exclusions import ExclusionIndex
    rows=collision_rows();protected=materialize(rows[0])
    ex=ExclusionIndex([dict(id='protected',role='selection',holdout='parameter',family=protected['family'],
        source=protected['source'],values=protected['values'],group='protected',variant='protected')])
    def forbidden(*args):raise AssertionError('excluded rows must never reach native')
    results=[]
    for workers in (1,2):
        choice,ledger=scan(rows,role='training',holdout='parameter',count=1,quotas={},structural=False,
            execute=forbidden,exclusions=ex,workers=workers,seconds=20)
        assert not choice.complete
        assert all('finite_window' in r['rules'] for r in ledger['rejections'])
        results.append((choice.payload,ledger))
    assert results[0]==results[1]


def test_reference_feasibility_is_not_native_pass_and_later_duplicates_fail():
    from oeis_learn.generalization.preparation import scan
    rows=collision_rows()
    choice,ledger=scan(rows,role='training',holdout='parameter',count=2,quotas={},structural=False,execute=None,seconds=20)
    assert choice.complete and ledger['status']=='FEASIBLE' and ledger['native_validation']=='NOT_RUN'
    seen={digest(r['source']) for r in choice.rows}
    choice,later=scan(rows,role='training',holdout='parameter',count=2,quotas={},structural=False,
        execute=None,seconds=20,unique_sources=True,seen_sources=seen)
    assert not choice.complete and later['status']=='BLOCKED'
    assert len(later['rejections'])==2


def stall(value):
    time.sleep(10)
    return value


def test_workers_timeout_and_are_reaped():
    import multiprocessing
    from oeis_learn.generalization.preparation import OrderedWorkers
    before={p.pid for p in multiprocessing.active_children()}
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        with OrderedWorkers(2,function=stall) as workers:
            workers.map([1,2],deadline=time.monotonic()+.2)
    assert time.monotonic()-started<5
    assert {p.pid for p in multiprocessing.active_children()}<=before


def test_observed_materialization_matches_independent_coverage():
    from oeis_learn.generalization.coverage import materialize_observed,observe
    for i in range(8):
        row=BigintSampler().sample(i*16)
        exact,coverage=materialize_observed(row)
        assert exact==materialize(row)
        assert coverage==observe(row['source'],tuple(map(str,range(100))),stage=row['stage'])


def test_batched_native_matches_serial_and_never_receives_excluded_rows():
    from oeis_learn.generalization.preparation import scan
    from oeis_learn.synthesis.reference import evaluate
    sampler=BigintSampler();rows=[sampler.sample(i*16) for i in range(20)]
    def serial(source,indices):
        r=evaluate(source,indices);return dict(outcome=r.outcome,values=r.values)
    def batched(*args):raise AssertionError('batch adapter should be used')
    calls=[]
    def batch(rows):
        calls.append(len(rows));return [serial(r['source'],r['indices']) for r in rows]
    batched.batch=batch
    kwargs=dict(role='training',holdout='parameter',count=10,quotas={},structural=False,seconds=20)
    a,al=scan(rows,execute=serial,**kwargs);b,bl=scan(rows,execute=batched,**kwargs)
    assert (a.payload,al)==(b.payload,bl) and max(calls)<=8


def test_native_batch_reorders_ids_and_rejects_missing_or_duplicate_responses():
    from oeis_learn.pipeline_validation.generalization_role import native_batch
    rows=[materialize(row) for row in collision_rows()]
    class Native:
        def request(self,op,candidates,indices,**kw):
            assert op=='execute' and len(candidates)==3 and tuple(indices)==tuple(rows[0]['indices'])
            return dict(complete=True,results=[dict(candidate_id=str(i),outcome='ok',values=rows[i]['values']) for i in (2,0,1)])
    result=native_batch(Native(),rows,deadline=time.monotonic()+10)
    assert [r['values'] for r in result]==[r['values'] for r in rows]
    class Bad:
        def request(self,*a,**kw):return dict(complete=True,results=[dict(candidate_id='0')]*3)
    with pytest.raises(ValueError):native_batch(Bad(),rows,deadline=time.monotonic()+10)
