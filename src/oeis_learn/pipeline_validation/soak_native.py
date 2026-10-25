"""Bounded diagnostic native pulses, independent of generated-program metrics."""
import time
from .identity import digest,seal


def pulse(client,control,selection,recipe,deadline):
    from .role import native_corpus
    from oeis_learn.fresh.pipeline_jobs import JobLedger,NativeRequest,logical_job,merge_boundary
    from oeis_learn.fresh.pipeline_scheduler import drive
    from oeis_learn.synthesis.admission import VisiblePrompt
    from oeis_learn.fresh.common import write
    from contextlib import nullcontext
    began=time.monotonic();ledger=JobLedger();timing={};rows=native_corpus()['rows'];results=[]
    ordinal=control.state['pulses']
    def job(i,row):
        admitted=time.monotonic();until=min(deadline,admitted+4)
        logical=logical_job(contract=recipe['identity'],snapshot=control.binding+':pulse'+str(ordinal),
            mode='native-health',arm='probe',prompt_id=row['id'],attempt_ordinal=0,method='fixture',seed=20261016)
        timing[row['id']]=dict(admitted=admitted,deadline=until)
        def operation():
            affinity=getattr(client.worker,'affinity',None)
            with affinity(i) if affinity else nullcontext():
                timing[row['id']]['dispatched']=time.monotonic()
                result=client.fit(row['source'],VisiblePrompt(tuple(row['indices']),tuple(row['values'])),deadline=until)
                timing[row['id']]['completed']=time.monotonic()
                return dict(complete=result.get('complete') is True,outcome=result['outcome'],
                    response=digest(result),admitted=len(result.get('admitted',[])))
        result=yield NativeRequest(logical,i,until,operation,cancel=client.worker.cancel_all)
        results.append(dict(id=row['id'],**result));return result
    for offset in range(0,len(rows),4):
        if time.monotonic()+4>deadline:raise TimeoutError('pulse cannot fit immutable workload window')
        drive(None,[job(i,rows[i]) for i in range(offset,min(offset+4,len(rows)))],workers=4,
            ledger=ledger,policies=recipe['pipeline'])
    boundary=ledger.boundary(recipe['pipeline'])
    selection['pipeline_boundary']=merge_boundary(selection['pipeline_boundary'],boundary)
    if not all(r['complete'] for r in results):raise RuntimeError('native health pulse incomplete')
    for r in results:
        if (r['id']=='no-fit')!=(r['admitted']==0):raise RuntimeError('native health independent admission mismatch')
    compact={key:{k:v for k,v in r.items() if k!='result'} for key,r in ledger.records.items()}
    write(control.output/f'native-pulse-{ordinal}.json',seal(dict(version='pipeline-soak-pulse/v1',
        results=results,timing=timing,jobs=compact,boundary=boundary,generated_work=False)))
    control.component('native_probes',time.monotonic()-began,jobs=len(rows));control.pulsed()
