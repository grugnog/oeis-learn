"""Bounded diagnostic native pulses, independent of generated-program metrics."""
import time
from .identity import digest,seal


def _native_pulse(client,control,recipe,deadline):
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
    failure=None
    try:
        for offset in range(0,len(rows),4):
            if time.monotonic()+4>deadline:raise TimeoutError('pulse cannot fit immutable workload window')
            drive(None,[job(i,rows[i]) for i in range(offset,min(offset+4,len(rows)))],workers=4,
                ledger=ledger,policies=recipe['pipeline'])
        if not all(r['complete'] for r in results):raise RuntimeError('native health pulse incomplete')
        for r in results:
            if (r['id']=='no-fit')!=(r['admitted']==0):raise RuntimeError('native health independent admission mismatch')
    except BaseException as exc:
        failure=exc
    finally:
        for key in ledger.records:ledger.cancel(key,'pulse failed/cancelled')
        boundary=ledger.boundary(recipe['pipeline'])
        compact={key:{k:v for k,v in r.items() if k!='result'} for key,r in ledger.records.items()}
        result=dict(ordinal=ordinal,seconds=time.monotonic()-began,version='pipeline-soak-pulse/v1',
            results=results,planned=[r['id'] for r in rows],unstarted=[r['id'] for r in rows if r['id'] not in timing],
            timing=timing,jobs=compact,boundary=boundary,generated_work=False,error=None if failure is None else type(failure).__name__+': '+str(failure))
        write(control.output/f'native-pulse-{ordinal}.json',seal(result))
    if failure:raise failure
    return result


def pulse(client,control,selection,recipe,deadline,*,gpu_probe=None):
    """GPU operations remain on the learner thread; one bounded native peer drains."""
    import threading
    from oeis_learn.fresh.pipeline_jobs import merge_boundary
    from oeis_learn.fresh.common import write
    from oeis_learn.synthesis.worker_pool import NonQuiescentWorker
    values={}
    def native():
        try:values['result']=_native_pulse(client,control,recipe,deadline)
        except BaseException as exc:values['error']=exc
    thread=threading.Thread(target=native,daemon=True);thread.start()
    failure=None
    try:
        if gpu_probe is not None:gpu_probe()
        while thread.is_alive() and time.monotonic()<deadline:thread.join(timeout=min(.1,max(0,deadline-time.monotonic())))
    except BaseException as exc:
        failure=exc
    finally:
        if thread.is_alive():
            cleanup=time.monotonic()+5;client.worker.cancel_all();thread.join(timeout=max(0,cleanup-time.monotonic()))
            if thread.is_alive():raise NonQuiescentWorker('native probe did not drain within five seconds')
    if 'error' in values:
        # The native peer has already written its complete partial ledger.
        import json
        from pathlib import Path
        p=control.output/f"native-pulse-{control.state['pulses']}.json"
        if p.is_file():
            partial=json.loads(p.read_text());selection['pipeline_boundary']=merge_boundary(selection['pipeline_boundary'],partial['boundary'])
        raise values['error']
    if 'result' not in values:raise RuntimeError('native probe terminal result missing')
    result=values['result']
    selection['pipeline_boundary']=merge_boundary(selection['pipeline_boundary'],result['boundary'])
    write(control.output/f"native-pulse-{result['ordinal']}.json",seal(result))
    control.component('native_probes',result['seconds'],jobs=len(result['results']));control.pulsed()
    if failure:raise failure
