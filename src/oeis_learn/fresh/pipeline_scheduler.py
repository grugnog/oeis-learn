"""Single GPU owner with bounded daemon native tasks and explicit drain boundary.

No optimizer/model call occurs on a native task thread. A noncooperative native
or reference operation escalates to the external role watchdog instead of joining
an executor indefinitely.
"""
import copy
import queue
import threading
import time
from oeis_learn.fresh.pipeline_jobs import NativeRequest,JobLedger


def drive(decoder,generators,*,workers=1,lookahead=1,policies=None,batch_size=8,sample=None,ledger=None,native_priority=False,native_expiry_grace=0):
    if workers not in (1,2,4) or lookahead not in (1,2) or batch_size not in (1,8,16,32):
        raise ValueError('unregistered scheduler capacity')
    if type(native_expiry_grace) is not int or native_expiry_grace not in (0,5):
        raise ValueError('registered native expiry cleanup allowance required')
    if sample is None:
        from oeis_learn.decoder.bigint_generation import sample_batch
        sample=sample_batch
    ready={};active={};results=[None]*len(generators);ledger=ledger if ledger is not None else JobLedger()
    done=set();completed=queue.Queue(maxsize=workers)
    def advance(i,value=None):
        try:
            ready[i]=generators[i].send(value)
            if isinstance(ready[i], NativeRequest):
                ledger.reserve(ready[i].job,deadline=ready[i].deadline,work=1)
        except StopIteration as stop:
            results[i]=stop.value;done.add(i);ready.pop(i,None)
    def run_native(i,request):
        try:
            value=request.operation()
            completed.put((i,copy.deepcopy(value),None))
        except BaseException as exc:
            completed.put((i,None,exc))
    next_prompt=0
    def admit():
        nonlocal next_prompt
        while next_prompt<len(generators) and len(ready)+len(active)<lookahead*batch_size:
            advance(next_prompt)
            next_prompt+=1
    admit()
    cursor=0
    try:
        while len(done)<len(generators):
            admit()
            # Observe before admission/dispatch; strict < deadline eligibility is owner-side.
            while not completed.empty():
                i,value,error=completed.get_nowait()
                thread,request=active.pop(i)
                observed=time.monotonic()
                if error is not None:
                    ledger.cancel(request.job['job_id'],str(error));raise error
                if not ledger.publish(request.job['job_id'],value,observed=observed):
                    value=request.expired()
                advance(i,value)
            now=time.monotonic()
            expired=[request for _,request in active.values() if now>=request.deadline]
            if expired:
                # The service deadline still decides eligibility. A cooperative
                # operation may need a bounded interval to acknowledge expiry.
                # Keep its slot charged and pause new work until it quiesces;
                # publish() then drops the late value through on_expired.
                if any(request.on_expired is None or now>=request.deadline+native_expiry_grace for request in expired):
                    raise TimeoutError('native/reference task deadline; external watchdog may be required')
                threading.Event().wait(.005)
                continue
            order=[(cursor+j)%len(generators) for j in range(len(generators))]
            for i in order:
                request=ready.get(i)
                if not isinstance(request,NativeRequest) or len(active)>=workers:
                    continue
                if time.monotonic()>=request.deadline:
                    ledger.cancel(request.job['job_id'],'queued native deadline',started=False)
                    ready.pop(i)
                    advance(i,request.expired())
                    continue
                thread=threading.Thread(target=run_native,args=(i,request),daemon=True)
                active[i]=(thread,request);ready.pop(i)
                thread.start();cursor=(i+1)%len(generators)
            # At most two batches of staged candidate records, tighter native bound wins.
            pending_native=sum(isinstance(r,NativeRequest) for r in ready.values())
            if pending_native<lookahead*batch_size and not (native_priority and (active or pending_native)):
                decode=[i for i in order if i in ready and not isinstance(ready[i],NativeRequest)]
                if decode:
                    first=ready[decode[0]]
                    group=[i for i in decode if ready[i].temperature==first.temperature and ready[i].memory.shape[1:]==first.memory.shape[1:]][:min(batch_size,lookahead*batch_size-pending_native)]
                    requests=[ready.pop(i) for i in group]
                    values=sample(decoder,requests)
                    if len(values)!=len(group):
                        raise RuntimeError('decoder row mapping mismatch')
                    for i,value in zip(group,values):
                        advance(i,value)
                    continue
            if active:
                # Wait for one short tick without busy-spinning or holding a mutable callback.
                threading.Event().wait(.005)
            elif ready:
                raise RuntimeError('scheduler blocked without active work')
    finally:
        # Drain/cancel within one total five-second envelope. Daemon threads cannot
        # confer clean checkpoint authority if they fail to acknowledge completion.
        deadline=time.monotonic()+5
        for request in ready.values():
            if isinstance(request,NativeRequest):
                ledger.cancel(request.job['job_id'],'cancelled before dispatch',started=False)
        for thread,request in active.values():
            if request.cancel is not None:
                request.cancel()
        for i,(thread,request) in list(active.items()):
            ledger.cancel(request.job['job_id'])
            thread.join(timeout=max(0,deadline-time.monotonic()))
        if any(thread.is_alive() for thread,_ in active.values()):
            from oeis_learn.synthesis.worker_pool import NonQuiescentWorker
            raise NonQuiescentWorker('nonquiescent reference/native task; terminate role via external watchdog')
    return results,ledger.boundary(policies or {})
