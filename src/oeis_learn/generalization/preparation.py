"""Ordered bounded source admission shared by feasibility and native preparation."""
import multiprocessing
import time
from itertools import islice
from .contracts import digest
from .cohorts import choose,CohortChoice
from .coverage import materialize_observed
from .exclusions import ExclusionIndex
from .resources import PREPARATION_SECONDS
from oeis_learn.pipeline_validation.identity import seal


def reference_candidate(row):
    from oeis_learn.data.bigint_programs import canonical_family
    from .targets import ground
    try:
        if row['origin']!='generic_random':raise ValueError('nonmechanical origin')
        if canonical_family(row['parameter_source'])!=row['family']:raise ValueError('descriptor family mismatch')
        if ground(row['parameter_source'],row.get('assignment',{}))!=row['source']:raise ValueError('concrete descriptor grounding mismatch')
        row=dict(row,variant=str(row['variant']))
        row.setdefault('group',digest([row['family'],row['variant']]))
        return materialize_observed(row),None
    except ValueError as exc:return None,str(exc)


def _worker(pipe,function):
    try:
        while True:
            item=pipe.recv()
            if item is None:return
            try:pipe.send((True,[function(row) for row in item]))
            except Exception as exc:pipe.send((False,type(exc).__name__))
    except (EOFError,BrokenPipeError):pass
    finally:pipe.close()


class OrderedWorkers:
    """At most64 rows in flight; caller owns each child and its bounded cleanup.

    Results commit in input order, regardless of completion order. Native callbacks
    stay in the parent, with one client and the existing externally bounded deadline.
    No process pool internals, implicit unbounded queues, or hidden retries.
    """
    def __init__(self,count,*,function=reference_candidate):
        if type(count) is not int or not 1<=count<=4:raise ValueError('one to four reference workers required')
        self.count=count;self.function=function;self.children=[]

    def __enter__(self):
        if self.count>1:
            try:
                ctx=multiprocessing.get_context('spawn')
                for _ in range(self.count):
                    parent,child=ctx.Pipe();process=ctx.Process(target=_worker,args=(child,self.function),daemon=True)
                    process.start();child.close();self.children.append((process,parent))
            except BaseException:
                self.__exit__(None,None,None);raise
        return self

    def map(self,rows,*,deadline):
        if len(rows)>64:raise ValueError('bounded reference batch required')
        if time.monotonic()>=deadline:raise TimeoutError('reference deadline exhausted')
        if not self.children:
            result=[]
            for row in rows:
                if time.monotonic()>=deadline:raise TimeoutError('reference deadline exhausted')
                result.append(self.function(row))
            return result
        chunks=[rows[i::self.count] for i in range(self.count)]
        for (_,pipe),chunk in zip(self.children,chunks):pipe.send(chunk)
        result=[None]*len(rows)
        for i,(_,pipe) in enumerate(self.children):
            remaining=deadline-time.monotonic()
            if remaining<=0 or not pipe.poll(remaining):raise TimeoutError('reference worker deadline exhausted')
            try:ok,values=pipe.recv()
            except EOFError as exc:raise RuntimeError('reference worker exited') from exc
            if not ok:raise RuntimeError('reference worker failed: '+values)
            for j,value in enumerate(values):result[i+j*self.count]=value
        return result

    def __exit__(self,*args):
        for process,pipe in self.children:
            pipe.close()
            if process.is_alive():process.terminate()
        deadline=time.monotonic()+1
        for process,_ in self.children:process.join(max(0,deadline-time.monotonic()))
        for process,_ in self.children:
            if process.is_alive():process.kill()
        survivors=[]
        for process,pipe in self.children:
            process.join(.2)
            if process.is_alive():survivors.append((process,pipe))
            else:process.close()
        self.children=survivors
        if survivors:raise RuntimeError('reference worker exit unacknowledged; external termination required')


def exclusion_record(row,role,holdout):
    return dict(id=row['id'],role=role,holdout=holdout,
        **{key:row[key] for key in ('family','source','values','group','variant')})


def scan(candidates,*,role,holdout,count,quotas,structural,execute,exclusions=None,
         draw_cap=1_000_000,seconds=21600,length_boundary=32,unique_sources=False,
         seen_sources=None,workers=1,progress=None,observer=None):
    if role not in {'training','selection','assessment'} or holdout not in {'parameter','structural'}:raise ValueError('invalid cohort role/holdout')
    if type(length_boundary) is not int or not 1<=length_boundary<=4096:raise ValueError('bounded frozen length bin required')
    if not 0<seconds<=PREPARATION_SECONDS or not 1<=draw_cap<=1_000_000:raise ValueError('preparation limits')
    exclusions=exclusions if exclusions is not None else ExclusionIndex()
    seen_sources=set(seen_sources or ())|{r[4] for r in exclusions.records if r[1]==role}
    began=time.monotonic();deadline=began+seconds;admitted=[];rejections=[];seen=0;expired=False;native_failed=False
    timings=dict(reference_seconds=0.0,native_seconds=0.0,selection_seconds=0.0);native_rows=0
    def observe(phase):
        if observer:observer(dict(phase=phase,draws=seen,admitted=len(admitted),native_rows=native_rows,
            elapsed_seconds=time.monotonic()-began,**timings))
    observe('reference')
    iterator=iter(candidates)
    with OrderedWorkers(workers) as pool:
        while seen<draw_cap:
            batch=list(islice(iterator,min(64,draw_cap-seen)))
            if not batch:break
            started=time.monotonic()
            try:results=pool.map(batch,deadline=deadline)
            except TimeoutError:expired=True;break
            finally:timings['reference_seconds']+=time.monotonic()-started
            pending=[]
            for row,(result,error) in zip(batch,results):
                if time.monotonic()>=deadline:expired=True;break
                seen+=1
                try:
                    if unique_sources:
                        from .refresh import claim_source
                        claim_source(row['source'],seen_sources)
                    if error:raise ValueError(error)
                    exact,observation=result
                    reasons=exclusions.reasons(exclusion_record(exact,role,holdout))
                    if reasons:
                        rejections.append(dict(draw=seen,reason='cross-role exclusion',rules=list(reasons)))
                        continue
                    tags=['family:'+row['family'],row['stage'],'long' if observation.nodes>=length_boundary else 'short','above64' if observation.above_64_bits else 'within64']
                    if observation.loop_active:tags.append('loop_active')
                    if observation.branch_active:tags.append('branch_active')
                    pending.append((seen,dict(exact,role=role,holdout=holdout,tags=tags)))
                except ValueError as exc:rejections.append(dict(draw=seen,reason=str(exc)))
            for start in range(0,len(pending),8):
                chunk=pending[start:start+8]
                if time.monotonic()>=deadline:expired=True;break
                observe('native' if execute is not None else 'reference')
                started=time.monotonic()
                try:
                    if execute is None:native=[None]*len(chunk)
                    elif hasattr(execute,'batch'):native=execute.batch([row for _,row in chunk])
                    else:native=[execute(row['source'],tuple(row['indices'])) for _,row in chunk]
                finally:
                    if execute is not None:timings['native_seconds']+=time.monotonic()-started
                if execute is not None:native_rows+=len(native)
                if len(native)!=len(chunk):raise ValueError('incomplete native batch response')
                for (draw,row),actual in zip(chunk,native):
                    if execute is not None and (actual.get('outcome')!='ok' or tuple(actual.get('values',()))!=tuple(row['values'])):
                        native_failed=True
                        rejections.append(dict(draw=draw,reason='native/reference disagreement'))
                    else:admitted.append(row)
                if time.monotonic()>=deadline:expired=True;break
            if progress and seen%1024==0:progress(seen,len(admitted))
            observe('reference')
            if expired or native_failed:break
    # No cohort can publish after exhaustion. Do not spend the reporting reserve
    # sorting/encoding a large partial cohort that cannot be used.
    observe('selection');started=time.monotonic()
    choice=CohortChoice(b'[]',('preparation_deadline',)) if expired else choose(admitted,count=count,quotas=quotas,structural=structural,draw_cap=draw_cap)
    timings['selection_seconds']+=time.monotonic()-started
    if native_failed:choice=CohortChoice(choice.payload,choice.shortfalls+('native_validation_failed',))
    ledger=seal(dict(version='generalization-cell-preparation/v1',role=role,draws=seen,rejections=sorted(rejections,key=lambda row:row['draw']),
        admitted=len(admitted),shortfalls=list(choice.shortfalls),
        status=('PASS' if execute is not None else 'FEASIBLE') if choice.complete else 'BLOCKED',
        native_validation='PERFORMED' if execute is not None else 'NOT_RUN'))
    observe('complete' if choice.complete else 'blocked')
    return choice,ledger


def pac_exact_row(row,execute):
    """Ground independently, then compare every native term before producing metadata."""
    from .targets import paired_learning_view
    from oeis_learn.data.bigint_programs import canonical_family
    fields={'id','role','family','source','parameter_source','assignment','indices'}
    if set(row)!=fields or row['role']!='training':raise ValueError('closed training-only PAC source row required')
    if row['family']!=canonical_family(row['parameter_source']):raise ValueError('PAC family source hash mismatch')
    if row['indices']!=list(map(str,range(100))):raise ValueError('PAC exact source indices required')
    pair=paired_learning_view(row['source'],row['parameter_source'],row['assignment'],indices=tuple(row['indices']),sharing='explicit')
    values=pair['verification']['values'];native=execute(row['source'],row['indices'])
    if native.get('outcome')!='ok' or native.get('values')!=values:raise ValueError('PAC native/reference disagreement')
    return pair,values
