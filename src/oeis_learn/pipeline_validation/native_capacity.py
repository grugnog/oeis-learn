"""Finite CPU-only native slot-wave experiments; no production/research authority."""
import errno
import json
import statistics
import time
import threading
from .identity import seal,write,validate,number,file_hash
from .receipts import Ledger

BASELINES=['cold','warmup','calibration-0','calibration-1','calibration-2']
SLOTS=['queued-diagnostic',*BASELINES,
    *[f'w{w}-{name}' for w in (2,4) for name in ('warmup','pair0-A','pair0-B','pair1-B','pair1-A')],
    'observed-w1','observed-w4']
RULE='two ordinary AB/BA pairs; both gains>=5%; fastest eligible median, fewer workers within5%; no adoption'
MEANING='fixed18-row native CPU capacity screen; no production/default/pipeline/research/learning admission'


class NativeEvidence:
    """Durable plans and clock events survive failure before trial serialization."""
    def __init__(self,output,corpus,ledger,timing):
        self.output=output;self.corpus=corpus;self.ledger=ledger;self.timing=timing;self.lock=threading.Lock()
    def __enter__(self):
        write(self.output/'job-plan.json',seal(dict(version='pipeline-native-job-plan/v1',corpus=self.corpus['identity'],
            rows=[r['id'] for r in self.corpus['rows']],admission='slot-waves/v1',service_seconds=4)))
        return self
    def event(self,kind,row,**data):
        with self.lock:
            with (self.output/'job-progress.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event=kind,row=row,**data),sort_keys=True,allow_nan=False)+'\n')
    def __exit__(self,typ,value,tb):
        records=self.ledger.records
        by_id={r['job']['prompt_id']:r for r in records.values()}
        write(self.output/'job-status.json',seal(dict(version='pipeline-native-job-status/v1',
            corpus=self.corpus['identity'],clean_exit=typ is None,error=None if typ is None else typ.__name__,
            rows={r['id']:dict(state=by_id[r['id']]['state'],record=by_id[r['id']],clock=self.timing.get(r['id']))
                if r['id'] in by_id else dict(state='UNSTARTED',record=None,clock=None) for r in self.corpus['rows']})))


def semantics(trial,corpus):
    if trial!=seal(trial) or not trial.get('complete') or trial.get('admission')!='slot-waves/v1':
        raise ValueError('complete immutable slot-wave trial required')
    if trial.get('corpus')!=corpus['identity'] or trial.get('failure') is not None:
        raise ValueError('full frozen corpus required')
    rows=trial['results']
    if len(rows)!=18 or [r['id'] for r in rows]!=[r['id'] for r in corpus['rows']] or not all(r['complete'] for r in rows):
        raise ValueError('all18 frozen rows must complete; no selected subset')
    if trial['boundary']['active'] or trial['boundary']['reserved']:raise ValueError('native work must drain')
    number(trial['primary_seconds'],positive=True)
    jobs=trial['jobs'];timing=trial['timing'];ids={r['id'] for r in rows}
    if (len(jobs)!=18 or set(timing)!=ids or trial['boundary']['consumed']!=18
            or sorted(jobs)!=trial['boundary']['frontier']):raise ValueError('exact18-job terminal debit ledger required')
    for job in jobs.values():
        clock=timing[job['job']['prompt_id']]
        if (job['state']!='COMMITTED' or job['consumed']!=1 or job['reserved']!=1
                or job['deadline']!=clock['deadline'] or job['observed']>=job['deadline']
                or clock['deadline']>min(trial['parent_deadline'],clock['admitted']+4)
                or not trial['started']<=clock['admitted']<=clock['dispatched']<=clock['native_completed']<=job['observed']
                or job['observed']>trial['started']+trial['primary_seconds']):
            raise ValueError('original admission clock, owner eligibility and complete debit required')
    return [dict(id=r['id'],stratum=r['stratum'],outcome=r['outcome'],admitted=r['admitted']) for r in rows]


def observed_trial(function,output,**kwargs):
    from oeis_learn.tracking.pipeline_profile import Recorder
    with Recorder(output/'observations',instrumented=True):return function(output,**kwargs)


def comparison(child,receipts,workers,corpus):
    expected=child['semantics'];rows=[];missing=[]
    keys=[f'w{workers}-{n}' for n in ('warmup','pair0-A','pair0-B','pair1-B','pair1-A')]
    for key in keys:
        try:
            r=receipts[key]
            if r['state']!='PASS' or r!=seal(r):raise ValueError('slot incomplete')
            trial=r['observations']['result']
            if trial.get('workers')!=(1 if key.endswith('-A') else workers):raise ValueError('worker/slot mismatch')
            if semantics(trial,corpus)!=expected:raise ValueError('baseline semantics differ')
        except (ValueError,KeyError,TypeError) as exc:missing.append(key+': '+str(exc))
    if not missing:
        for pair,order in enumerate(('AB','BA')):
            a=receipts[f'w{workers}-pair{pair}-A']['observations']['result']['primary_seconds']
            b=receipts[f'w{workers}-pair{pair}-B']['observations']['result']['primary_seconds']
            rows.append(dict(pair=pair,order=order,baseline_seconds=a,candidate_seconds=b,gain=1-b/a,
                baseline_rows_per_second=18/a,candidate_rows_per_second=18/b))
    return seal(dict(version='pipeline-native-capacity-comparison/v1',workers=workers,child=child['identity'],
        status='INCONCLUSIVE' if missing else 'COMPLETE',missing=missing,pairs=rows,
        supported=bool(rows) and min(r['gain'] for r in rows)>=.05-1e-12,
        candidate_median=statistics.median(r['candidate_seconds'] for r in rows) if rows else None,
        optimization_adoption=False,research_cost_admission=False))


def freeze_child(config,corpus,receipts):
    from .comparisons import calibrate
    if any(receipts[k]!=seal(receipts[k]) for k in BASELINES):raise ValueError('immutable baseline receipts required')
    baseline=[receipts[k]['observations']['result'] for k in BASELINES]
    expected=semantics(baseline[0],corpus)
    if any(receipts[k]['state']!='PASS' or semantics(r,corpus)!=expected for k,r in zip(BASELINES,baseline)):
        raise ValueError('complete equal-semantic baseline roster required')
    calibration=calibrate([dict(complete=True,seconds=r['primary_seconds'],identity=receipts[k]['identity'])
        for k,r in zip(BASELINES[2:],baseline[2:])],minimum=30,cap=270,parent=config['contract'],workload=corpus['identity'])
    return seal(dict(version='pipeline-native-capacity-child/v1',source=config['source'],parent=config['contract'],
        corpus=corpus['identity'],semantics=expected,baselines={k:receipts[k]['identity'] for k in BASELINES},
        calibration=calibration,workers=[2,4],slots=SLOTS[6:],memory_ceiling=config['options']['memory_ceiling'],
        admission='slot-waves/v1',service_seconds=4,primary_measurement='ordinary/v1',decision_rule=RULE,
        interpretation=MEANING))


def investigate(inputs,output,config):
    from .role import native_trial,native_corpus
    from .runner import public_files,ResourceBreach
    from oeis_learn.synthesis.worker_pool import NonQuiescentWorker
    options=config['options']
    if set(options)!={'seconds','memory_ceiling'} or options['seconds']!=4470:
        raise ValueError('closed finite native capacity options required')
    number(options['memory_ceiling'],positive=True)
    output.mkdir(parents=True,exist_ok=True)
    corpus=native_corpus();write(output/'workload.json',corpus)
    binding=dict(source=config['source'],contract=config['contract'],corpus=corpus['identity'])
    ledgers={k:Ledger(output/'slots'/k,k,binding) for k in SLOTS};receipts={}
    child=None;failure=None;fatal=None;reports=[];deadline=time.monotonic()+options['seconds']
    def run(key,workers,seconds,*,queued=False,observed=False):
        if time.monotonic()+seconds+60>deadline:raise TimeoutError('whole native screen cannot fit next trial and reporting')
        ledger=ledgers[key];ledger.transition('READY');ledger.transition('RUNNING')
        print('NATIVE CAPACITY '+key+'; output '+str(ledger.root),flush=True)
        try:
            fn=observed_trial if observed else lambda function,*a,**k:function(*a,**k)
            trial=fn(native_trial,ledger.root,workers=workers,corpus=corpus,contract=config['contract'],cap=seconds,
                admission='queued8/v1' if queued else 'slot-waves/v1')
            # The old diagnostic may end incompletely without preventing the new
            # named admission hypothesis; all scored slots require complete work.
            state='PASS' if queued or trial.get('complete') else 'INCONCLUSIVE'
            receipts[key]=ledger.transition(state,files=public_files(ledger.root),observations={'result':trial})
            return trial
        except BaseException as exc:
            receipts[key]=ledger.transition('PARTIAL' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'FAIL',
                reason=f'{type(exc).__name__}: {exc}',files=public_files(ledger.root))
            raise
    try:
        run('queued-diagnostic',1,180,queued=True)
        for key in BASELINES:run(key,1,180)
        child=freeze_child(config,corpus,receipts)
        path=output/'comparison-contract.json';write(path,child);expected=file_hash(path)
        seconds=child['calibration']['reservation_seconds']
        if 12*seconds+60>deadline-time.monotonic():raise TimeoutError('finite candidate and observation reservations cannot fit')
        for key in SLOTS[6:]:
            if file_hash(path)!=expected:raise ValueError('prospective native child changed')
            workers=int(key[-1]) if key.startswith('observed') else 1 if key.endswith('-A') else int(key[1])
            run(key,workers,seconds,observed=key.startswith('observed'))
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
        if isinstance(exc,(ResourceBreach,MemoryError,KeyboardInterrupt,NonQuiescentWorker)) or isinstance(exc,OSError) and exc.errno in {errno.ENOSPC,errno.EDQUOT}:fatal=exc
    finally:
        for key,ledger in ledgers.items():
            if ledger.read()['state']=='PENDING':receipts[key]=ledger.transition('BLOCKED',reason=failure or 'baseline/child unavailable')
            else:receipts.setdefault(key,ledger.read())
        if child:reports=[comparison(child,receipts,w,corpus) for w in (2,4)]
        eligible={c['workers']:c['candidate_median'] for c in reports if c['supported']}
        if child:
            eligible[1]=statistics.median(receipts[k]['observations']['result']['primary_seconds'] for k in BASELINES[2:])
        selected=min(w for w,t in eligible.items() if t<=min(eligible.values())*1.05) if eligible else 1
        result=seal(dict(version='pipeline-native-capacity-report/v1',status='COMPLETE' if child and not failure and all(c['status']=='COMPLETE' for c in reports) and all(r['state']=='PASS' for r in receipts.values()) else 'INCONCLUSIVE',
            source=config['source'],parent=config['contract'],corpus=corpus['identity'],child=child['identity'] if child else None,
            selected_workers=selected,comparisons=reports,slots={k:dict(state=r['state'],receipt=r['identity'],reason=r.get('reason')) for k,r in receipts.items()},
            failure=failure,interpretation=MEANING,optimization_adoption=False,research_cost_admission=False,
            observation_policy='ordinary scored trials; observed-only1/4 follow-ups excluded; CPU telemetry overhead uncalibrated'))
        write(output/'native-capacity.json',result)
    if fatal is not None:raise fatal
    return result


class NativeCapacityStages:
    def __init__(self,adapters):self.a=adapters

    def __call__(self,stage,output):
        a=self.a
        if stage.id=='native.capacity.report':
            receipt=a.receipts.get('native.capacity',{})
            report=seal(dict(version='pipeline-native-capacity-disposition/v1',state=receipt.get('state','UNSTARTED'),
                reason=receipt.get('reason','not dispatched'),result=receipt.get('observations',{}).get('result'),
                slots={k:dict(state='UNSTARTED',reason=receipt.get('reason','not dispatched')) for k in SLOTS} if not receipt.get('observations',{}).get('result') else receipt['observations']['result'].get('slots',{}),
                optimization_adoption=False,research_cost_admission=False,interpretation=MEANING))
            write(output/'disposition.json',report);return report
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        from .identity import digest
        from .benchmarks import legacy
        from .runner import ResourceBreach
        host=a.host(a.args.engine);inputs=output/'input';inputs.mkdir()
        from .resources import observe,reserve
        policy=dict(host.policy['pipeline_resources'],available_bytes=observe()['available'])
        write(output/'resource-reservation.json',reserve(policy,8<<30))
        options=dict(seconds=4470,memory_ceiling=host.policy['memory_bytes'])
        write(inputs/'expected.json',dict(version='pipeline-role/v1',source=digest(collect_snapshot(a.root)),
            contract=a.contract['identity'],mode='native-capacity',options=options))
        image=a.images.images[(a.args.engine,'qualification')]
        legacy(a.root).container_stage(host,image,inputs,output/'role',mode='native-capacity',seconds=4500,gpu=False,
            argv=['python','-m','oeis_learn.pipeline_validation.role','native-capacity'])
        result=json.loads((output/'role/summary.json').read_text())
        if result.get('status')=='FAIL':
            if result.get('fatal'):raise ResourceBreach(result.get('error','native role containment failure'))
            raise RuntimeError(result.get('error','native capacity role failed'))
        if result['status']!='COMPLETE':return dict(result,stage_state='INCONCLUSIVE')
        return result
