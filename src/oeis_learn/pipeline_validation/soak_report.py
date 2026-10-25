"""Independent engineering verdicts; a soak never becomes research authority."""
import json
import statistics
from pathlib import Path
from .identity import seal,number
from .model_sizes import EXECUTION,PROFILES,parameters


def window_review(states):
    legs=[]
    for leg in (0,1):
        windows=[w for w in states if w['leg']==leg and 1<=w['ordinal']<=10]
        windows.sort(key=lambda w:w['ordinal'])
        missing=[]
        if len(windows)!=10 or len({w['ordinal'] for w in windows})!=10:missing.append('ten complete steady windows required')
        rates=[];lengths=[]
        for w in windows:
            train=w['components'].get('training',{})
            try:
                if not w.get('complete') or w['stop']-w['start']<299:raise ValueError('complete300s window required')
                rates.append(number(train['tokens'],positive=True)/number(train['seconds'],positive=True))
                lengths.append(train['tokens']/number(train['examples'],positive=True))
            except (KeyError,ValueError,TypeError):missing.append('complete actual training work required')
        drift=None
        if not missing:
            if max(lengths)>1.05*min(lengths):missing.append('window length mixtures do not match within5%')
            else:drift=1-statistics.median(rates[-2:])/statistics.median(rates[:2])
        legs.append(dict(leg=leg,windows=len(windows),rates=rates,mean_target_lengths=lengths,
            drift=drift,investigate_five_percent_decline=drift is not None and drift>=.05,
            missing=missing,verdict='INCONCLUSIVE' if missing else 'PASS' if drift<=.10 else 'FAIL'))
    return legs


def resource_review(root,windows,live_bound):
    verdicts=[]
    for leg,path in [(0,'soak.persistent/leg-0/resource-samples.jsonl'),(1,'soak.persistent/role/resource-samples.jsonl')]:
        try:samples=[json.loads(x) for x in (Path(root)/path).read_text().splitlines()]
        except (OSError,ValueError):
            verdicts.append(dict(leg=leg,verdict='INCONCLUSIVE',reason='mandatory resource samples missing'));continue
        selected=[w for w in windows if w['leg']==leg and 1<=w['ordinal']<=10]
        complete=[]
        for w in selected:
            values=[s for s in samples if w['start']<=s['time']<w['planned_stop']]
            if len(values)<240 or any(any(type(s.get(k)) is not int for k in ('anon','file','current','driver','cpu_usage_usec')) for s in values):
                complete.append(False)
            else:complete.append(True)
        steady=[s for s in samples if any(w['start']<=s['time']<w['planned_stop'] for w in selected)]
        known=all(complete) and len(complete)==10
        peak=max((s.get('current',0)+s.get('driver',0) for s in steady if s.get('current') is not None and s.get('driver') is not None),default=0)
        first=steady[:600];last=steady[-600:]
        growth={k:(statistics.median(s[k] for s in last)-statistics.median(s[k] for s in first)) if known else None for k in ('anon','file','driver')}
        gpu=[s['gpu_busy_percent'] for s in steady if s.get('gpu_busy_percent') is not None]
        cpu=(steady[-1]['cpu_usage_usec']-steady[0]['cpu_usage_usec'])/1e6/(steady[-1]['time']-steady[0]['time']) if known and len(steady)>1 else None
        verdicts.append(dict(leg=leg,verdict='INCONCLUSIVE' if not known else 'PASS' if peak<=live_bound else 'FAIL',
            steady_samples=len(steady),role_upper_peak_bytes=peak,registered_live_upper_bytes=live_bound,
            category_growth_bytes=growth,gpu_busy_percent_mean=statistics.mean(gpu) if gpu else None,cpu_equivalents=cpu,
            interpretation='cgroup and driver are conservative overlapping domains; file cache and growth are reported, not a leak-freedom claim'))
    return verdicts


def report(receipts,*,root=None):
    def result(key):
        r=receipts.get(key,{})
        if r!=seal(r) or r.get('state')!='PASS':return None
        return r.get('observations',{}).get('result')
    persistent=result('soak.persistent');parity=result('soak.parity');joint=result('soak.joint');scorer=result('soak.scorer')
    missing=[];windows=[];legs=[];resources=[];choice=None
    if not parity:missing.append('current full-size CPU/HIP parity')
    if not joint:missing.append('joint full-size stress and complete-state recovery fixture')
    if not persistent:missing.append('two complete persistent work windows')
    if persistent:
        choice=persistent['choice']
        if choice!=seal(choice):raise ValueError('immutable choice required')
        trials=persistent['results']
        if len(trials)!=2:missing.append('two persistent results required')
        else:
            for i,r in enumerate(trials):
                eng=r.get('trainer',{}).get('engineering',{})
                if (r.get('execution')!=EXECUTION or r.get('resources',{}).get('qualified') is not True
                    or r.get('profile')!=PROFILES['m100'] or not eng.get('complete') or eng.get('window')!=i
                    or r.get('trainer',{}).get('counts',{}).get('actual')!=parameters(PROFILES['m100'])):
                    missing.append('actual full-size source/device/resource/window audit differs')
                if eng.get('examples',0)<choice['data_counts']['training']:
                    missing.append('complete training-partition traversal required in each leg')
                if any(w['start']<persistent['lease']['started'] or w['planned_stop']>persistent['lease']['expires']
                    for w in eng.get('state',{}).get('windows',[])):
                    missing.append('work windows cannot precede/renew their immutable lease')
            state=trials[-1]['trainer']['engineering']['state'];windows=state['windows'];legs=window_review(windows)
            for leg in (0,1):
                if not any(p['leg']==leg and p['state']=='PASS' for p in state.get('pulse_slots',[])):
                    missing.append('native coexistence pulse not exercised in leg'+str(leg))
            if root is not None:resources=resource_review(root,windows,choice['live_memory_upper_bytes'])
            else:missing.append('resource sample audit not available')
    sustained='INCONCLUSIVE' if missing or not legs or not resources or any(r['verdict']=='INCONCLUSIVE' for r in [*legs,*resources]) else 'FAIL' if any(r['verdict']=='FAIL' for r in [*legs,*resources]) else 'PASS'
    current={k:dict(state=v['state'],reason=v.get('reason',''),receipt=v.get('identity')) for k,v in receipts.items()}
    engines={k.split('.')[1] for k in receipts if k.startswith('engine.')}
    portability='PASS' if engines=={'docker','podman'} and all(receipts.get(k,{}).get('state') in {'PASS','REUSED'} for e in engines
        for k in ('cpu.'+e+'.semantic','operations.'+e+'.workflow','operations.'+e+'.retirement','soak.operations.'+e)) else 'PENDING'
    totals=persistent['results'][-1]['trainer']['engineering']['state']['components'] if persistent and len(persistent.get('results',[]))==2 else {}
    generated=totals.get('generation_native',{});elapsed=persistent.get('elapsed_seconds') if persistent else None
    rates={k:generated.get(k,0)*60/elapsed if elapsed else None for k in
        ('scheduled','started','unstarted','decoded','emitted_tokens','native_started','native_complete','visible_verified')}
    return seal(dict(version='pipeline-soak-report/v1',sustained_operation=sustained,
        correctness_recovery='PASS' if parity and joint and scorer else 'PENDING',portability=portability,
        primary_engineering_readiness='PASS' if sustained=='PASS' and parity and joint and scorer else 'FAIL' if sustained=='FAIL' else 'PENDING',
        isolated_cpu_scorer='PASS' if scorer else 'PENDING',
        missing=missing,choice=choice,window_reviews=legs,resource_reviews=resources,stages=current,
        operation_totals=totals,verification_scope='visible20; no continuation/final claims',
        generated_work_per_whole_soak_minute=rates,
        natural_generation_to_native='EXERCISED' if generated.get('native_started',0) else 'NOT_EXERCISED',
        diagnostic_native_load='separately labelled health pulses; excluded from generated-work rates',
        native_pulse_slots=persistent['results'][-1]['trainer']['engineering']['state'].get('pulse_slots',[]) if persistent else [],
        whole_soak_seconds=persistent.get('elapsed_seconds') if persistent else None,
        optimization_adoption='INCONCLUSIVE',supported_default='RETAIN_BASELINE',
        confirmation=dict(eligible=False,reason='original complete-work/mandatory-strata/fixed-budget baseline is not established by disposable size/batch work',
            slots=[dict(scope=s,pair=i,order=o,arm=a,state='UNSTARTED') for s in ('cold','warm') for i,o in enumerate(('AB','BA','AB','BA','AB')) for a in o]),
        costs=dict(diverse32='UNKNOWN_COMPLETE_SCHEDULE',transfer128_128='UNKNOWN_COMPLETE_SCHEDULE',
            within_eighty_percent=False,reason='training/soak observations do not measure every original control and independent hidden-scoring schedule'),
        learning='NOT_ADMITTED',research='HELD',final_access='SEALED',long_run_admission='NOT_ADMITTED',
        historical={'008:T093':'DEFERRED','008:T096':'NEGATIVE 7730/14400 seconds'},
        interpretation='full-scale engineering investigation; useful generated-program counts and diagnostic native probes remain separate; no learned-quality claim'))
