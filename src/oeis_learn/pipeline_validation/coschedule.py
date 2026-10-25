"""Finite GPU/native coexistence in one quota, one GPU owner and one archive."""
import errno
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import threading
import time
from .identity import seal,validate,digest,number,write,file_hash
from .receipts import Ledger

BASIS=seal(dict(version='pipeline-coschedule-basis/v1',gpu_archive='sha256:d86b1a4499159acc72c78e546868bf6b33d4915d927f6484ec6010dc6b9558c5',
    native_archive='sha256:5449f91f92665a6be42060ab3521f21ecf6363e2888f725a76e2f93de70dc7b2',gpu_batch=128,native_workers=4))
BASELINES=['cold','warmup','calibration-0','calibration-1','calibration-2']
COMPARE=['candidate-warmup','pair0-A','pair0-B','pair1-B','pair1-A']
REPEATS=[f'repeat-{i:02d}' for i in range(10)]
STUDY_SLOTS=BASELINES+COMPARE+REPEATS
NATIVE_CYCLES=8
MEANING='component co-scheduling and shared-controller repeatability; no generated-program/pipeline/default/research/learning admission'
RULE='two complete ordinary AB/BA gains>=5% selects overlap, otherwise serial; incomplete blocks10 observed repeats'
REF_FIELDS={'version','source','contract','corpus','semantics','trial','cycle_seconds','replay_reference','weights','basis'}
CHILD_FIELDS={'version','source','contract','reference','native_reference','weights','workload','baseline_receipts','calibration','resource_device','threads','slots','basis','rule','interpretation'}


def work_identity(gpu_reference,native_reference):
    return digest(dict(version='pipeline-coschedule-work/v1',gpu_reference=gpu_reference,native_reference=native_reference,
        gpu_rows=128,unique_visible_rows=32,supplied_tokens=131072,boundaries=640,native_cycles=8,native_rows=144,workers=4,batch=128))


def read_reference(inputs,config):
    ref=json.loads((inputs/'native-reference.json').read_text());validate(ref,REF_FIELDS,'pipeline-coschedule-native-reference/v1')
    from .role import native_corpus
    gpu=json.loads((inputs/'reference.json').read_text())
    if (ref['source']!=config['source'] or ref['contract']!=config['contract'] or ref['basis']!=BASIS
            or ref['corpus']!=native_corpus()['identity'] or len(ref['semantics'])!=18 or ref['cycle_seconds']!=8
            or gpu!=seal(gpu) or ref['replay_reference']!=gpu['identity'] or ref['weights']!=gpu['weights']
            or file_hash(inputs/'weights.pt')!=ref['weights']):raise ValueError('frozen co-scheduling references/source/weights required')
    return ref


def reference_role(inputs,output,config):
    from .hardware_ladder import reference_role as gpu_reference
    from .role import native_trial,native_corpus
    from .native_capacity import semantics
    began=time.monotonic();result=gpu_reference(inputs,output,config)
    left=config['options']['seconds']-(time.monotonic()-began)
    if left<35:raise TimeoutError('reference cannot reserve native reference and export')
    corpus=native_corpus();native=native_trial(output/'native-reference-trial',workers=4,corpus=corpus,contract=config['contract'],cap=30,admission='slot-waves/v1')
    expected=semantics(native,corpus);cycle=math.ceil(max(8,2*native['primary_seconds']))
    if cycle>8:raise ValueError('native reference cannot fit fixed8s cycle reservation; no clamp')
    gpu=json.loads((output/'reference.json').read_text())
    ref=seal(dict(version='pipeline-coschedule-native-reference/v1',source=config['source'],contract=config['contract'],
        corpus=corpus['identity'],semantics=expected,trial=native['identity'],cycle_seconds=cycle,
        replay_reference=gpu['identity'],weights=gpu['weights'],basis=BASIS))
    write(output/'native-reference.json',ref)
    return dict(result,native_reference=ref['identity'],native_rows=18)


def gpu_role(inputs,output,config):
    from .hardware_ladder import numeric_role
    if config['options'].get('batch_size')!=128 or config['options'].get('measurement')!='ordinary/v1':
        raise ValueError('fixed ordinary GPU128 co-scheduling component required')
    return numeric_role(inputs,output,config)


class Cancellation:
    def __init__(self):self.stop=threading.Event();self.lock=threading.Lock();self.worker=None
    def bind(self,client):
        with self.lock:
            self.worker=client.worker
            if self.stop.is_set():
                self.worker.cancel_all();raise InterruptedError('co-scheduling peer stopped')
    def cancel(self,deadline=None):
        self.stop.set()
        with self.lock:
            if self.worker is not None:
                self.worker.cancel_all()
                if deadline is not None:
                    for slot in getattr(self.worker,'slots',[self.worker]):slot.cleanup_deadline=min(getattr(slot,'cleanup_deadline',float('inf')),deadline)


def plan_native(output,config,reference):
    return {i:Ledger(output/f'cycle-{i:02d}',f'cycle-{i:02d}',dict(source=config['source'],native_reference=reference['identity'])) for i in range(8)}


def finish_native(output,ledgers,failure=None):
    records={}
    for i,ledger in ledgers.items():
        if ledger.read()['state']=='PENDING':records[i]=ledger.transition('BLOCKED',reason=failure or 'native cycle not admitted')
        else:records[i]=ledger.read()
    result=seal(dict(version='pipeline-coschedule-native/v1',status='COMPLETE' if all(r['state']=='PASS' for r in records.values()) else 'PARTIAL',
        cycles=sum(r['state']=='PASS' for r in records.values()),rows=18*sum(r['state']=='PASS' for r in records.values()),
        cycle_receipts=[records[i]['identity'] for i in range(8)],failure=failure))
    path=output/'native-group.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old!=seal(old) or {k:v for k,v in old.items() if k not in {'identity','failure'}}!={k:v for k,v in result.items() if k not in {'identity','failure'}}:
            raise ValueError('terminal native group changed')
        return old
    write(path,result);return result


def native_run(inputs,output,config,reference,deadline,control,*,ledgers=None):
    from .role import native_trial,native_corpus
    from .native_capacity import semantics
    from .runner import public_files
    corpus=native_corpus();ledgers=ledgers if ledgers is not None else plan_native(output,config,reference)
    records={};failure=None
    try:
        for i,ledger in ledgers.items():
            if control.stop.is_set():raise InterruptedError('co-scheduling peer stopped before cycle admission')
            if time.monotonic()+8>deadline:raise TimeoutError('native cycle reservation cannot fit immutable trial parent')
            rows=[dict(r,id=f'c{i:02d}-'+r['id']) for r in corpus['rows']]
            cycle=seal(dict(corpus,rows=rows));ledger.transition('READY');ledger.transition('RUNNING')
            try:
                value=native_trial(ledger.root,workers=4,corpus=cycle,contract=config['contract'],cap=8,admission='slot-waves/v1',on_client=control.bind)
                semantic=semantics(value,cycle)
                semantic=[dict(r,id=r['id'][4:]) for r in semantic]
                if semantic!=reference['semantics'] or value['primary_seconds']>8:raise ValueError('complete native cycle parity/wall reservation required')
                records[i]=ledger.transition('PASS',files=public_files(ledger.root),observations={'result':value})
            except BaseException as exc:
                records[i]=ledger.transition('PARTIAL' if isinstance(exc,(TimeoutError,InterruptedError,KeyboardInterrupt)) else 'FAIL',reason=f'{type(exc).__name__}: {exc}',files=public_files(ledger.root));raise
    except BaseException as exc:failure=f'{type(exc).__name__}: {exc}';raise
    finally:
        result=finish_native(output,ledgers,failure)
    return result


def gpu_run(inputs,output,config,seconds,*,cleanup_state=None):
    from .supervision import launch
    output.mkdir(parents=True,exist_ok=True);target=output/'input';target.mkdir()
    for name in ('weights.pt','work.json','reference.json','reference.pt'):shutil.copyfile(inputs/name,target/name)
    cfg=dict(version='pipeline-role/v1',source=config['source'],contract=config['contract'],mode='coschedule-gpu',
        options=dict(seconds=240,memory_ceiling=config['options']['memory_ceiling'],render_node=config['options']['render_node'],batch_size=128,measurement='ordinary/v1'))
    write(target/'expected.json',cfg);write(output/'expected.json',cfg)
    try:
        launch([sys.executable,'-m','oeis_learn.pipeline_validation.role','coschedule-gpu','--input',str(target),'--output',str(output)],output,
            seconds=seconds,stream_cap=8<<20,grace_seconds=.25,cleanup_state=cleanup_state)
    except BaseException as exc:
        try:failed=json.loads((output/'summary.json').read_text())
        except (OSError,ValueError):failed={}
        if failed.get('fatal'):
            from .runner import ResourceBreach
            raise ResourceBreach(failed.get('error','GPU resource failure')) from exc
        raise
    result=json.loads((output/'summary.json').read_text())
    if result.get('fatal'):
        from .runner import ResourceBreach
        raise ResourceBreach(result.get('error','GPU resource failure'))
    if result.get('status')!='COMPLETE':raise RuntimeError('GPU child incomplete')
    return result


def trial(inputs,output,config,*,schedule,observed,seconds,gpu_executor=None,native_executor=None):
    from .runner import ResourceBreach
    from oeis_learn.synthesis.worker_pool import NonQuiescentWorker
    from oeis_learn.tracking.pipeline_profile import Recorder
    if schedule not in {'serial','overlap'} or type(observed) is not bool or not 400<=seconds<=570:
        raise ValueError('registered co-scheduling trial required')
    output.mkdir(parents=True,exist_ok=True);ref=read_reference(inputs,config);deadline=time.monotonic()+seconds
    control=Cancellation();native={};gpu=None;thread=None;failure=None;times={};began=time.monotonic()
    ledgers=plan_native(output/'native',config,ref)
    gpu_cleanup={};gpu_failed=False
    if gpu_executor is None:
        gpu_executor=lambda i,o,c,s:gpu_run(i,o,c,s,cleanup_state=gpu_cleanup)
    native_executor=native_executor or native_run
    write(output/'trial-plan.json',seal(dict(version='pipeline-coschedule-trial-plan/v1',schedule=schedule,observed=observed,
        source=config['source'],native_reference=ref['identity'],workload=work_identity(ref['replay_reference'],ref['identity']),native_cycles=8,parent_deadline=deadline)))
    def run_native():
        start=time.monotonic()
        try:native['result']=native_executor(inputs,output/'native',config,ref,deadline,control,ledgers=ledgers)
        except BaseException as exc:native['error']=exc
        finally:times['native']=time.monotonic()-start
    try:
        with Recorder(output/'observations',instrumented=observed,render_node=config['options']['render_node']):
            if schedule=='overlap':thread=threading.Thread(target=run_native,daemon=True);thread.start()
            start=time.monotonic()
            try:gpu=gpu_executor(inputs,output/'gpu',config,270)
            except BaseException:gpu_failed=True;raise
            times['gpu']=time.monotonic()-start
            if schedule=='serial':run_native()
            elif thread:
                while thread.is_alive():
                    if time.monotonic()>=deadline:raise TimeoutError('immutable co-scheduling trial parent expired')
                    thread.join(timeout=min(.1,max(0,deadline-time.monotonic())))
            if 'error' in native:raise native['error']
            if time.monotonic()>=deadline:raise TimeoutError('co-scheduling export cannot fit trial parent')
    except BaseException as exc:failure=exc
    finally:
        cleanup=gpu_cleanup.get('deadline',time.monotonic()+5) if gpu_failed else time.monotonic()+5
        try:control.cancel(cleanup)
        finally:
            if thread:thread.join(timeout=max(0,cleanup-time.monotonic()))
        write(output/'trial-progress.json',seal(dict(version='pipeline-coschedule-progress/v1',schedule=schedule,
            gpu_complete=gpu is not None,native_complete=native.get('result',{}).get('status')=='COMPLETE',
            peer_drained=not thread or not thread.is_alive(),component_seconds=times,error=None if failure is None else type(failure).__name__)))
        if thread and thread.is_alive():raise NonQuiescentWorker('native peer failed five-second shared drain; terminate role')
        terminal=finish_native(output/'native',ledgers,None if failure is None else type(failure).__name__)
    if failure is not None:raise failure
    n=native['result']
    if n.get('status')!='COMPLETE' or n.get('cycles')!=8 or n.get('rows')!=144 or n.get('cycle_receipts')!=terminal['cycle_receipts'] or terminal['status']!='COMPLETE':
        raise ValueError('all eight native cycle receipts must be complete and match')
    result=seal(dict(version='pipeline-coschedule-trial/v1',status='COMPLETE',source=config['source'],contract=config['contract'],
        reference=ref['replay_reference'],native_reference=ref['identity'],workload=work_identity(ref['replay_reference'],ref['identity']),
        schedule=schedule,observed=observed,gpu=gpu,native=n,component_seconds=times,gpu_memory=gpu.get('resources',{}),interpretation=MEANING))
    write(output/'trial.json',result)
    result=seal(dict(result,primary_seconds=time.monotonic()-began))
    if result['primary_seconds']>seconds:raise TimeoutError('whole co-scheduling primary interval overrun')
    write(output/'measurement.json',result);return result


def complete(result,config,reference,*,schedule,observed):
    if result!=seal(result) or result.get('status')!='COMPLETE':raise ValueError('complete immutable co-scheduling trial required')
    expected=dict(source=config['source'],contract=config['contract'],reference=reference['replay_reference'],native_reference=reference['identity'],
        workload=work_identity(reference['replay_reference'],reference['identity']),schedule=schedule,observed=observed)
    if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('co-scheduling mode/work/reference mismatch')
    gpu=result['gpu'];native=result['native'];memory=result['gpu_memory']
    if (gpu.get('status')!='COMPLETE' or gpu.get('boundaries')!=640 or gpu.get('reference')!=reference['replay_reference']
            or gpu.get('batch_size')!=128 or gpu.get('bookkeeping')!='hoisted/v1' or gpu.get('measurement_mode')!='ordinary/v1'
            or gpu.get('resources')!=memory or native.get('status')!='COMPLETE'
            or native.get('cycles')!=8 or native.get('rows')!=144 or len(native.get('cycle_receipts',[]))!=8
            or len(set(native['cycle_receipts']))!=8
            or memory.get('qualified') is not True or memory.get('ceiling_bytes')!=config['options']['memory_ceiling']
            or gpu.get('torch_threads')!={'intraop':2,'interop':1} or not gpu.get('device',{}).get('hip')):
        raise ValueError('complete shared-resource640/144 work, actual HIP/threads required')
    number(result['primary_seconds'],positive=True)
    return dict(ceiling_bytes=memory['ceiling_bytes'],device=gpu['device'])


def freeze(config,reference,receipts):
    from .comparisons import calibrate
    values=[];rows=[]
    for key in BASELINES:
        r=receipts[key]
        if r!=seal(r) or r['state']!='PASS':raise ValueError('all five immutable baseline receipts required')
        value=r['observations']['result'];values.append(complete(value,config,reference,schedule='serial',observed=False));rows.append(value)
    if any(v!=values[0] for v in values):raise ValueError('common resource/device required')
    cal=calibrate([dict(complete=True,seconds=r['primary_seconds'],identity=receipts[k]['identity']) for k,r in zip(BASELINES[2:],rows[2:])],
        minimum=400,cap=570,parent=config['contract'],workload=work_identity(reference['replay_reference'],reference['identity']))
    return seal(dict(version='pipeline-coschedule-child/v1',source=config['source'],contract=config['contract'],reference=reference['replay_reference'],
        native_reference=reference['identity'],weights=reference['weights'],workload=work_identity(reference['replay_reference'],reference['identity']),
        baseline_receipts={k:receipts[k]['identity'] for k in BASELINES},calibration=cal,resource_device=values[0],threads={'intraop':2,'interop':1},
        slots=COMPARE+REPEATS,basis=BASIS,rule=RULE,interpretation=MEANING))


def compare(child,receipts,config,reference):
    missing=[];pairs=[]
    if child is not None:
        validate(child,CHILD_FIELDS,'pipeline-coschedule-child/v1')
        if child['basis']!=BASIS or child['rule']!=RULE or child['slots']!=COMPARE+REPEATS:raise ValueError('registered child required')
    for key in COMPARE:
        try:
            if child is None:raise ValueError('comparison child unavailable')
            r=receipts[key]
            if r['state']!='PASS' or r!=seal(r):raise ValueError('trial incomplete')
            expected=complete(r['observations']['result'],config,reference,schedule='serial' if key.endswith('-A') else 'overlap',observed=False)
            if expected!=child['resource_device']:raise ValueError('resource/device differs')
        except (ValueError,KeyError,TypeError) as exc:missing.append(key+': '+str(exc))
    if not missing:
        for pair,order in enumerate(('AB','BA')):
            a=receipts[f'pair{pair}-A']['observations']['result']['primary_seconds'];b=receipts[f'pair{pair}-B']['observations']['result']['primary_seconds']
            pairs.append(dict(pair=pair,order=order,baseline_seconds=a,candidate_seconds=b,gain=1-b/a))
    return seal(dict(version='pipeline-coschedule-comparison/v1',status='INCONCLUSIVE' if missing else 'COMPLETE',child=child['identity'] if child else None,
        missing=missing,pairs=pairs,selected_schedule=None if missing else 'overlap' if min(p['gain'] for p in pairs)>=.05-1e-12 else 'serial',
        optimization_adoption=False,research_cost_admission=False))


def study(inputs,output,config):
    from .runner import public_files,ResourceBreach
    from oeis_learn.synthesis.worker_pool import NonQuiescentWorker
    options=config['options']
    if set(options)!={'seconds','memory_ceiling','render_node'} or options['seconds']!=12470:raise ValueError('closed finite co-scheduling options required')
    ref=read_reference(inputs,config);deadline=time.monotonic()+12470
    ledgers={k:Ledger(output/'trials'/k,k,dict(source=config['source'],contract=config['contract'],native_reference=ref['identity'])) for k in STUDY_SLOTS}
    receipts={};child=None;decision=None;failure=None;fatal=None
    def run(key,schedule,observed,seconds):
        if time.monotonic()+seconds+60>deadline:raise TimeoutError('whole study cannot fit next trial and report')
        ledger=ledgers[key];ledger.transition('READY');ledger.transition('RUNNING');print('COSCHEDULE '+key+'; output '+str(ledger.root),flush=True)
        try:
            r=trial(inputs,ledger.root,config,schedule=schedule,observed=observed,seconds=seconds)
            state='PASS'
            try:
                envelope=complete(r,config,ref,schedule=schedule,observed=observed)
                if child is not None and envelope!=child['resource_device']:raise ValueError('frozen resource/device differs')
            except ValueError:state='INCONCLUSIVE'
            receipts[key]=ledger.transition(state,files=public_files(ledger.root),observations={'result':r})
        except BaseException as exc:
            receipts[key]=ledger.transition('PARTIAL' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'FAIL',reason=f'{type(exc).__name__}: {exc}',files=public_files(ledger.root));raise
    try:
        for key in BASELINES:run(key,'serial',False,570)
        child=freeze(config,ref,receipts);path=output/'comparison-contract.json';write(path,child);sha=file_hash(path)
        seconds=child['calibration']['reservation_seconds']
        if 15*seconds+60>deadline-time.monotonic():raise TimeoutError('remaining finite comparison/repeat roster cannot fit')
        for key in COMPARE:
            if file_hash(path)!=sha:raise ValueError('prospective co-scheduling child changed')
            run(key,'serial' if key.endswith('-A') else 'overlap',False,seconds)
        decision=compare(child,receipts,config,ref);decision_path=output/'comparison.json';write(decision_path,decision);decision_hash=file_hash(decision_path)
        if decision['status']!='COMPLETE':raise ValueError('incomplete comparison blocks repeatability')
        for key in REPEATS:
            if file_hash(path)!=sha or file_hash(decision_path)!=decision_hash:raise ValueError('frozen child/schedule decision changed')
            run(key,decision['selected_schedule'],True,seconds)
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
        if isinstance(exc,(ResourceBreach,MemoryError,KeyboardInterrupt,NonQuiescentWorker)) or isinstance(exc,OSError) and exc.errno in {errno.ENOSPC,errno.EDQUOT}:fatal=exc
    finally:
        for key,ledger in ledgers.items():
            if ledger.read()['state']=='PENDING':receipts[key]=ledger.transition('BLOCKED',reason=failure or 'preceding work incomplete')
            else:receipts.setdefault(key,ledger.read())
        decision=decision or compare(child,receipts,config,ref)
        repeated=[dict(id=k,state=receipts[k]['state'],receipt=receipts[k]['identity'],primary_seconds=receipts[k].get('observations',{}).get('result',{}).get('primary_seconds')) for k in REPEATS]
        times=[r['primary_seconds'] for r in repeated if r['state']=='PASS']
        result=seal(dict(version='pipeline-coschedule-report/v1',status='COMPLETE' if not failure and all(r['state']=='PASS' for r in receipts.values()) else 'INCONCLUSIVE',
            source=config['source'],contract=config['contract'],child=child['identity'] if child else None,comparison=decision,
            repeatability=dict(trials=repeated,complete=len(times)==10,minimum=min(times) if len(times)==10 else None,maximum=max(times) if len(times)==10 else None,median=statistics.median(times) if len(times)==10 else None),
            slots={k:dict(state=r['state'],receipt=r['identity'],reason=r.get('reason')) for k,r in receipts.items()},failure=failure,
            interpretation=MEANING,optimization_adoption=False,research_cost_admission=False))
        write(output/'coschedule-report.json',result)
    if fatal is not None:raise fatal
    return result


class CoscheduleStages:
    def __init__(self,adapters):
        from .replay import ReplayStages
        self.a=adapters;self.replay=ReplayStages(adapters,batching='ladder/v1');self.native_hash=None
    def __call__(self,stage,output):
        a=self.a
        if stage.id=='coschedule.reference':
            from .registry import Stage
            result=self.replay(Stage(stage.id,1,seconds=600,adapter='hardware-reference'),output,mode_override='coschedule-reference')
            self.native_hash=file_hash(self.replay.reference/'native-reference.json');return result
        if stage.id=='coschedule.report':
            r=a.receipts.get('coschedule.study',{});result=seal(dict(version='pipeline-coschedule-disposition/v1',state=r.get('state','UNSTARTED'),reason=r.get('reason','not dispatched'),
                result=r.get('observations',{}).get('result'),planned_slots=STUDY_SLOTS,
                slots=r.get('observations',{}).get('result',{}).get('slots',{}) or {k:dict(state='UNSTARTED',reason=r.get('reason','study not dispatched')) for k in STUDY_SLOTS},
                optimization_adoption=False,research_cost_admission=False))
            write(output/'disposition.json',result);return result
        from .resources import resolve,observe,reserve
        from .runner import CapabilityError,ResourceBreach
        from .benchmarks import legacy
        if self.replay.reference is None:raise CapabilityError('frozen co-scheduling references required')
        reference=self.replay.reference
        if file_hash(reference/'native-reference.json')!=self.native_hash or file_hash(reference/'reference.json')!=self.replay.reference_hash or file_hash(reference/'private/reference.pt')!=self.replay.payload_hash:
            raise ValueError('co-scheduling reference changed before dispatch')
        host=a.host(a.args.engine);facts=observe();facts['limits'].append(host.policy['memory_bytes'])
        try:policy=resolve(**facts,profile=a.contract['resource_policy'])
        except ValueError as exc:raise CapabilityError(str(exc)) from exc
        if min(policy['cpus'],host.policy['cpus'])<6:raise CapabilityError('co-scheduling reserves two Torch plus four native CPU equivalents')
        inputs=output/'input';inputs.mkdir()
        for name in ('weights.pt','work.json'):os.link(self.replay.prepared/'input'/name,inputs/name)
        for name in ('reference.json','native-reference.json'):shutil.copyfile(reference/name,inputs/name)
        os.link(reference/'private/reference.pt',inputs/'reference.pt')
        peak=(16<<30)+12*(inputs/'weights.pt').stat().st_size
        try:reservation=reserve(policy,peak)
        except ValueError as exc:raise CapabilityError(str(exc)) from exc
        if reservation['reserved_bytes']>math.floor(.9*policy['ceiling_bytes']):raise CapabilityError('shared co-scheduling reservation needs ten percent unreserved capacity')
        write(output/'resource-reservation.json',dict(policy=policy,reservation=reservation,ownership='one container,one shared quota,one active GPU child'))
        import copy
        host=copy.copy(host);host.policy=dict(host.policy,memory_bytes=policy['ceiling_bytes'],cpu_memory_bytes=policy['ceiling_bytes'],cpus=min(policy['cpus'],host.policy['cpus']))
        cfg=dict(version='pipeline-role/v1',source=json.loads((inputs/'work.json').read_text())['manifest']['binding']['source'],contract=a.contract['identity'],mode='coschedule-study',
            options=dict(seconds=12470,memory_ceiling=host.policy['memory_bytes'],render_node=host.policy['render_node']))
        write(inputs/'expected.json',cfg)
        with host.gpu_lock():legacy(a.root).container_stage(host,a.images.images[(a.args.engine,'gpu')],inputs,output/'role',mode='coschedule-study',seconds=12500,gpu=True,
            argv=['python','-m','oeis_learn.pipeline_validation.role','coschedule-study'])
        result=json.loads((output/'role/summary.json').read_text())
        if result.get('fatal'):raise ResourceBreach(result.get('error','co-scheduling containment failure'))
        if result.get('status')=='FAIL':raise RuntimeError(result.get('error','co-scheduling role failed'))
        if result.get('status')!='COMPLETE' or result.get('resources',{}).get('qualified') is not True:return dict(result,stage_state='INCONCLUSIVE')
        return result
