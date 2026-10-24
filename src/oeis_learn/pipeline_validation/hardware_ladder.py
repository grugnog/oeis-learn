"""One finite hardware handoff: telemetry calibration and three capacity branches."""
import json
import math
import statistics
import time
from .identity import seal,validate,number,write,file_hash,digest
from .comparisons import calibrate,ORDER
from .replay import TRIALS,ReplayStages
from .replay_comparison import envelope,comparison_report

BASIS=seal(dict(version='pipeline-hardware-basis/v1',
    archive='sha256:b075bcee4348130494859e56110ad69c1c0318d0ae32d54b8e4d377b7786a5a5',
    source='sha256:10ea8ca7c2cc459cacdd9ec886e8ae198b9032b8bf725a98600d276c4e6912e4',
    report='sha256:7cc536cc6b34fe48113a592dc401e7adb4960529f7149df71a16a88888fac0ed',
    batch8_to16_gains=[.4900320562301599,.489559411725661]))
OVERHEAD=['hardware.overhead.ordinary','hardware.overhead.telemetry']
KEYS=[f'r{r}.p{p}' for r in range(128) for p in (1,32,128,513,1024)]
MEANING='finite forced-replay hardware screen; no production/default/research/learning admission'
RULE='both paired gains >=5%; candidate median <=95% predecessor before expansion; smaller within5% of fastest'
CHILD_FIELDS={'version','validation','source','reference','workload','weights','baseline_trials','calibration',
    'resource_device','threads','measurement','candidate_batch','slots','capacity','limits','basis','decision_rule','interpretation'}


def ladder_plan(bos):
    return seal(dict(version='pipeline-hardware-workload/v1',visible_rows=32,row_mapping=[r%32 for r in range(128)],
        positions=[1,32,128,513,1024],tokens=[[bos]*1024 for _ in range(128)],batches=[16,32,64,128],
        bookkeeping='hoisted/v1',backend='eager',precision='float32',measurement='calibrated-common-mode/v1',
        meaning='32 unique visible inputs cyclically replicated to128 physical cache rows;131072 supplied tokens/640 row boundaries'))


def prepare_inputs(recipe,prompts,weights,output,**binding):
    from .replay import prepare_inputs as prepare
    return prepare(recipe,prompts,weights,output,plan_factory=ladder_plan,**binding)


def reference_role(inputs,output,config):
    from .replay import reference_role as reference
    from .replay_batching import batch_reference
    options=config['options']
    if set(options)!={'seconds','memory_ceiling','render_node','batch_size','measurement'} or options['batch_size']!=16:
        raise ValueError('closed hardware reference options required')
    def execute(decoder,memory,plan,deadline,observed):
        for key in KEYS:observed(key)
        # Rows are independent; compute each of the32 unique full-prefix oracles
        # once, then apply the prospectively frozen cyclic mapping.
        unique=batch_reference(decoder,memory[:32],dict(plan,tokens=plan['tokens'][:32]),deadline)
        return {f'r{r}.p{p}':unique[f'r{plan["row_mapping"][r]}.p{p}'] for r in range(128) for p in plan['positions']}
    cfg=dict(config,options={k:v for k,v in options.items() if k not in {'batch_size','measurement'}})
    return reference(inputs,output,cfg,plan_factory=ladder_plan,execute=execute,keys=KEYS,
        memory_transform=lambda memory:memory.repeat(4,1,1),
        oracle='CPU full-prefix32 unique visible rows; frozen row%32 expansion to128 physical cache rows')


def numeric_role(inputs,output,config):
    """Numerical executor shared by qualified wrapper and actual tensor fixtures."""
    from .replay import replay_role
    from .replay_batching import batch_cached
    options=config['options']
    if (set(options)!={'seconds','memory_ceiling','render_node','batch_size','measurement'}
            or options['batch_size'] not in (16,32,64,128) or options['measurement'] not in {'ordinary/v1','aggregate1Hz/v1'}):
        raise ValueError('closed hardware numerical options required')
    cfg=dict(config,options={k:v for k,v in options.items() if k not in {'batch_size','measurement'}})
    def execute(decoder,memory,plan,deadline,observed):
        return batch_cached(decoder,memory,plan,deadline,observed,batch_size=options['batch_size'])
    result=replay_role(inputs,output,cfg,bookkeeping='hoisted/v1',plan_factory=ladder_plan,execute=execute,keys=KEYS,
        instrumented=options['measurement']=='aggregate1Hz/v1')
    return dict(result,batch_size=options['batch_size'],bookkeeping='hoisted/v1',measurement_mode=options['measurement'])


def slots(batch):
    if batch not in (32,64,128):raise ValueError('registered capacity candidate required')
    prefix=f'hardware.b{batch}'
    return [dict(id=prefix+'.warmup',arm='B',batch_size=batch,bookkeeping='hoisted/v1',pair=None,scored=False),
        *[dict(id=f'{prefix}.pair{pair}.{arm}',arm=arm,batch_size=16 if arm=='A' else batch,
            bookkeeping='hoisted/v1',pair=pair,scored=True) for pair,order in enumerate(('AB','BA')) for arm in order]]


def complete_result(receipt,reference,workload,*,batch,mode):
    result=receipt.get('observations',{}).get('result',{})
    if receipt.get('state')!='PASS' or seal(result)!=result:raise ValueError('complete immutable hardware role required')
    expected=dict(status='COMPLETE',reference=reference,workload=workload,batch_size=batch,
        measurement_mode=mode,boundaries=640,bookkeeping='hoisted/v1')
    if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('hardware work/mode/row identity mismatch')
    envelope(result);number(result['verified_seconds'],positive=True)
    return result


def validate_mode(record):
    validate(record,{'version','status','reference','workload','mode','overhead','trials','resource_device','threads'},'pipeline-hardware-mode/v1')
    if record['status']!='COMPLETE' or record['mode'] not in {'ordinary/v1','aggregate1Hz/v1'}:
        raise ValueError('complete mode calibration required')
    overhead=record['overhead']
    if type(overhead) not in (int,float) or not math.isfinite(overhead) or overhead<=-1:
        raise ValueError('finite matched overhead required')
    if record['mode']!=('ordinary/v1' if overhead>.05+1e-12 else 'aggregate1Hz/v1') or set(record['trials'])!=set(OVERHEAD):
        raise ValueError('frozen five-percent observation rule required')
    if set(record['resource_device'])!={'ceiling_bytes','device'}:raise ValueError('closed hardware envelope required')
    envelope(dict(resources=dict(qualified=True,ceiling_bytes=record['resource_device']['ceiling_bytes']),
        device=record['resource_device']['device'],torch_threads=record['threads']))
    return record


def mode_report(receipts,reference,workload):
    try:
        ordinary=complete_result(receipts.get(OVERHEAD[0],{}),reference,workload,batch=16,mode='ordinary/v1')
        measured=complete_result(receipts.get(OVERHEAD[1],{}),reference,workload,batch=16,mode='aggregate1Hz/v1')
        if envelope(ordinary)!=envelope(measured):raise ValueError('mode resource/device mismatch')
        overhead=measured['verified_seconds']/ordinary['verified_seconds']-1
        return validate_mode(seal(dict(version='pipeline-hardware-mode/v1',status='COMPLETE',reference=reference,
            workload=workload,mode='ordinary/v1' if overhead>.05+1e-12 else 'aggregate1Hz/v1',overhead=overhead,
            trials={k:receipts[k]['identity'] for k in OVERHEAD},resource_device=envelope(ordinary),threads=ordinary['torch_threads'])))
    except (ValueError,KeyError,TypeError) as exc:
        return seal(dict(version='pipeline-hardware-mode-failure/v1',status='INCONCLUSIVE',reason=str(exc),
            trials={k:receipts.get(k,{}).get('identity') for k in OVERHEAD},mode=None))


def baselines(receipts,reference,workload,measurement):
    validate_mode(measurement)
    rows=[complete_result(receipts.get(k,{}),reference,workload,batch=16,mode=measurement['mode']) for k in TRIALS]
    if any(envelope(r)!=measurement['resource_device'] for r in rows):raise ValueError('baseline resource/device differs from calibration')
    return rows


def validate_child(child):
    validate(child,CHILD_FIELDS,'pipeline-hardware-comparison/v1')
    batch=child['candidate_batch']
    if (child['slots']!=slots(batch) or child['basis']!=BASIS or child['decision_rule']!=RULE
            or child['interpretation']!=MEANING or set(child['baseline_trials'])!=set(TRIALS)):
        raise ValueError('registered finite hardware policy required')
    mode=validate_mode(child['measurement'])
    if (mode['reference']!=child['reference'] or mode['workload']!=child['workload']
            or mode['resource_device']!=child['resource_device'] or mode['threads']!=child['threads']):
        raise ValueError('frozen mode/reference/resource binding required')
    cal=child['calibration']
    validate(cal,{'version','parent','workload','baseline_receipts','reservation_seconds','rule','cold_runs','warmups','calibration_count','confirmation_order'},'pipeline-calibration/v1')
    component=number(cal['reservation_seconds'],positive=True)
    if (cal['parent']!=child['reference'] or cal['workload']!=child['workload'] or cal['cold_runs']!=1
            or cal['warmups']!=1 or cal['calibration_count']!=3 or cal['confirmation_order']!=ORDER
            or len(cal['baseline_receipts'])!=3 or cal['rule']!=digest('ceil(max(2 * max_elapsed, declared_minimum))')
            or not 60<=component<=280 or component!=int(component)
            or child['limits']!=dict(component_seconds=component,internal_seconds=component+30,external_seconds=component+60)):
        raise ValueError('finite baseline-only hardware reservation required')
    c=child['capacity']
    if set(c)!={'baseline_peak_bytes','weights_bytes','candidate_peak_bytes','headroom_bytes','required_bytes'}:
        raise ValueError('closed hardware capacity required')
    peak=number(c['baseline_peak_bytes'],positive=True);weights=number(c['weights_bytes'],positive=True)
    candidate=max((8<<30)+12*weights,math.ceil(batch/16*peak));headroom=max(1<<30,math.ceil(.1*candidate))
    if (c!=dict(baseline_peak_bytes=peak,weights_bytes=weights,candidate_peak_bytes=candidate,
            headroom_bytes=headroom,required_bytes=candidate+headroom)
            or candidate+headroom>math.floor(.9*child['resource_device']['ceiling_bytes'])):
        raise ValueError('hardware capacity/headroom cannot fit')
    return child


def freeze_child(receipts,reference,measurement,batch,*,weight_bytes):
    if seal(reference)!=reference:raise ValueError('immutable reference required')
    rows=baselines(receipts,reference['identity'],reference['workload'],measurement)
    calibration=calibrate([dict(complete=True,seconds=r['verified_seconds'],identity=r['identity']) for r in rows[2:]],
        minimum=60,cap=280,parent=reference['identity'],workload=reference['workload'])
    peak=max(number(r['resources']['role_peak_upper_bytes'],positive=True) for r in rows)
    candidate=max((8<<30)+12*weight_bytes,math.ceil(batch/16*peak));headroom=max(1<<30,math.ceil(.1*candidate))
    return validate_child(seal(dict(version='pipeline-hardware-comparison/v1',validation=reference['contract'],source=reference['source'],
        reference=reference['identity'],workload=reference['workload'],weights=reference['weights'],
        baseline_trials={k:receipts[k]['identity'] for k in TRIALS},calibration=calibration,
        resource_device=measurement['resource_device'],threads=measurement['threads'],measurement=measurement,
        candidate_batch=batch,slots=slots(batch),capacity=dict(baseline_peak_bytes=peak,weights_bytes=weight_bytes,
            candidate_peak_bytes=candidate,headroom_bytes=headroom,required_bytes=candidate+headroom),
        limits=dict(component_seconds=calibration['reservation_seconds'],internal_seconds=calibration['reservation_seconds']+30,
            external_seconds=calibration['reservation_seconds']+60),basis=BASIS,decision_rule=RULE,interpretation=MEANING)))


def branch_report(child,receipts,*,previous_seconds):
    report=comparison_report(child,receipts,planned_slots=child['slots'],boundaries=640,validator=validate_child,
        version='pipeline-hardware-branch/v1',interpretation=MEANING)
    # The common comparison parser additionally needs the frozen observation mode.
    missing=[s['id']+': measurement mode mismatch' for s in child['slots']
        if receipts.get(s['id'],{}).get('state')=='PASS' and receipts[s['id']]['observations']['result'].get('measurement_mode')!=child['measurement']['mode']]
    if missing:report.update(status='INCONCLUSIVE',missing=report['missing']+missing,pairs=[],paired_gain=None)
    median=statistics.median(r['candidate_seconds'] for r in report['pairs']) if report['pairs'] else None
    supported=report['status']=='COMPLETE' and min(r['gain'] for r in report['pairs'])>=.05-1e-12
    return seal(dict(report,candidate_batch=child['candidate_batch'],candidate_median=median,supported=supported,
        advance=supported and median<=.95*number(previous_seconds,positive=True),supplied_tokens=131072,
        rows=128,unique_visible_rows=32,row_boundaries=640,previous_seconds=previous_seconds))


def hardware_role(inputs,output,config):
    options=config['options'];batch=options.get('batch_size')
    if 'comparison' not in options:
        if batch!=16:raise ValueError('candidate hardware role requires prospective child')
        return numeric_role(inputs,output,config)
    child=json.loads((inputs/'comparison-contract.json').read_text());validate_child(child)
    slot=json.loads((inputs/'comparison-slot.json').read_text())
    if (set(options)!={'seconds','memory_ceiling','render_node','batch_size','measurement','comparison','slot'}
            or slot not in child['slots'] or options['comparison']!=child['identity'] or options['slot']!=slot['id']
            or batch!=slot['batch_size'] or options['measurement']!=child['measurement']['mode']
            or options['seconds']!=child['limits']['internal_seconds'] or options['memory_ceiling']!=child['resource_device']['ceiling_bytes']
            or config['source']!=child['source'] or config['contract']!=child['validation']
            or (inputs/'weights.pt').stat().st_size!=child['capacity']['weights_bytes']):
        raise ValueError('hardware slot/source/measurement/deadline mismatch')
    ref=json.loads((inputs/'reference.json').read_text())
    if ref['identity']!=child['reference'] or ref['workload']!=child['workload'] or ref['weights']!=child['weights']:
        raise ValueError('hardware frozen reference mismatch')
    cfg=dict(config,options={k:v for k,v in options.items() if k not in {'comparison','slot'}})
    result=numeric_role(inputs,output,cfg)
    if result['device']!=child['resource_device']['device'] or result['torch_threads']!=child['threads']:
        raise ValueError('hardware device/threads mismatch')
    return dict(result,comparison=child['identity'],slot=slot['id'])


class HardwareStages:
    def __init__(self,adapters):
        self.a=adapters;self.replay=ReplayStages(adapters,batching='ladder/v1');self.measurement=None

    def reference(self):
        if self.replay.reference is None:raise ValueError('hardware reference unavailable')
        if (file_hash(self.replay.reference/'reference.json')!=self.replay.reference_hash
                or file_hash(self.replay.reference/'private/reference.pt')!=self.replay.payload_hash):
            raise ValueError('hardware frozen reference changed')
        return json.loads((self.replay.reference/'reference.json').read_text())

    def __call__(self,stage,output):
        a=self.a
        if stage.adapter=='hardware-reference':return self.replay(stage,output)
        if stage.id in OVERHEAD:
            return self.replay(stage,output,trial_options={'measurement':'ordinary/v1' if stage.id==OVERHEAD[0] else 'aggregate1Hz/v1'})
        if stage.id=='hardware.measurement':
            ref=self.reference();report=mode_report(a.receipts,ref['identity'],ref['workload'])
            write(output/'measurement.json',report)
            if report['status']=='COMPLETE':self.measurement=report
            return dict(report,stage_state='PASS' if report['status']=='COMPLETE' else 'INCONCLUSIVE')
        if stage.id in TRIALS:
            if self.measurement is None:raise ValueError('hardware mode must freeze before calibration')
            return self.replay(stage,output,trial_options={'measurement':self.measurement['mode']})
        if stage.id=='hardware.baseline':
            ref=self.reference();rows=baselines(a.receipts,ref['identity'],ref['workload'],self.measurement)
            report=seal(dict(version='pipeline-hardware-baseline/v1',status='COMPLETE',median=statistics.median(r['verified_seconds'] for r in rows[2:]),
                trials={k:a.receipts[k]['identity'] for k in TRIALS},measurement=self.measurement['identity']))
            write(output/'baseline.json',report);return report
        if stage.adapter=='hardware-branch':return self.branch(stage,output)
        reports={k:a.receipts.get('hardware.b'+str(k),{}).get('observations',{}).get('result',{}) for k in (32,64,128)}
        candidates={16:a.receipts.get('hardware.baseline',{}).get('observations',{}).get('result',{}).get('median')}
        candidates.update({k:r['candidate_median'] for k,r in reports.items() if r.get('status')=='COMPLETE' and r.get('supported')})
        candidates={k:v for k,v in candidates.items() if v is not None}
        selected=min(k for k,v in candidates.items() if v<=min(candidates.values())*1.05) if candidates else None
        report=seal(dict(version='pipeline-hardware-report/v1',branches=reports,experimental_batch=selected,
            measurement=self.measurement,optimization_adoption=False,research_cost_admission=False,interpretation=MEANING))
        write(output/'hardware-report.json',report);return report

    def branch(self,stage,output):
        from .receipts import Ledger
        from .runner import public_files,ResourceBreach,CapabilityError
        from .registry import Stage
        batch=int(stage.id.split('b')[-1]);a=self.a;ref=self.reference();planned=slots(batch)
        ledgers={s['id']:Ledger(output/'roles'/s['id'],s['id'],dict(source=ref['source'],contract=a.contract['identity'])) for s in planned}
        previous=a.receipts['hardware.baseline']['observations']['result'] if batch==32 else a.receipts['hardware.b'+str(batch//2)]['observations']['result']
        previous_seconds=previous['median'] if batch==32 else previous.get('candidate_median')
        if batch!=32 and not previous.get('advance'):
            for ledger in ledgers.values():ledger.transition('NOT_SELECTED',reason='prospective expansion threshold not met')
            report=seal(dict(version='pipeline-hardware-branch/v1',status='NOT_RUN',candidate_batch=batch,advance=False,
                candidate_median=None,supported=False,slots=[dict(slot=s,state='NOT_SELECTED',reason='prospective expansion threshold not met') for s in planned],
                optimization_adoption=False,research_cost_admission=False,interpretation=MEANING))
            write(output/'branch-report.json',report);return report
        child=None;rows={};failure=None;fatal=None;state='PASS';deadline=time.monotonic()+stage.seconds
        try:
            child=freeze_child(a.receipts,ref,self.measurement,batch,weight_bytes=(self.replay.prepared/'input/weights.pt').stat().st_size)
            path=output/'comparison-contract.json';write(path,child);sha=file_hash(path)
            for slot in planned:
                if file_hash(path)!=sha:raise ValueError('hardware prospective child changed')
                if time.monotonic()+child['limits']['external_seconds']+35>deadline:raise TimeoutError('hardware branch remaining reservation cannot fit')
                ledger=ledgers[slot['id']];ledger.transition('READY');ledger.transition('RUNNING')
                print('HARDWARE ROLE '+slot['id']+'; output '+str(ledger.root),flush=True)
                try:
                    result=self.replay(Stage(slot['id'],2,seconds=340,adapter='hardware-role'),ledger.root,
                        comparison=(child,slot),trial_options={'measurement':self.measurement['mode']})
                    rows[slot['id']]=ledger.transition(result.get('stage_state','PASS'),files=public_files(ledger.root),observations={'result':result})
                    if rows[slot['id']]['state']!='PASS':raise CapabilityError('hardware role not qualified')
                except BaseException as exc:
                    if ledger.read()['state']=='RUNNING':rows[slot['id']]=ledger.transition('PARTIAL' if isinstance(exc,(TimeoutError,KeyboardInterrupt,CapabilityError)) else 'FAIL',
                        reason=f'{type(exc).__name__}: {exc}',files=public_files(ledger.root))
                    raise
        except BaseException as exc:
            failure=f'{type(exc).__name__}: {exc}';state='PARTIAL' if isinstance(exc,(TimeoutError,CapabilityError)) else 'FAIL'
            if isinstance(exc,(ResourceBreach,MemoryError,KeyboardInterrupt)) or isinstance(exc,OSError) and exc.errno in {28,122}:fatal=exc
        for key,ledger in ledgers.items():
            if ledger.read()['state']=='PENDING':rows[key]=ledger.transition('BLOCKED',reason=failure or 'preceding hardware role incomplete')
            else:rows.setdefault(key,ledger.read())
        report=branch_report(child,rows,previous_seconds=previous_seconds) if child else seal(dict(version='pipeline-hardware-branch/v1',status='INCONCLUSIVE',
            candidate_batch=batch,candidate_median=None,supported=False,advance=False,slots=[dict(slot=s,state=rows[s['id']]['state']) for s in planned],
            optimization_adoption=False,research_cost_admission=False,interpretation=MEANING))
        report=seal(dict(report,failure=failure));write(output/'branch-report.json',report)
        if fatal is not None:raise fatal
        if report['status']!='COMPLETE' and state=='PASS':state='INCONCLUSIVE'
        return dict(report,stage_state=state)
