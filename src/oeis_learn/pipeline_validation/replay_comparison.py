"""Closed two-pair replay experiment; never production or research admission."""
import json
from pathlib import Path
import statistics

from .identity import seal, validate, number, write, file_hash, digest
from .replay import TRIALS, replay_report

SLOTS=[dict(id='replay.compare.warmup',arm='B',bookkeeping='hoisted/v1',pair=None,scored=False),
    *[dict(id=f'replay.compare.pair{pair}.{arm}',arm=arm,
        bookkeeping='baseline' if arm=='A' else 'hoisted/v1',pair=pair,scored=True)
      for pair,order in enumerate(('AB','BA')) for arm in order]]
FIELDS={'version','validation','source','reference','workload','weights','calibration',
    'baseline_trials','resource_device','threads','measurement','slots','limits','interpretation'}
INTERPRETATION='observational exploratory forced replay only; no optimization or research admission'


def envelope(result):
    if result['resources']['qualified'] is not True:raise ValueError('qualified replay resources required')
    number(result['resources']['ceiling_bytes'],positive=True)
    device=result['device']
    if set(device)!={'name','hip'} or not all(isinstance(v,str) and v for v in device.values()):
        raise ValueError('actual HIP device identity required')
    if result['torch_threads']!={'intraop':2,'interop':1}:raise ValueError('frozen replay threads required')
    return dict(ceiling_bytes=result['resources']['ceiling_bytes'],device=device)


def validate_comparison(child):
    validate(child,FIELDS,'pipeline-replay-comparison/v1')
    if (child['slots']!=SLOTS or child['measurement']!='aggregate1Hz/v1'
            or child['interpretation']!=INTERPRETATION or set(child['baseline_trials'])!=set(TRIALS)
            or child['threads']!={'intraop':2,'interop':1}):
        raise ValueError('registered replay comparison policy required')
    calibration=child['calibration']
    validate(calibration,{'version','parent','workload','baseline_receipts','reservation_seconds','rule',
        'cold_runs','warmups','calibration_count','confirmation_order'},'pipeline-calibration/v1')
    if calibration['parent']!=child['reference'] or calibration['workload']!=child['workload']:
        raise ValueError('comparison calibration binding mismatch')
    if (calibration['rule']!=digest('ceil(max(2 * max_elapsed, declared_minimum))')
            or calibration['cold_runs']!=1 or calibration['warmups']!=1 or calibration['calibration_count']!=3
            or calibration['confirmation_order']!=['AB','BA','AB','BA','AB']
            or len(calibration['baseline_receipts'])!=3):
        raise ValueError('registered three-observation calibration rule required')
    component=number(calibration['reservation_seconds'],positive=True)
    if (component<60 or component!=int(component) or component+60>600
            or child['limits']!=dict(component_seconds=component,internal_seconds=component+30,external_seconds=component+60)):
        raise ValueError('replay component plus setup/startup cannot fit finite role cap')
    if set(child['resource_device'])!={'ceiling_bytes','device'}:raise ValueError('closed comparison resources required')
    envelope(dict(resources=dict(qualified=True,ceiling_bytes=child['resource_device']['ceiling_bytes']),
        device=child['resource_device']['device'],torch_threads=child['threads']))
    return child


def freeze_comparison(report,receipts,reference):
    # Recompute from all five immutable receipts; a supplied favorable summary
    # cannot substitute for missing/changed baseline work.
    if seal(reference)!=reference:raise ValueError('immutable reference required before comparison freeze')
    expected=replay_report(receipts,reference['identity'],reference['workload'])
    if report!=expected or report['status']!='COMPLETE':raise ValueError('complete baseline replay required before candidates')
    values=[]
    for key in TRIALS:
        result=receipts[key]['observations']['result']
        if seal(result)!=result:raise ValueError('immutable baseline result required')
        values.append(envelope(result))
    if any(value!=values[0] for value in values):raise ValueError('baseline resource/device mismatch')
    component=report['calibration']['reservation_seconds']
    child=seal(dict(version='pipeline-replay-comparison/v1',validation=reference['contract'],
        source=reference['source'],reference=reference['identity'],workload=reference['workload'],weights=reference['weights'],
        calibration=report['calibration'],baseline_trials={key:receipts[key]['identity'] for key in TRIALS},
        resource_device=values[0],threads={'intraop':2,'interop':1},measurement='aggregate1Hz/v1',
        slots=SLOTS,limits=dict(component_seconds=component,internal_seconds=component+30,external_seconds=component+60),
        interpretation=INTERPRETATION))
    return validate_comparison(child)


def comparison_report(child,receipts):
    slots=[];pairs=[];missing=[]
    if child is not None:validate_comparison(child)
    for slot in SLOTS:
        receipt=receipts.get(slot['id'],{})
        result=receipt.get('observations',{}).get('result',{})
        reason=None
        try:
            if child is None:raise ValueError('comparison contract never frozen')
            if receipt.get('state')!='PASS' or result.get('status')!='COMPLETE':raise ValueError('arm incomplete or unstarted')
            if seal(result)!=result:raise ValueError('changed arm result')
            for field,expected in dict(comparison=child['identity'],slot=slot['id'],bookkeeping=slot['bookkeeping'],
                    reference=child['reference'],workload=child['workload'],boundaries=12,torch_threads=child['threads']).items():
                if result.get(field)!=expected:raise ValueError('arm '+field+' mismatch')
            if envelope(result)!=child['resource_device']:raise ValueError('arm resource/device mismatch')
            number(result['verified_seconds'],positive=True);number(result['decode_seconds'])
        except (ValueError,TypeError,KeyError) as exc:
            reason=str(exc);missing.append(slot['id']+': '+reason)
        slots.append(dict(slot=slot,receipt=receipt.get('identity'),state=receipt.get('state','UNSTARTED'),
            result=result.get('identity'),complete=reason is None,reason=reason,
            verified_seconds=result.get('verified_seconds') if reason is None else None))
    if not missing:
        for pair,order in enumerate(('AB','BA')):
            arms={s['slot']['arm']:s['verified_seconds'] for s in slots if s['slot']['pair']==pair}
            pairs.append(dict(pair=pair,order=order,baseline_seconds=arms['A'],candidate_seconds=arms['B'],gain=1-arms['B']/arms['A']))
    gains=[p['gain'] for p in pairs]
    return seal(dict(version='pipeline-replay-comparison-report/v1',status='INCONCLUSIVE' if missing else 'COMPLETE',
        comparison=child['identity'] if child else None,slots=slots,missing=missing,pairs=pairs,
        paired_gain=dict(median=statistics.median(gains),minimum=min(gains),maximum=max(gains)) if gains else None,
        optimization_adoption=False,research_cost_admission=False,interpretation=INTERPRETATION))


def comparison_role(inputs,output,config):
    """Validate the prospective child/slot inside the real isolated GPU role."""
    from .replay import replay_role
    child=json.loads((inputs/'comparison-contract.json').read_text());validate_comparison(child)
    slot=json.loads((inputs/'comparison-slot.json').read_text())
    options=config['options']
    if (set(options)!={'seconds','memory_ceiling','render_node','comparison','slot'}
            or options['comparison']!=child['identity'] or options['slot']!=slot.get('id')):
        raise ValueError('expected comparison/slot identity mismatch')
    if (slot not in SLOTS or config['source']!=child['source'] or config['contract']!=child['validation']
            or config['options']['seconds']!=child['limits']['internal_seconds']):
        raise ValueError('frozen comparison slot/source/validation/deadline mismatch')
    ref=json.loads((inputs/'reference.json').read_text())
    if (ref['identity']!=child['reference'] or ref['workload']!=child['workload'] or ref['weights']!=child['weights']
            or config['options']['memory_ceiling']!=child['resource_device']['ceiling_bytes']):
        raise ValueError('frozen comparison reference/resource mismatch')
    replay_config=dict(config,options={k:v for k,v in options.items() if k not in {'comparison','slot'}})
    result=replay_role(inputs,output,replay_config,bookkeeping=slot['bookkeeping'])
    if result['device']!=child['resource_device']['device'] or result['torch_threads']!=child['threads']:
        raise ValueError('frozen comparison device/threads mismatch')
    return dict(result,comparison=child['identity'],slot=slot['id'],bookkeeping=slot['bookkeeping'])


class ComparisonStages:
    """Prospective child custody around the existing reference/replay adapters."""
    def __init__(self,replay):
        self.replay=replay;self.child_path=None;self.child_hash=None

    def child(self):
        if self.child_path is None:return None
        if file_hash(self.child_path)!=self.child_hash:raise ValueError('frozen comparison changed before dispatch')
        return validate_comparison(json.loads(self.child_path.read_text()))

    def __call__(self,stage,output):
        a=self.replay.a
        if stage.adapter=='replay-comparison-freeze':
            if (file_hash(self.replay.reference/'reference.json')!=self.replay.reference_hash
                    or file_hash(self.replay.reference/'private/reference.pt')!=self.replay.payload_hash):
                raise ValueError('frozen replay reference changed before comparison freeze')
            ref=json.loads((self.replay.reference/'reference.json').read_text())
            baseline=replay_report(a.receipts,ref['identity'],ref['workload'])
            child=freeze_comparison(baseline,a.receipts,ref)
            path=output/'comparison-contract.json';write(path,child)
            self.child_path=path;self.child_hash=file_hash(path)
            return child
        if stage.adapter=='replay-comparison-report':
            report=comparison_report(self.child(),a.receipts)
            write(output/'comparison-report.json',report)
            return dict(report,stage_state='PASS' if report['status']=='COMPLETE' else 'INCONCLUSIVE')
        child=self.child()
        if child is None:raise ValueError('comparison must freeze before candidate dispatch')
        slot=next(s for s in SLOTS if s['id']==stage.id)
        return self.replay(stage,output,comparison=(child,slot))
