"""Hardware-only equal-work batching; no learning or production promotion."""
import json
import math
from .identity import seal,validate,number,write,file_hash
from .replay import TRIALS,replay_report,ReplayStages,check_deadline
from .replay_comparison import (
    FIELDS,SLOTS,INTERPRETATION,freeze_comparison,validate_comparison,comparison_report,
)

BATCH_SLOTS=[dict(slot,bookkeeping='hoisted/v1',batch_size=8 if slot['arm']=='A' else 16) for slot in SLOTS]
MEANING='observational equal-work forced replay batching only; no pipeline/default/learning admission'
KEYS=[f'r{r}.p{p}' for r in range(32) for p in (1,32,128,513,1024)]


def batch_plan(bos):
    return seal(dict(version='pipeline-replay-batching-workload/v1',positions=[1,32,128,513,1024],
        tokens=[[bos]*1024 for _ in range(32)],batches=[8,16],bookkeeping='hoisted/v1',
        backend='eager',precision='float32',measurement='aggregate1Hz/v1',
        meaning='32 fixed visible rows; identical32768 supplied BOS tokens/160 row boundaries; not generated programs'))


def prepare_inputs(recipe,prompts,weights,output,**binding):
    from .replay import prepare_inputs as prepare
    return prepare(recipe,prompts,weights,output,plan_factory=batch_plan,**binding)


def batch_reference(decoder,memory,plan,deadline,observed=None):
    import torch
    outputs={}
    with torch.no_grad():
        tokens=torch.tensor(plan['tokens'],device=memory.device,dtype=torch.long)
        for p in plan['positions']:
            check_deadline(deadline)
            for r in range(32):
                if observed:observed(f'r{r}.p{p}')
            value=decoder(tokens[:,:p],memory)[:,-1:].detach().cpu()
            for r in range(32):outputs[f'r{r}.p{p}']=value[r:r+1]
    return outputs


def batch_cached(decoder,memory,plan,deadline,observed=None,*,batch_size):
    import torch
    if batch_size not in (8,16) or len(memory)!=32:raise ValueError('registered32-row batch8/16 workload required')
    outputs={}
    def capture(value,start,p):
        for offset in range(batch_size):
            key=f'r{start+offset}.p{p}';outputs[key]=value[offset:offset+1]
            if observed:observed(key)
    with torch.no_grad():
        tokens=torch.tensor(plan['tokens'],device=memory.device,dtype=torch.long)
        for start in range(0,32,batch_size):
            check_deadline(deadline)
            end=start+batch_size
            value,cache=decoder.prefill(tokens[start:end,:1],memory[start:end],capacity=1024,backend='eager')
            capture(value,start,1)
            for p in range(2,1025):
                check_deadline(deadline)
                value=decoder.decode_step(tokens[start:end,p-1],cache)
                if p in plan['positions']:capture(value,start,p)
            del cache
    return outputs


def reference_role(inputs,output,config):
    from .replay import reference_role as reference
    cfg=dict(config,options={k:v for k,v in config['options'].items() if k!='batch_size'})
    if config['options'].get('batch_size')!=8:raise ValueError('baseline reference options required')
    return reference(inputs,output,cfg,plan_factory=batch_plan,execute=batch_reference,keys=KEYS,
        oracle='CPU full-prefix dense logits for32 distinct frozen visible rows at five positions')


def batch_role(inputs,output,config):
    from .replay import replay_role
    options=config['options'];batch=options.get('batch_size')
    if batch not in (8,16):raise ValueError('registered batch size required')
    cfg=dict(config,options={k:v for k,v in options.items() if k!='batch_size'})
    def execute(decoder,memory,plan,deadline,observed):
        return batch_cached(decoder,memory,plan,deadline,observed,batch_size=batch)
    result=replay_role(inputs,output,cfg,bookkeeping='hoisted/v1',plan_factory=batch_plan,execute=execute,keys=KEYS)
    return dict(result,batch_size=batch,bookkeeping='hoisted/v1')


def baseline_report(receipts,reference,workload):
    report=replay_report(receipts,reference,workload);missing=list(report['missing'])
    for key in TRIALS:
        result=receipts.get(key,{}).get('observations',{}).get('result',{})
        if any(result.get(k)!=v for k,v in dict(batch_size=8,boundaries=160,bookkeeping='hoisted/v1').items()):
            missing.append(key+': fixed-work batch8 baseline required')
    return seal(dict(report,version='pipeline-replay-batching-baseline/v1',missing=missing,
        status='INCONCLUSIVE' if missing else 'COMPLETE',calibration=None if missing else report['calibration'],authority=MEANING))


def legacy_view(child):
    # Reuse the closed common source/calibration/resource/deadline validator.
    # This transient structural view is never exported or used as old evidence.
    return seal({**{k:v for k,v in child.items() if k!='capacity'},
        'version':'pipeline-replay-comparison/v1','slots':SLOTS,'interpretation':INTERPRETATION})


def validate_batching(child):
    validate(child,FIELDS|{'capacity'},'pipeline-replay-batching-comparison/v1')
    if child['slots']!=BATCH_SLOTS or child['interpretation']!=MEANING:raise ValueError('registered batching slots required')
    validate_comparison(legacy_view(child))
    c=child['capacity']
    if set(c)!={'baseline_peak_bytes','candidate_peak_bytes','headroom_bytes','required_bytes'}:
        raise ValueError('closed batching capacity required')
    peak=number(c['baseline_peak_bytes'],positive=True);candidate=max(8<<30,2*peak)
    headroom=max(1<<30,math.ceil(candidate*.1))
    if (c!=dict(baseline_peak_bytes=peak,candidate_peak_bytes=candidate,headroom_bytes=headroom,required_bytes=candidate+headroom)
            or candidate+headroom>math.floor(.9*child['resource_device']['ceiling_bytes'])):
        raise ValueError('batching capacity with ten percent unreserved cannot fit')
    return child


def freeze_batching(report,receipts,reference):
    expected=replay_report(receipts,reference['identity'],reference['workload'])
    if report!=expected or baseline_report(receipts,reference['identity'],reference['workload'])['status']!='COMPLETE':
        raise ValueError('complete fixed-work batch8 calibration required')
    common=freeze_comparison(report,receipts,reference)
    peak=max(number(receipts[k]['observations']['result']['resources']['role_peak_upper_bytes'],positive=True) for k in TRIALS)
    candidate=max(8<<30,2*peak);headroom=max(1<<30,math.ceil(candidate*.1))
    return validate_batching(seal(dict(common,version='pipeline-replay-batching-comparison/v1',
        slots=BATCH_SLOTS,interpretation=MEANING,capacity=dict(baseline_peak_bytes=peak,
            candidate_peak_bytes=candidate,headroom_bytes=headroom,required_bytes=candidate+headroom))))


def batching_report(child,receipts):
    report=comparison_report(child,receipts,planned_slots=BATCH_SLOTS,boundaries=160,validator=validate_batching,
        version='pipeline-replay-batching-report/v1',interpretation=MEANING)
    pairs=[dict(row,baseline_supplied_tokens_per_second=32768/row['baseline_seconds'],
        candidate_supplied_tokens_per_second=32768/row['candidate_seconds']) for row in report['pairs']]
    return seal(dict(report,pairs=pairs,supplied_tokens=32768,rows=32,row_boundaries=160))


def comparison_role(inputs,output,config):
    child=json.loads((inputs/'comparison-contract.json').read_text());validate_batching(child)
    slot=json.loads((inputs/'comparison-slot.json').read_text());options=config['options']
    if (slot not in BATCH_SLOTS or set(options)!={'seconds','memory_ceiling','render_node','comparison','slot','batch_size'}
            or options['comparison']!=child['identity'] or options['slot']!=slot['id']
            or options['batch_size']!=slot['batch_size'] or config['source']!=child['source']
            or config['contract']!=child['validation'] or options['seconds']!=child['limits']['internal_seconds']):
        raise ValueError('frozen batching slot/source/validation/deadline mismatch')
    ref=json.loads((inputs/'reference.json').read_text())
    if (ref['identity']!=child['reference'] or ref['workload']!=child['workload'] or ref['weights']!=child['weights']
            or options['memory_ceiling']!=child['resource_device']['ceiling_bytes']):raise ValueError('batching reference/resource mismatch')
    cfg=dict(config,options={k:v for k,v in options.items() if k not in {'comparison','slot'}})
    result=batch_role(inputs,output,cfg)
    if result['device']!=child['resource_device']['device'] or result['torch_threads']!=child['threads']:
        raise ValueError('batching device/threads mismatch')
    return dict(result,comparison=child['identity'],slot=slot['id'])


class BatchingStages:
    def __init__(self,adapters):
        self.a=adapters;self.replay=ReplayStages(adapters,batching=True);self.path=None;self.hash=None

    def child(self):
        if self.path is None:return None
        if file_hash(self.path)!=self.hash:raise ValueError('frozen batching child changed')
        return validate_batching(json.loads(self.path.read_text()))

    def __call__(self,stage,output):
        a=self.a
        if stage.id=='replay.report':
            ref=json.loads((self.replay.reference/'reference.json').read_text()) if self.replay.reference else {}
            report=baseline_report(a.receipts,ref.get('identity'),ref.get('workload'))
            write(output/'batching-baseline.json',report)
            return dict(report,stage_state='PASS' if report['status']=='COMPLETE' else 'INCONCLUSIVE')
        if stage.id=='replay.compare.freeze':
            if (file_hash(self.replay.reference/'reference.json')!=self.replay.reference_hash
                    or file_hash(self.replay.reference/'private/reference.pt')!=self.replay.payload_hash):
                raise ValueError('frozen reference changed before batching freeze')
            ref=json.loads((self.replay.reference/'reference.json').read_text())
            child=freeze_batching(replay_report(a.receipts,ref['identity'],ref['workload']),a.receipts,ref)
            self.path=output/'comparison-contract.json';write(self.path,child);self.hash=file_hash(self.path)
            return child
        if stage.id=='replay.compare.report':
            report=batching_report(self.child(),a.receipts);write(output/'batching-report.json',report)
            return dict(report,stage_state='PASS' if report['status']=='COMPLETE' else 'INCONCLUSIVE')
        if stage.id in {s['id'] for s in BATCH_SLOTS}:
            child=self.child()
            if child is None:raise ValueError('batching child must freeze before candidate dispatch')
            return self.replay(stage,output,comparison=(child,next(s for s in BATCH_SLOTS if s['id']==stage.id)))
        return self.replay(stage,output)
