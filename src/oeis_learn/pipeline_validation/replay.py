"""Baseline-only forced replay. No generation, solver, updates or study authority."""
import json
import os
from pathlib import Path
import shutil
import time
from .identity import digest,seal,validate,write,file_hash,number

TRIALS=['replay.cold','replay.warmup',*[f'replay.calibration-{i}' for i in range(3)]]


class BoundaryLedger:
    """At most 12 queued/verified boundaries; a missing terminal is partial."""
    def __init__(self,output):
        self.output=Path(output)
        self.states={**{f'b{b}.p{p}':'UNSTARTED' for b in (1,8) for p in (1,32,128,513,1024)},
            'mixed.reordered':'UNSTARTED','mixed.inactive':'UNSTARTED'}

    def __enter__(self):
        self.handle=(self.output/'boundary-progress.jsonl').open('x')
        self.event(dict(event='PLAN',boundaries=list(self.states)))
        return self

    def event(self,row):
        self.handle.write(json.dumps(row,sort_keys=True)+'\n');self.handle.flush()

    def queued(self,key):
        if self.states.get(key)!='UNSTARTED':raise ValueError('unknown/duplicate replay boundary')
        self.states[key]='INCOMPLETE'
        self.event(dict(event='QUEUED_UNVERIFIED',boundary=key))

    def verified(self,key,record):
        if self.states.get(key)!='INCOMPLETE':raise ValueError('unqueued/duplicate verified boundary')
        self.states[key]='COMPLETE'
        self.event(dict(event='VERIFIED',boundary=key,output=record))

    def __exit__(self,kind,value,trace):
        self.handle.close()
        complete=kind is None and all(v=='COMPLETE' for v in self.states.values())
        write(self.output/'boundary-status.json',dict(status='COMPLETE' if complete else 'PARTIAL',
            expected=len(self.states),states=self.states,error=None if kind is None else kind.__name__))
        if kind is None and not complete:raise ValueError('incomplete replay boundary ledger')


def replay_plan(bos):
    return seal(dict(version='pipeline-replay-workload/v1',positions=[1,32,128,513,1024],
        batches=[1,8],tokens=[[bos]*1024 for _ in range(8)],mixed_order=[6,1,6],
        mixed_steps=32,bookkeeping='baseline',backend='eager',precision='float32',measurement='aggregate1Hz/v1',
        meaning='forced BOS replay, including mixed/finished/reordered/inactive rows; not generated programs'))


def replay_report(receipts,reference,workload):
    rows=[];missing=[];envelopes=set()
    for key in TRIALS:
        row=receipts.get(key,{})
        trial=row.get('observations',{}).get('result',{})
        if (row.get('state')!='PASS' or trial.get('status')!='COMPLETE'
                or not reference or trial.get('reference')!=reference or trial.get('workload')!=workload
                or trial.get('resources',{}).get('qualified') is not True):
            missing.append(key)
        if key.startswith('replay.calibration-'):
            rows.append(dict(complete=key not in missing,seconds=trial.get('verified_seconds'),identity=trial.get('identity')))
        try:
            ceiling=number(trial['resources']['ceiling_bytes'],positive=True)
            device=trial['device']
            if not all(isinstance(device[k],str) and device[k] for k in ('hip','name')):
                raise ValueError('device identity required')
            envelopes.add((ceiling,device['hip'],device['name']))
        except (KeyError,TypeError,ValueError):missing.append(key+': resource/device identity unavailable')
    if len(envelopes)!=1:missing.append('one consistent resource/device envelope required')
    calibration=None
    if not missing:
        from .comparisons import calibrate
        try:calibration=calibrate(rows,minimum=60,cap=600,parent=reference,workload=workload)
        except ValueError as exc:missing.append(str(exc))
    return seal(dict(version='pipeline-replay-report/v1',status='INCONCLUSIVE' if missing else 'COMPLETE',
        reference=reference,workload=workload,missing=missing,calibration=calibration,
        trials={key:receipts.get(key,{}).get('identity') for key in TRIALS},
        optimization_adoption=False,research_cost_admission=False,
        authority='baseline replay viability only; not full-pipeline, generated-program or research admission'))


def prepare_inputs(recipe,prompts,weights,output,**binding):
    """Reuse the existing parent's selection checks, without reading hidden truth."""
    from oeis_learn.decoder.bigint_codec import BOS_ID
    prompts=[{k:row[k] for k in ('id','values')} for row in prompts[:8]]
    if len(prompts)!=8 or len({r['id'] for r in prompts})!=8 or any(len(r['values'])!=20 for r in prompts):
        raise ValueError('eight distinct frozen twenty-visible-term replay prompts required')
    output=Path(output);inputs=output/'input';inputs.mkdir()
    plan=replay_plan(BOS_ID)
    manifest=seal(dict(version='pipeline-replay-input/v1',binding=binding,
        weights=file_hash(weights),recipe=digest(recipe),prompts=digest(prompts),plan=plan))
    shutil.copyfile(weights,inputs/'weights.pt')
    write(inputs/'work.json',dict(manifest=manifest,recipe=recipe,prompts=prompts))
    write(output/'manifest.json',manifest)
    return manifest


def load_models(inputs,config,device):
    import torch
    from oeis_learn.fresh.bigint_worker import models
    from oeis_learn.decoder.bigint_codec import BOS_ID
    work=json.loads((inputs/'work.json').read_text())
    if set(work)!={'manifest','recipe','prompts'}:raise ValueError('closed replay payload required')
    m=work['manifest'];validate(m,{'version','binding','weights','recipe','prompts','plan'},'pipeline-replay-input/v1')
    if (m['plan']!=replay_plan(BOS_ID) or m['binding']['source']!=config['source']
            or m['weights']!=file_hash(inputs/'weights.pt') or m['recipe']!=digest(work['recipe'])
            or m['prompts']!=digest(work['prompts']) or len(work['prompts'])!=8):
        raise ValueError('frozen replay payload/weights/source mismatch')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    encoder,decoder,counts=models(work['recipe'],device)
    weights=torch.load(inputs/'weights.pt',map_location='cpu',weights_only=True)
    encoder.load_state_dict(weights['encoder']);decoder.load_state_dict(weights['decoder'])
    encoder.eval();decoder.eval();decoder.cache_bookkeeping='baseline'
    if decoder.attention_backend!='eager' or next(decoder.parameters()).dtype!=torch.float32:
        raise ValueError('replay requires existing eager FP32 profile')
    return work,encoder,decoder


def role_deadline(config):
    options=config['options']
    if set(options)!={'seconds','memory_ceiling','render_node'} or not 0<number(options['seconds'],positive=True)<=600:
        raise ValueError('closed bounded replay options required')
    return time.monotonic()+options['seconds']


def tensor_record(value):
    import hashlib
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256='sha256:'+hashlib.sha256(value.numpy().tobytes()).hexdigest())


def check_deadline(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('frozen replay deadline')


def full_reference(decoder,memory,plan,deadline,observed=None):
    """Independent full-prefix oracle for every dense checkpoint."""
    import torch
    outputs={}
    with torch.no_grad():
        for batch in plan['batches']:
            tokens=torch.tensor(plan['tokens'][:batch],device=memory.device,dtype=torch.long)
            for position in plan['positions']:
                check_deadline(deadline)
                key=f'b{batch}.p{position}'
                outputs[key]=decoder(tokens[:,:position],memory[:batch])[:,-1:].detach().cpu()
                if observed:observed(key)
        # Mixed cache behavior is separately checked against the CPU baseline;
        # numerical full-prefix parity is established by the dense oracle above.
        outputs.update({k:v.detach().cpu() for k,v in mixed_replay(decoder,memory,plan,deadline).items()})
        if observed:
            for key in ('mixed.reordered','mixed.inactive'):observed(key)
    return outputs


def mixed_replay(decoder,memory,plan,deadline):
    import torch
    check_deadline(deadline)
    tokens=torch.tensor(plan['tokens'],device=memory.device,dtype=torch.long)
    _,cache=decoder.prefill(tokens[:,:1],memory,capacity=1024,backend='eager')
    for step in range(1,plan['mixed_steps']):
        check_deadline(deadline)
        decoder.decode_step(tokens[:,step],cache,active=[True,False]*4)
    if cache.lengths!=[32,1]*4:raise RuntimeError('mixed replay row mapping changed')
    cache.reorder(plan['mixed_order']);cache.finished[1]=True
    out=decoder.decode_step(tokens[:3,0],cache)
    if cache.lengths!=[33,1,33] or torch.count_nonzero(out[1]).item():raise RuntimeError('finished row mutated')
    before=cache.tokens.clone();lengths=list(cache.lengths)
    zero=decoder.decode_step(tokens[:3,0],cache,active=[False]*3)
    if not torch.equal(before,cache.tokens) or cache.lengths!=lengths or torch.count_nonzero(zero).item():
        raise RuntimeError('inactive replay mutated cache')
    return {'mixed.reordered':out,'mixed.inactive':zero}


def cached_replay(decoder,memory,plan,deadline,observed=None):
    import torch
    outputs={}
    with torch.no_grad():
        for batch in plan['batches']:
            check_deadline(deadline)
            tokens=torch.tensor(plan['tokens'][:batch],device=memory.device,dtype=torch.long)
            value,cache=decoder.prefill(tokens[:,:1],memory[:batch],capacity=1024,backend='eager')
            outputs[f'b{batch}.p1']=value
            if observed:observed(f'b{batch}.p1')
            for position in range(2,1025):
                check_deadline(deadline)
                value=decoder.decode_step(tokens[:,position-1],cache)
                if position in plan['positions']:
                    key=f'b{batch}.p{position}';outputs[key]=value
                    if observed:observed(key)
            del cache
        outputs.update(mixed_replay(decoder,memory,plan,deadline))
        if observed:
            for key in ('mixed.reordered','mixed.inactive'):observed(key)
    return outputs


def verify_outputs(outputs,expected,verified=None,deadline=None):
    import torch
    if set(outputs)!=set(expected):raise ValueError('missing/extra replay boundary')
    records={}
    for key,value in outputs.items():
        if deadline is not None:check_deadline(deadline)
        actual=value.detach().cpu();oracle=expected[key]
        if not torch.isfinite(actual).all() or not torch.isfinite(oracle).all():raise RuntimeError('nonfinite replay output')
        tolerance=dict(atol=2e-5,rtol=2e-4) if value.is_cuda else dict(atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(actual,oracle,**tolerance)
        if not torch.equal(actual.argmax(-1),oracle.argmax(-1)):raise RuntimeError('replay token mismatch')
        records[key]=tensor_record(actual)
        if deadline is not None:check_deadline(deadline)
        if verified:verified(key,records[key])
    return records


def reference_role(inputs,output,config):
    with BoundaryLedger(output) as evidence:
        return _reference_role(inputs,output,config,evidence)


def _reference_role(inputs,output,config,evidence):
    import torch
    deadline=role_deadline(config)
    work,encoder,decoder=load_models(inputs,config,'cpu');m=work['manifest']
    from oeis_learn.synthesis.contracts import canonical_integer
    with torch.no_grad():
        memory=encoder.forward_from_sequences([[canonical_integer(v) for v in row['values']] for row in work['prompts']],device='cpu')
        expected=full_reference(decoder,memory,m['plan'],deadline,evidence.queued)
    if not torch.isfinite(memory).all() or any(not torch.isfinite(v).all() for v in expected.values()):
        raise RuntimeError('nonfinite frozen replay reference')
    for key,value in expected.items():
        check_deadline(deadline);evidence.verified(key,tensor_record(value))
    check_deadline(deadline)
    path=output/'private/reference.pt';path.parent.mkdir()
    torch.save(dict(memory=memory,expected=expected),path)
    ref=seal(dict(version='pipeline-replay-reference/v1',source=config['source'],contract=config['contract'],
        workload=m['identity'],weights=m['weights'],payload=file_hash(path),memory=tensor_record(memory),
        expected={key:tensor_record(value) for key,value in expected.items()},oracle='CPU full-prefix dense logits; CPU baseline mixed-cache invariants'))
    write(output/'reference.json',ref)
    check_deadline(deadline)
    return dict(status='COMPLETE',reference=ref['identity'],workload=m['identity'],authority='frozen visible-only replay reference; no hidden truth')


def replay_role(inputs,output,config,*,bookkeeping='baseline'):
    if bookkeeping not in {'baseline','hoisted/v1'}:raise ValueError('registered cache bookkeeping required')
    with BoundaryLedger(output) as evidence:
        result,deadline=_replay_role(inputs,output,config,evidence,bookkeeping=bookkeeping)
        finished=time.monotonic()
    result['verified_seconds']+=time.monotonic()-finished
    check_deadline(deadline)
    return result


def _replay_role(inputs,output,config,evidence,*,bookkeeping):
    import torch
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('one actual HIP device required for replay; no CPU fallback')
    deadline=role_deadline(config)
    work,encoder,decoder=load_models(inputs,config,'cuda');del encoder
    decoder.cache_bookkeeping=bookkeeping
    ref=json.loads((inputs/'reference.json').read_text())
    validate(ref,{'version','source','contract','workload','weights','payload','memory','expected','oracle'},'pipeline-replay-reference/v1')
    if (ref['source']!=config['source'] or ref['contract']!=config['contract']
            or ref['workload']!=work['manifest']['identity'] or ref['weights']!=work['manifest']['weights']
            or ref['payload']!=file_hash(inputs/'reference.pt')):raise ValueError('changed replay reference')
    payload=torch.load(inputs/'reference.pt',map_location='cpu',weights_only=True)
    if ref['memory']!=tensor_record(payload['memory']) or ref['expected']!={k:tensor_record(v) for k,v in payload['expected'].items()}:
        raise ValueError('reference tensor hashes differ')
    memory=payload['memory'].to('cuda');check_deadline(deadline)
    from oeis_learn.tracking.pipeline_profile import Recorder
    with Recorder(output/'observations',instrumented=True,render_node=config['options']['render_node']):
        torch.cuda.synchronize();started=time.monotonic()
        outputs=cached_replay(decoder,memory,work['manifest']['plan'],deadline,evidence.queued)
        torch.cuda.synchronize();decode_seconds=time.monotonic()-started
        records=verify_outputs(outputs,payload['expected'],evidence.verified,deadline)
        write(output/'boundaries.json',records)
    check_deadline(deadline)
    return dict(status='COMPLETE',reference=ref['identity'],workload=ref['workload'],
        decode_seconds=decode_seconds,verified_seconds=time.monotonic()-started,boundaries=len(records),
        device=dict(name=torch.cuda.get_device_name(),hip=torch.version.hip),
        torch_threads=dict(intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads()),
        measurement=f'fresh role and fresh {bookkeeping} KV caches; decode through verification/export; setup/startup separately observed',
        authority='forced replay viability only; not generated-program, optimization or research admission'),deadline


class ReplayStages:
    """One host entry point; CPU reference and five isolated finite GPU roles."""
    def __init__(self,adapters):
        self.a=adapters;self.prepared=None;self.reference=None;self.reference_hash=None;self.payload_hash=None

    def __call__(self,stage,output,*,comparison=None):
        a=self.a
        if stage.adapter=='replay-report':
            ref=json.loads((self.reference/'reference.json').read_text()) if self.reference else {}
            report=replay_report(a.receipts,ref.get('identity'),ref.get('workload'))
            write(output/'replay-report.json',report)
            return dict(report,stage_state='PASS' if report['status']=='COMPLETE' else 'INCONCLUSIVE')
        from .benchmarks import legacy
        from .runner import CapabilityError
        from .resources import observe,resolve,reserve
        module=legacy(a.root);host=a.host(a.args.engine)
        if stage.adapter=='replay-reference':
            if not a.args.run or not a.args.inputs:raise CapabilityError('replay needs --run and --inputs for frozen parent weights/visible prompts')
            self.prepared=output/'prepared';self.prepared.mkdir();host.select_gpu()
            from oeis_learn.experiments.source_snapshot import collect_snapshot
            m=module.prepare(a.root/a.args.run,a.root/a.args.inputs,self.prepared,
                image=a.images.images[(a.args.engine,'gpu')],source=digest(collect_snapshot(a.root)),render_node=host.policy['render_node'],replay=True)
            if a.contract['weights_ref'] and a.contract['weights_ref']['sha256']!=m['weights']:raise ValueError('manifest weights differ from frozen parent')
        inputs=output/'input';inputs.mkdir()
        source=self.prepared/'input'
        for name in ['weights.pt','work.json']:os.link(source/name,inputs/name)
        gpu=stage.adapter in {'replay-trial','replay-comparison-trial'}
        if gpu:
            if self.reference is None:raise ValueError('frozen replay reference required')
            if (file_hash(self.reference/'reference.json')!=self.reference_hash
                    or file_hash(self.reference/'private/reference.pt')!=self.payload_hash):
                raise ValueError('frozen replay reference changed before dispatch')
            shutil.copyfile(self.reference/'reference.json',inputs/'reference.json')
            os.link(self.reference/'private/reference.pt',inputs/'reference.pt')
        observations=observe();observations['limits'].append(host.policy['memory_bytes'])
        try:
            resource=resolve(**observations,profile=a.contract['resource_policy'])
            reservation=reserve(resource,12*(inputs/'weights.pt').stat().st_size+(4<<30))
        except ValueError as exc:raise CapabilityError(str(exc)) from exc
        write(output/'resource-reservation.json',dict(policy=resource,reservation=reservation))
        import copy
        host=copy.copy(host);host.policy=dict(host.policy,memory_bytes=resource['ceiling_bytes'],cpu_memory_bytes=resource['ceiling_bytes'])
        mode='replay' if gpu else 'replay-reference'
        seconds=stage.seconds;internal=seconds-30
        if comparison is not None:
            child,slot=comparison
            if resource['ceiling_bytes']!=child['resource_device']['ceiling_bytes']:
                raise CapabilityError('comparison resource ceiling changed')
            write(inputs/'comparison-contract.json',child);write(inputs/'comparison-slot.json',slot)
            seconds=child['limits']['external_seconds'];internal=child['limits']['internal_seconds']
            mode='replay-compare'
        m=json.loads((inputs/'work.json').read_text())['manifest']
        config=dict(version='pipeline-role/v1',source=m['binding']['source'],
            contract=a.contract['identity'],mode=mode,options=dict(seconds=internal,memory_ceiling=resource['ceiling_bytes'],render_node=host.policy['render_node']))
        if comparison is not None:config['options'].update(comparison=child['identity'],slot=slot['id'])
        write(inputs/'expected.json',config)
        from contextlib import nullcontext
        with host.gpu_lock() if gpu else nullcontext():
            module.container_stage(host,a.images.images[(a.args.engine,'gpu')],inputs,output/'role',mode=mode,
                seconds=seconds,gpu=gpu,argv=['python','-m','oeis_learn.pipeline_validation.role',mode])
        result=json.loads((output/'role/summary.json').read_text())
        if result['status']!='COMPLETE':raise RuntimeError('replay role incomplete: '+str(result.get('error')))
        if not gpu:
            self.reference=output/'role'
            self.reference_hash=file_hash(self.reference/'reference.json')
            self.payload_hash=file_hash(self.reference/'private/reference.pt')
        if result.get('resources',{}).get('qualified') is not True:
            return dict(result,stage_state='INCONCLUSIVE')
        return result
