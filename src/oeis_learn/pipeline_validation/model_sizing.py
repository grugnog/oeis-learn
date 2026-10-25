"""Finite fresh-model hardware observations; no learned model selection."""
import copy
import json
import os
from pathlib import Path
import statistics
import time
from .identity import seal,digest,write,file_hash,number,validate
from .model_sizes import PROFILES,TRIALS,SLOTS,validate_profile,parameters,peak_estimate

MEANING='fresh-model hardware sizing; no model-quality/default/persistent-state/research/learning admission'


def attempt_plan(prompts):
    if len(prompts)!=32 or len({r['id'] for r in prompts})!=32:raise ValueError('32 distinct real prompts required')
    return [dict(id=f'b{batch}-'+r['id'],prompt=r['id'],batch=batch,seed=20261016+i,
        state='UNSTARTED',tokens=[],decoded=False,native=None) for batch in (8,32) for i,r in enumerate(prompts)]


def generation_counts(rows,seconds):
    if seconds<=0:raise ValueError('positive whole-role denominator required')
    decoded=sum(r['decoded'] is True for r in rows)
    verified=sum(bool(r.get('native',{}).get('verified')) for r in rows if r.get('native') is not None)
    emitted=sum(len(r['tokens']) for r in rows)
    return dict(scheduled=len(rows),started=sum(r['state']!='UNSTARTED' for r in rows),
        decoded=decoded,verified=verified,emitted_tokens=emitted,
        decoded_per_whole_minute=decoded*60/seconds,verified_per_whole_minute=verified*60/seconds)


def summarize_profiles(receipts):
    profiles={};eligible=[]
    for key,p in PROFILES.items():
        rows=[receipts.get(f'sizing.{key}.{t}',{}) for t in TRIALS]
        results=[r.get('observations',{}).get('result',{}) for r in rows]
        missing=[]
        for t,r,v in zip(TRIALS,rows,results):
            try:
                if r!=seal(r) or r.get('state')!='PASS':raise ValueError('complete immutable receipt required')
                complete_result(v,p)
            except (ValueError,KeyError,TypeError) as exc:missing.append(t+': '+str(exc))
        envelopes={(v.get('source'),v.get('contract'),v.get('workload'),v.get('reference'),
            v.get('resources',{}).get('ceiling_bytes'),digest(v.get('device'))) for v in results}
        if len(envelopes)!=1:missing.append('one source/reference/resource/device envelope required')
        ok=not missing
        if ok:eligible.append(key)
        profiles[key]=dict(profile=p,parameters=parameters(p),complete=ok,missing=missing,
            training_tokens_per_second_median=statistics.median(v['training_tokens_per_second'] for v in results[2:]) if ok else None,
            training_updates_per_second_median=statistics.median(v['training_updates_per_second'] for v in results[2:]) if ok else None,
            training_only_72h_extrapolation_tokens=statistics.median(v['training_tokens_per_second'] for v in results[2:])*72*3600 if ok else None,
            projection_caveat='short training-only extrapolation excludes generation/validation/checkpoints/export/thermal drift; not a complete cost receipt',
            slots={t:dict(state=r.get('state','UNSTARTED'),reason=r.get('reason','not dispatched'),receipt=r.get('identity')) for t,r in zip(TRIALS,rows)},
            trials={t:v for t,v in zip(TRIALS,results)})
    return seal(dict(version='pipeline-model-sizing-report/v1',profiles=profiles,hardware_eligible_profiles=eligible,
        model_selection='NOT_DECIDED',optimization_adoption=False,research_cost_admission=False,interpretation=MEANING))


def complete_result(result,profile):
    if result!=seal(result):raise ValueError('immutable sizing result required')
    expected=dict(status='COMPLETE',profile=profile,counts=parameters(profile),training_updates=32,
        validation_rows=32,numerical_parity=True,changed_parameters=True,torch_threads={'intraop':2,'interop':1})
    if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('same full sizing work/shape/actual threads required')
    memory=result.get('resources',{})
    if memory.get('qualified') is not True or memory.get('breach') is not None or not result.get('device',{}).get('hip'):
        raise ValueError('qualified aggregate memory and actual HIP required')
    number(memory['ceiling_bytes'],positive=True)
    for key in ('source','contract','workload','reference','initial_parameters','trained_parameters'):
        if not isinstance(result.get(key),str) or not result[key]:raise ValueError('source/reference/weights identities required')
    if result['initial_parameters']==result['trained_parameters']:raise ValueError('finite changed actual parameters required')
    for key in ('training_tokens','training_tokens_per_second','training_updates_per_second','whole_seconds'):
        number(result[key],positive=True)
    if result['whole_seconds']>1080 or result.get('generation',{}).get('scheduled')!=64 or result['generation'].get('started')!=64:
        raise ValueError('all64 fixed-budget attempts inside immutable parent required')
    return result


def prepare_inputs(recipe,prompts,training,validation,output,**binding):
    attempt_plan(prompts)
    if len(training)!=128 or len(validation)!=32:raise ValueError('frozen128 training/32 validation rows required')
    if recipe['model']['seed']!=20261016 or recipe['inputs']['profile']!='hybrid-chunks/v1' or recipe['inputs']['mask']!={k:True for k in ('magnitude','modular','difference','exact_bytes')}:
        raise ValueError('existing full mathematical input profile required')
    if recipe['learning']['batch_size']!=8 or recipe['learning']['learning_rate']!=.0001:
        raise ValueError('unchanged batch8/LR1e-4 required')
    for r in training+validation:
        if len(r['values'])!=20 or not r['tokens'] or len(r['tokens'])>1024:raise ValueError('bounded real training token rows required')
    payload=dict(recipe=recipe,prompts=prompts,training=training,validation=validation)
    m=seal(dict(version='pipeline-sizing-input/v1',binding=binding,payload=digest(payload),profiles=PROFILES,
        slots=SLOTS,updates=32,training_batch=8,validation_rows=32,generation_batches=[8,32],body_tokens=1024,
        initialization='fresh seeded per profile; no inherited trained weights'))
    target=Path(output)/'input';target.mkdir();write(target/'work.json',dict(manifest=m,**payload));write(Path(output)/'manifest.json',m)
    return m


def read_work(inputs,config):
    work=json.loads((inputs/'work.json').read_text());m=work['manifest']
    validate(m,{'version','binding','payload','profiles','slots','updates','training_batch','validation_rows',
        'generation_batches','body_tokens','initialization'},'pipeline-sizing-input/v1')
    if (set(work)!={'manifest','recipe','prompts','training','validation'} or m!=seal(m)
        or m['profiles']!=PROFILES or m['slots']!=SLOTS or m['binding']['source']!=config['source']
        or m['payload']!=digest({k:v for k,v in work.items() if k!='manifest'})
        or m['updates']!=32 or m['training_batch']!=8 or m['validation_rows']!=32
        or m['generation_batches']!=[8,32] or m['body_tokens']!=1024
        or len(work['training'])!=128 or len(work['validation'])!=32):
        raise ValueError('frozen sizing work/source identity mismatch')
    attempt_plan(work['prompts']);return work


def options(config,cpu=False):
    opt=config['options']
    if set(opt)!={'seconds','memory_ceiling','render_node','profile'} or opt['seconds']!=(570 if cpu else 1080):
        raise ValueError('closed model sizing options required')
    return validate_profile(opt['profile'])


def reference_role(inputs,output,config):
    import torch
    from oeis_learn.experiments.bootstrap_models import construct_sizing_models
    from .replay import tensor_record,check_deadline
    p=options(config,True);work=read_work(inputs,config);deadline=time.monotonic()+570
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    encoder,decoder,counts=construct_sizing_models(p);encoder.eval();decoder.eval()
    private=output/'private';private.mkdir()
    torch.save(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict()),private/'weights.pt')
    with torch.no_grad():
        memory=encoder.forward_from_sequences([[int(v) for v in r['values']] for r in work['prompts'][:2]])
        tokens=torch.ones((2,128),dtype=torch.long)
        expected={}
        for pos in (1,32,128):
            check_deadline(deadline);expected[str(pos)]=decoder(tokens[:,:pos],memory)[:,-1:].cpu()
    torch.save(dict(memory=memory,expected=expected),private/'oracle.pt')
    ref=seal(dict(version='pipeline-sizing-reference/v1',source=config['source'],contract=config['contract'],
        profile=p,workload=work['manifest']['identity'],weights=file_hash(private/'weights.pt'),oracle=file_hash(private/'oracle.pt'),
        counts=counts,memory=tensor_record(memory),expected={k:tensor_record(v) for k,v in expected.items()}))
    write(output/'reference.json',ref);check_deadline(deadline)
    return dict(status='COMPLETE',profile=p,reference=ref['identity'],counts=counts,interpretation=MEANING)


def fit_generated(output,rows,prompts,config,deadline):
    """Same independent native admission, bounded waves; never substitute fixtures."""
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.service_policy import ServiceLimits
    from oeis_learn.synthesis.admission import VisiblePrompt
    from oeis_learn.fresh.pipeline_jobs import JobLedger,NativeRequest,logical_job
    from oeis_learn.fresh.pipeline_scheduler import drive
    output.mkdir(parents=True,exist_ok=True)
    ledger=JobLedger();timing={};selected=[r for r in rows if r['decoded']];by_id={r['id']:r for r in prompts}
    write(output/'native-plan.json',seal(dict(version='pipeline-sizing-native-plan/v1',rows=[r['id'] for r in rows],decoded=[r['id'] for r in selected],workers=4,service_seconds=4)))
    try:
        if not selected:return
        with SynthesisClient(config['contract'],service_limits=ServiceLimits(max_inflight_batches=4),cache_root=output/'private/native-cache') as client:
            def request(i,row):
                prompt=by_id[row['prompt']];admitted=time.monotonic();until=min(deadline,admitted+4)
                logical=logical_job(contract=config['contract'],snapshot=config['options']['profile']['identity'],mode='sizing',arm='fit',prompt_id=row['id'],attempt_ordinal=0,method='generated',seed=row['seed'])
                timing[row['id']]=dict(admitted=admitted,deadline=until)
                def operation():
                    from contextlib import nullcontext
                    affinity=getattr(client.worker,'affinity',None)
                    with affinity(i) if affinity else nullcontext():
                        timing[row['id']]['dispatched']=time.monotonic()
                        result=client.fit(row['source'],VisiblePrompt(tuple(prompt['indices']),tuple(prompt['values'])),
                            policy='none' if '(param.get ' in row['source'] else 'eligible-literals/v1',deadline=until,seed=row['seed'])
                        # Exact raw payload stays private and bounded by the global
                        # disk watchdog. Public receipts retain every admission hash,
                        # independent-verifier identity and strict terminal clock.
                        raw=output/'private/native-results'/(digest(row['id'])[7:]+'.json');write(raw,result)
                        compact=dict(complete=result.get('complete',False),outcome=result['outcome'],response_sha256=digest(result),
                            admitted=[dict(source_sha256=a.get('source_sha256'),assignment_sha256=digest(a.get('assignment')),
                                verification_sha256=digest(a.get('verification')),scope=a.get('verification',{}).get('scope')) for a in result.get('admitted',[])])
                        timing[row['id']]['native_completed']=time.monotonic();return compact
                result=yield NativeRequest(logical,i,until,operation,cancel=client.worker.cancel_all,
                    on_expired=lambda:dict(complete=False,outcome='deadline',admitted=[]))
                evidence_path=output/(digest(row['id'])[7:]+'-native.json');write(evidence_path,result)
                row['native']=dict(complete=result.get('complete',False),outcome=result['outcome'],verified=bool(result.get('complete') and result.get('admitted')),
                    admitted=len(result.get('admitted',[])),evidence=file_hash(evidence_path))
                return result
            for offset in range(0,len(selected),4):
                if time.monotonic()+4>deadline:raise TimeoutError('native wave cannot fit immutable sizing parent')
                drive(None,[request(i,selected[i]) for i in range(offset,min(offset+4,len(selected)))],workers=4,
                    ledger=ledger,policies={'scope':'model-sizing','selection':'visible-only/v2'})
            records={v['job']['prompt_id']:v for v in ledger.records.values()}
            for row in selected:
                if records[row['id']]['state']!='COMMITTED' and row['native'] is not None:row['native']['verified']=False
    finally:
        for k in ledger.records:ledger.cancel(k)
        write(output/'native-jobs.json',seal(dict(version='pipeline-sizing-native-jobs/v1',records=ledger.records,timing=timing,
            boundary=ledger.boundary({'scope':'model-sizing'}))))


def gpu_role(inputs,output,config):
    import torch
    from oeis_learn.experiments.bootstrap_models import construct_sizing_models
    from oeis_learn.fresh.bigint_worker import _batch_loss,validation_loss
    from oeis_learn.fresh.pipeline_profile import parameter_digest
    from oeis_learn.decoder.bigint_generation import GenerationRequest,sample_batch
    from oeis_learn.tracking.pipeline_profile import Recorder
    from .replay import check_deadline,verify_outputs,tensor_record
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one actual HIP device required')
    p=options(config);work=read_work(inputs,config);started=time.monotonic();deadline=started+1080
    rows=attempt_plan(work['prompts']);write(output/'attempt-plan.json',seal(dict(version='pipeline-sizing-attempt-plan/v1',rows=rows)))
    ref=json.loads((inputs/'reference.json').read_text())
    validate(ref,{'version','source','contract','profile','workload','weights','oracle','counts','memory','expected'},'pipeline-sizing-reference/v1')
    if (ref!=seal(ref) or ref['profile']!=p or ref['workload']!=work['manifest']['identity'] or ref['source']!=config['source']
        or ref['contract']!=config['contract'] or ref['weights']!=file_hash(inputs/'weights.pt') or ref['oracle']!=file_hash(inputs/'oracle.pt')):
        raise ValueError('frozen sizing profile/reference/source/weights required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    timings={};success=False
    try:
        with Recorder(output/'observations',instrumented=True,render_node=config['options']['render_node'],host_counters=False,sample_bytes=16<<10):
            encoder,decoder,counts=construct_sizing_models(p)
            weights=torch.load(inputs/'weights.pt',map_location='cpu',weights_only=True)
            encoder.load_state_dict(weights['encoder']);decoder.load_state_dict(weights['decoder']);del weights
            encoder=encoder.to('cuda');decoder=decoder.to('cuda');decoder.cache_bookkeeping='hoisted/v1'
            if counts!=ref['counts']:raise ValueError('actual parameters differ from frozen reference')
            oracle=torch.load(inputs/'oracle.pt',map_location='cpu',weights_only=True)
            if tensor_record(oracle['memory'])!=ref['memory'] or {k:tensor_record(v) for k,v in oracle['expected'].items()}!=ref['expected']:raise ValueError('CPU oracle payload differs')
            encoder.eval();decoder.eval()
            with torch.no_grad():
                memory=encoder.forward_from_sequences([[int(v) for v in r['values']] for r in work['prompts'][:2]],device='cuda')
                torch.testing.assert_close(memory.cpu(),oracle['memory'],atol=2e-5,rtol=2e-4)
                tokens=torch.ones((2,128),dtype=torch.long,device='cuda');value,cache=decoder.prefill(tokens[:,:1],memory,capacity=128,backend='eager')
                actual={'1':value}
                for pos in range(2,129):
                    check_deadline(deadline);value=decoder.decode_step(tokens[:,pos-1],cache)
                    if pos in (32,128):actual[str(pos)]=value
                verify_outputs(actual,oracle['expected'],deadline=deadline)
            del oracle,cache,memory,actual,value
            initial=parameter_digest(encoder,decoder)
            optimizer=torch.optim.AdamW([*encoder.parameters(),*decoder.parameters()],lr=work['recipe']['learning']['learning_rate'])
            encoder.train();decoder.train();begin=time.monotonic();training_tokens=0
            for step in range(32):
                if time.monotonic()>=min(started+600,deadline):raise TimeoutError('fixed training/validation reservation expired')
                batch=work['training'][(step*8)%128:(step*8)%128+8]
                _batch_loss(encoder,decoder,batch,'cuda',optimizer=optimizer)
                training_tokens+=sum(len(r['tokens']) for r in batch)
            torch.cuda.synchronize();timings['training']=time.monotonic()-begin
            trained=parameter_digest(encoder,decoder)
            if trained==initial:raise RuntimeError('disposable updates did not change actual model parameters')
            begin=time.monotonic();val=validation_loss(encoder,decoder,work['validation'],'cuda',deadline=min(started+600,deadline))
            torch.cuda.synchronize();timings['validation']=time.monotonic()-begin
            if not val['complete']:raise TimeoutError('all32 validation rows required')
            encoder.eval();decoder.eval()
            with torch.no_grad():memory=encoder.forward_from_sequences([[int(v) for v in r['values']] for r in work['prompts']],device='cuda')
            by_key={(r['batch'],r['prompt']):r for r in rows}
            order=(32,8) if p['name']=='m32' else (8,32)
            for batch_size in order:
                begin=time.monotonic();arm=[by_key[(batch_size,r['id'])] for r in work['prompts']]
                for offset in range(0,32,batch_size):
                    if time.monotonic()+60+64+30>deadline:raise TimeoutError('generation group/native/export cannot fit immutable sizing parent')
                    until=time.monotonic()+60
                    requests=[GenerationRequest(memory[i:i+1],20261016+i,0,until) for i in range(offset,min(offset+batch_size,32))]
                    sampled=sample_batch(decoder,requests,batch_size=batch_size,capacity_qualified=batch_size==32,bookkeeping='hoisted/v1')
                    if len(sampled)!=len(requests):raise RuntimeError('complete sizing sampler row mapping required')
                    for row,result in zip(arm[offset:offset+batch_size],sampled):
                        row.update(state='COMPLETE' if result['complete'] else 'PARTIAL',tokens=result['tokens'],decoded=bool(result['complete']),
                            source=result['source'],source_sha256=digest(result['source']),generation_seconds=result['generation_seconds'],failure=result.get('failure'))
                    with (output/'attempt-progress.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(batch=batch_size,offset=offset,rows=arm[offset:offset+batch_size]),sort_keys=True)+'\n')
                torch.cuda.synchronize();timings[f'generation_b{batch_size}']=time.monotonic()-begin
                begin=time.monotonic();fit_generated(output/f'native-b{batch_size}',arm,work['prompts'],config,deadline);timings[f'native_b{batch_size}']=time.monotonic()-begin
            if parameter_digest(encoder,decoder)!=trained:raise RuntimeError('sampling mutated the frozen trained model')
            check_deadline(deadline);success=True
        telemetry=json.loads((output/'observations/telemetry.json').read_text())
        if telemetry['collector_error'] or not telemetry['samples']:raise RuntimeError('bounded telemetry collection required')
    except torch.cuda.OutOfMemoryError as exc:
        raise MemoryError('sizing HIP allocation exhausted; stop whole invocation') from exc
    finally:
        write(output/'attempts.json',rows)
        write(output/'progress.json',seal(dict(version='pipeline-sizing-progress/v1',profile=p,complete=success,timings=timings,counts=generation_counts(rows,max(.000001,time.monotonic()-started)))))
    seconds=time.monotonic()-started
    if seconds>1080:raise TimeoutError('sizing primary/export exceeded immutable parent')
    return dict(status='COMPLETE',profile=p,counts=counts,reference=ref['identity'],workload=work['manifest']['identity'],source=config['source'],contract=config['contract'],
        numerical_parity=True,changed_parameters=True,initial_parameters=initial,trained_parameters=trained,
        training_updates=32,training_tokens=training_tokens,training_tokens_per_second=training_tokens/timings['training'],
        training_updates_per_second=32/timings['training'],validation_rows=32,validation_rows_per_second=32/timings['validation'],
        whole_seconds=seconds,timings=timings,generation=generation_counts(rows,seconds),
        batch_generation={str(b):dict(generation_counts([r for r in rows if r['batch']==b],seconds),emitted_tokens_per_generation_second=sum(len(r['tokens']) for r in rows if r['batch']==b)/timings[f'generation_b{b}']) for b in (8,32)},
        torch_threads=dict(intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads()),
        device=dict(hip=torch.version.hip,name=torch.cuda.get_device_name()),interpretation=MEANING)


class SizingStages:
    def __init__(self,adapters):self.a=adapters;self.prepared=None;self.prepared_hash=None;self.references={}
    def __call__(self,stage,output):
        a=self.a
        if stage.id=='sizing.report':
            r=summarize_profiles(a.receipts);write(output/'model-sizing-report.json',r)
            return dict(r,stage_state='PASS' if len(r['hardware_eligible_profiles'])==3 else 'INCONCLUSIVE')
        from .benchmarks import legacy
        from .runner import CapabilityError,ResourceBreach
        from .resources import observe,resolve,reserve
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        module=legacy(a.root);host=a.host(a.args.engine)
        if stage.id=='sizing.prepare':
            if not a.args.run or not a.args.inputs:raise CapabilityError('model sizing requires existing checked --run/--inputs')
            self.prepared=output/'prepared';self.prepared.mkdir();host.select_gpu()
            m=module.prepare(a.root/a.args.run,a.root/a.args.inputs,self.prepared,image=a.images.images[(a.args.engine,'gpu')],source=digest(collect_snapshot(a.root)),render_node=host.policy['render_node'],sizing=True)
            self.prepared_hash=file_hash(self.prepared/'input/work.json')
            if a.contract['weights_ref'] and a.contract['weights_ref']['sha256']!=m['binding']['parent_weights']:raise ValueError('parent weights lineage differs')
            return dict(workload=m['identity'],interpretation=MEANING)
        key=stage.id.split('.')[1];profile=PROFILES[key];cpu=stage.adapter=='sizing-reference'
        if file_hash(self.prepared/'input/work.json')!=self.prepared_hash:raise ValueError('prospective sizing work changed before dispatch')
        inputs=output/'input';inputs.mkdir();os.link(self.prepared/'input/work.json',inputs/'work.json')
        if not cpu:
            refroot,hashes=self.references[key]
            for name,relative in [('reference.json','reference.json'),('weights.pt','private/weights.pt'),('oracle.pt','private/oracle.pt')]:
                if file_hash(refroot/relative)!=hashes[name]:raise ValueError('frozen sizing reference changed before dispatch')
                os.link(refroot/relative,inputs/name)
        facts=observe();facts['limits'].append(host.policy['memory_bytes'])
        try:
            policy=resolve(**facts,profile=a.contract['resource_policy']);reservation=reserve(policy,peak_estimate(profile))
            if reservation['reserved_bytes']>.9*policy['ceiling_bytes']:raise ValueError('ten percent unreserved capacity required')
        except ValueError as exc:raise CapabilityError(str(exc)) from exc
        host=copy.copy(host);host.policy=dict(host.policy,memory_bytes=policy['ceiling_bytes'],cpu_memory_bytes=policy['ceiling_bytes'],cpus=min(host.policy['cpus'],policy['cpus']))
        if host.policy['cpus']<6:raise CapabilityError('two Torch and four native equivalents required')
        write(output/'resource-reservation.json',dict(policy=policy,reservation=reservation,profile=profile,meaning='prospective estimate; actual aggregate watchdog mandatory'))
        mode='sizing-reference' if cpu else 'sizing-trial';cfg=dict(version='pipeline-role/v1',source=digest(collect_snapshot(a.root)),contract=a.contract['identity'],mode=mode,
            options=dict(seconds=570 if cpu else 1080,memory_ceiling=host.policy['memory_bytes'],render_node=host.policy['render_node'],profile=profile))
        write(inputs/'expected.json',cfg)
        from contextlib import nullcontext
        with nullcontext() if cpu else host.gpu_lock():
            module.container_stage(host,a.images.images[(a.args.engine,'gpu')],inputs,output/'role',mode=mode,seconds=600 if cpu else 1110,gpu=not cpu,
                argv=['python','-m','oeis_learn.pipeline_validation.role',mode])
        result=json.loads((output/'role/summary.json').read_text())
        if result.get('fatal'):raise ResourceBreach(result.get('error','sizing resource failure'))
        if result.get('status')=='FAIL':raise RuntimeError(result.get('error','sizing role failed'))
        if not cpu:
            complete_result(result,profile)
            frozen=json.loads((inputs/'reference.json').read_text())
            if (result['source']!=cfg['source'] or result['contract']!=cfg['contract'] or result['reference']!=frozen['identity']
                or result['workload']!=frozen['workload'] or result['resources']['ceiling_bytes']!=cfg['options']['memory_ceiling']):
                raise ValueError('frozen sizing source/reference/current resource envelope differs')
        elif result.get('resources',{}).get('qualified') is not True:raise ValueError('CPU reference memory observation required')
        if cpu:self.references[key]=(output/'role',{n:file_hash(output/'role'/r) for n,r in [('reference.json','reference.json'),('weights.pt','private/weights.pt'),('oracle.pt','private/oracle.pt')]})
        else:
            host_timing=json.loads((output/'role/host-timing.json').read_text())
            seconds=number(host_timing['seconds_through_cleanup'],positive=True)
            attempts=json.loads((output/'role/attempts.json').read_text())
            result=seal(dict(result,host_seconds_through_cleanup=seconds,
                generation_including_container_startup_cleanup=generation_counts(attempts,seconds)))
        return result
