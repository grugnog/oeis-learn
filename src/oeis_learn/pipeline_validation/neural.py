"""Isolated disposable neural workload, using the existing trainer and checkpoint store.

Only selected training diagnostics enter the GPU role. Hidden scoring is a later
CPU-only role. This module cannot create a reusable learning run or expose finals.
"""
from copy import deepcopy
import json
from pathlib import Path
import time
from .identity import seal,write,digest,file_hash


def gpu(inputs,output,config):
    from .attempts import AttemptEvidence
    with AttemptEvidence(output/'attempt-ledger.jsonl') as evidence:
        return _gpu(inputs,output,config,evidence)


def _gpu(inputs,output,config,evidence):
    import torch
    from oeis_learn.fresh.bigint_worker import models,_batch_loss,validation_loss,client
    from oeis_learn.fresh.bigint_diagnostics import generate
    from oeis_learn.fresh.pipeline_profile import snapshot,restore,compact_group,parameter_digest
    from oeis_learn.tracking.pipeline_profile import Recorder
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('one actual HIP device and finite updates required; no CPU fallback')
    work=json.loads((inputs/'work.json').read_text());manifest=work['manifest']
    if digest({k:v for k,v in manifest.items() if k!='identity'})!=manifest['identity']:
        raise ValueError('diagnostic manifest identity mismatch')
    if file_hash(inputs/'weights.pt')!=manifest['weights_sha256'] or digest(work['recipe'])!=manifest['recipe_sha256']:
        raise ValueError('frozen weights/recipe mismatch')
    if digest({k:work[k] for k in ('prompts','training','validation')})!=manifest['payload_sha256']:
        raise ValueError('visible diagnostic payload mismatch')
    options=config['options'];recipe=deepcopy(work['recipe']);recipe.pop('identity',None)
    policy=options['policy']
    if policy is not None:
        from oeis_learn.fresh.pipeline_jobs import validate_policy
        recipe['pipeline']=validate_policy(policy)
    recipe['identity']=digest(recipe)
    rows={r['id']:r for r in work['prompts']}
    ids=options['prompt_ids'] or manifest['prompt_ids']
    if len(set(ids))!=len(ids) or not ids or set(ids)-set(rows):
        raise ValueError('frozen prompt subset required')
    if options['assessment'] not in ('fixed-work','fixed-budget','cost-full') or options['bookkeeping'] not in ('baseline','hoisted/v1'):
        raise ValueError('unknown benchmark assessment/decoder')
    counts_per_mode=dict(correct=8,shuffled=8 if options['assessment']=='cost-full' else 1,absent=8 if options['assessment']=='cost-full' else 1,template=8)
    evidence.plan(ids,counts_per_mode,parent=digest(dict(contract=config['contract'],weights=manifest['weights_sha256'])))
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    started=time.monotonic();deadline=started+options['seconds']
    encoder,decoder,counts=models(recipe,'cuda')
    weights=torch.load(inputs/'weights.pt',map_location='cpu',weights_only=True)
    encoder.load_state_dict(weights['encoder']);decoder.load_state_dict(weights['decoder']);del weights
    decoder.cache_bookkeeping=options['bookkeeping']
    optimizer=torch.optim.AdamW([*encoder.parameters(),*decoder.parameters()],lr=recipe['learning']['learning_rate'])
    initial=snapshot(encoder,decoder,optimizer)
    initial_hash=parameter_digest(encoder,decoder)
    timings={};result={};update_times=[];validation_times=[]
    with Recorder(output/'observations',instrumented=options['telemetry'],render_node=manifest['render_node']) as recorder:
        encoder.train();decoder.train()
        begin=time.monotonic()
        for update in range(500):
            if time.monotonic()>=deadline:raise TimeoutError('disposable update workload deadline')
            offset=(update*8)%128
            step_start=time.monotonic()
            _batch_loss(encoder,decoder,work['training'][offset:offset+8],'cuda',optimizer=optimizer)
            update_times.append(time.monotonic()-step_start)
        torch.cuda.synchronize()
        timings['updates']=time.monotonic()-begin
        if parameter_digest(encoder,decoder)==initial_hash:
            raise RuntimeError('actual finite updates did not change parameters')
        begin=time.monotonic();validation=validation_loss(encoder,decoder,work['validation'],'cuda',deadline=deadline,measure=lambda seconds,n:validation_times.append(seconds/n))
        torch.cuda.synchronize();timings['validation']=time.monotonic()-begin
        trained=snapshot(encoder,decoder,optimizer)
        restore(initial,encoder,decoder,optimizer)
        settings=dict(contract_sha256=config['contract'],recipe=recipe)
        begin=time.monotonic()
        prewarm_seconds=0
        with client(settings,output/'private') as synthesis:
            if options.get('cache_state')=='warm':
                from oeis_learn.synthesis.admission import VisiblePrompt
                warm_start=time.monotonic()
                for job in options['prewarm']:
                    if job['prompt'] not in ids:continue
                    row=rows[job['prompt']]
                    synthesis.fit(job['source'],VisiblePrompt(tuple(row['indices']),tuple(row['values'])),
                        policy='none' if '(param.get ' in job['source'] else 'eligible-literals/v1',
                        deadline=min(deadline,time.monotonic()+4),seed=job['seed'])
                prewarm_seconds=time.monotonic()-warm_start
                begin=time.monotonic()
            result=generate(encoder,decoder,[rows[k] for k in ids],recipe,synthesis,deadline=deadline,
                diagnostic_attempts=None if options['assessment']=='cost-full' else dict(correct=8,shuffled=1,absent=1,template=8),
                portfolio_seconds=180 if options['assessment']=='fixed-work' else 60,evidence=evidence.event)
            native_stats=synthesis.worker.stats
        torch.cuda.synchronize();timings['generation_native']=time.monotonic()-begin
        prewarm=[dict(prompt=key,source=item['source'],seed=item['seed']) for key,row in result['records'].items() for mode,portfolio in row['details'].items() if mode!='template' for item in portfolio['details'] if item['complete'] and item['source']]
        write(output/'prewarm.json',prewarm)
        selections,attempts=compact_group(result)
        from .workloads import material_strata
        timings.update(material_strata(attempts,rows))
        started_attempts=len(attempts)
        seen={(a['prompt'],a['mode'],a['position']) for a in attempts}
        for key in ids:
            for mode,count in counts_per_mode.items():
                for position in range(count):
                    if (key,mode,position) not in seen:
                        attempts.append(dict(prompt=key,mode=mode,position=position,seed=None,tokens=0,tokens_sha256=digest([]),source_sha256=None,generation_seconds=0,batch_size=0,generation=dict(complete=False,outcome='unstarted',reason='parent deadline'),fitting=dict(complete=False,outcome='unstarted',reason='parent deadline'),raw=dict(complete=False,outcome='unstarted',reason='parent deadline')))
        attempts.sort(key=lambda a:(ids.index(a['prompt']),['correct','shuffled','absent','template'].index(a['mode']),a['position']))
        write(output/'private/selections.json',selections)
        write(output/'attempts.json',attempts)
        complete_ids=[key for key,row in result['records'].items() if row['complete']]
        # Semantic source/job seals are compared, never cost-bearing v1 seals.
        semantic={key:{mode:None if selected is None else dict(
            source=selected['selected']['source_sha256'],assignment=selected['assignment'],
            original=selected['original_source_sha256'],transformed=digest(selected['transformed_source']),
            attempt=selected['selected'].get('attempt_ordinal'),snapshot=selected['selected'].get('provenance',{}).get('snapshot'))
            for mode,selected in row['selections'].items()} for key,row in result['records'].items()}
        write(output/'selections-public.json',semantic)
        begin=time.monotonic()
        checkpoint=checkpoint_roundtrip(output/'private/checkpoints',recipe,trained,config['contract'],result.get('pipeline_boundary'))
        timings['checkpoint']=time.monotonic()-begin
    summary=seal(dict(version='pipeline-neural-trial/v1',status='COMPLETE' if len(complete_ids)==len(ids) and validation['complete'] else 'PARTIAL',
        source=config['source'],contract=config['contract'],weights=manifest['weights_sha256'],
        options={k:v for k,v in options.items() if k!='prewarm'},prewarm_identity=digest(options.get('prewarm',[])),prewarm_seconds=prewarm_seconds,counts=counts,updates=500,validation=validation,
        scheduled_attempts=len(ids)*sum(counts_per_mode.values()),attempted=started_attempts,complete_ids=complete_ids,prompt_ids=ids,
        original_denominator=len(manifest['prompt_ids']),tokens=digest([(r['prompt'],r['mode'],r['position'],r['tokens_sha256']) for r in attempts]),
        semantic=digest(semantic),timings=timings,checkpoint=checkpoint,native=native_stats,
        cost_components=dict(training_p95=max(update_times),teacher_forcing_p95=max(validation_times,default=0),
            generation_p95=max((a['generation_seconds'] for a in attempts if a['mode']!='template'),default=0),
            fit_p95=max((a.get('native_seconds') or 0 for a in attempts),default=0),
            reference_p95=max((a.get('native_seconds') or 0 for a in attempts),default=0),checkpoint_p95=timings['checkpoint']),
        seconds=time.monotonic()-started,device=dict(torch=torch.__version__,hip=torch.version.hip,name=torch.cuda.get_device_name()),
        authority='disposable diagnostic; weights-only parent; no research resume'))
    return summary


def checkpoint_roundtrip(root,recipe,state,parent,boundary):
    from oeis_learn.fresh.bigint_state import CheckpointStore
    from oeis_learn.synthesis.state import ExperimentContract,BudgetLedger
    from oeis_learn.fresh.pipeline_jobs import JobLedger
    h=digest({'diagnostic':parent,'recipe':recipe['identity']})
    contract=ExperimentContract(data=h,split=h,model=h,inputs=h,decoder=h,solver=h,runtime=h,
        resources=h,policy=recipe['identity'],track='generic_only',foundation=None,seed=recipe['model']['seed'],budget=600)
    store=CheckpointStore(root,contract,policies=recipe.get('pipeline'))
    selection={'diagnostic':True,'nonresumable_research':True}
    if recipe.get('pipeline'):
        selection['pipeline_boundary']=boundary or JobLedger().boundary(recipe['pipeline'])
    from oeis_learn.fresh.learning_policy import Curriculum
    from oeis_learn.fresh.replay import SynthesisReplay
    states=dict(weights=dict(encoder=state['encoder'],decoder=state['decoder']),optimizer=state['optimizer'],
        rng={k:state[k] for k in ('python','numpy','torch','cuda')},data_order={'completed':500,'cursor':4000%128,'size':128,'order':'fixed-sequential-diagnostic'},selection=selection,
        curriculum=Curriculum().snapshot(),replay=SynthesisReplay().snapshot())
    pointer=store.save(states,BudgetLedger(600))
    loaded=store.load(pointer)
    import torch
    for model in ('encoder','decoder'):
        if not all(torch.equal(value,loaded['states']['weights'][model][key]) for key,value in states['weights'][model].items()):
            raise RuntimeError('diagnostic full checkpoint roundtrip mismatch')
    return dict(pointer=pointer,authority='diagnostic only; no DurableRun; not research resumable')


def score(inputs,output,config):
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.fresh.bigint_diagnostics import score as score_visible
    work=json.loads((inputs/'score.json').read_text())
    truth=json.loads((inputs/'truth.json').read_text())
    selections=json.loads((inputs/'selections.json').read_text())
    if digest(truth)!=work['truth'] or digest(selections)!=work['selections'] or set(truth)!=set(selections):
        raise ValueError('scorer truth/selections identity mismatch')
    from oeis_learn.synthesis.contracts import ResourceProfile
    with SynthesisClient(config['contract'],limits=ResourceProfile(**work['resources']),cache_root=output/'private/cache') as client:
        result=score_visible(selections,truth,client,deadline=time.monotonic()+config['options']['seconds'])
    return dict(status='COMPLETE' if all(v['complete']==v['denominator'] for v in result['splits'].values()) else 'PARTIAL',result=result)
