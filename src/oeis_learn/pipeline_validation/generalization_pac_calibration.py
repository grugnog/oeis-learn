"""Representative actual-shape measurements and conservative pre-research freeze."""
import time
from pathlib import Path
from .identity import seal,write

REQUIRED=('short_update','long_update','cold_native','warm_native','early_shard','later_shard','save','load','reward_probe','grade','common_panel','semantic','restore','publication','cleanup')


def forecast(samples,*,updates,arm,existing):
    from oeis_learn.generalization.budgets import pac_work_allowance
    from .identity import number
    from oeis_learn.generalization.resources import pac_diagnostic_reservation
    if set(samples)!=set(REQUIRED) or any(not values for values in samples.values()):raise ValueError('complete representative calibration required')
    for values in samples.values():
        for value in values:number(value,positive=True)
    worst={k:max(v) for k,v in samples.items()}
    update=max(worst['short_update'],worst['long_update'])*updates
    diagnostics=pac_diagnostic_reservation(updates,arm=arm,costs=dict(reward_probe=worst['reward_probe'],grade=worst['grade'],
        common_loss_probe=worst['common_panel']/256,semantic_attempt=max(5,worst['semantic']),mode_rng_restore=worst['restore'],publication=worst['publication'],cleanup=worst['cleanup']),
        update_seconds=update,initial_final_overhead=2*worst['common_panel'],existing=existing.get('diagnostics',0))
    learner=pac_work_allowance('learner',forecast=update+diagnostics['seconds'],startup=worst['load'],publication=worst['save'],existing=existing.get('learner',0))
    return dict(version='pac-calibration-forecast/v1',learner=learner,diagnostics=diagnostics,
        generation=pac_work_allowance('generation',forecast=None,startup=worst['load'],publication=worst['publication'],existing=existing.get('generation',0)),
        preparation=pac_work_allowance('preparation',forecast=None,startup=worst['cold_native'],publication=worst['publication'],existing=existing.get('preparation',0)),
        samples={k:dict(count=len(v),worst_seconds=max(v)) for k,v in samples.items()},guaranteed_worst_case=False,
        uncertainty='finite representative samples; cold/warm and early/later costs remain distinct',update_forecast_seconds=update)


def calibrate(inputs,output,settings,*,seconds):
    import json,torch
    from oeis_learn.pipeline_validation.generalization_training import run_torch,PacPreparedBatches,torch_update
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.fresh.bigint_worker import models,client
    from .generalization_curriculum import observe_targets,semantic_probe
    from .generalization_execution import verify_execution
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.generalization.budgets import charge_tokens
    from oeis_learn.decoder.bigint_codec import decode_body
    verify_execution()
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('actual HIP required for representative calibration')
    if set(settings)!={'recipe','source','environment','bundle','study'}:raise ValueError('closed production calibration request')
    output=Path(output);deadline=time.monotonic()+seconds;samples={k:[] for k in REQUIRED}
    def measured(kind,callback):
        if time.monotonic()>=deadline:raise TimeoutError('calibration allowance exhausted')
        print('calibration: '+kind,flush=True);torch.cuda.synchronize();start=time.monotonic()
        value=callback();torch.cuda.synchronize();samples[kind].append(max(time.monotonic()-start,1e-9));return value
    bundle=settings['bundle'];root=inputs['training_targets']
    batches=PacPreparedBatches(root,bundle,arm='C',batch_size=32,updates=25000000)
    literals=PacPreparedBatches(root,bundle,arm='A',batch_size=32,updates=25000000)
    ordered=sorted(batches.rows.values(),key=lambda r:len(r['tokens']))
    selected=[ordered[0],ordered[-1]]
    for kind,index in [('early_shard',0),('later_shard',len(ordered)-1)]:measured(kind,lambda index=index:batches._row(index))
    with client(dict(recipe=settings['recipe'],contract_sha256=settings['source']),output/'private/native') as native:
        for row in selected:
            literal=literals.rows[row['id']]
            # Learner rows intentionally omit source text. Their literal tokens
            # were verified against the sealed source by PacPreparedBatches.
            source=decode_body(literal['tokens'])
            def execute():
                value=native.execute(source,list(map(str,range(100))),deadline=min(deadline,time.monotonic()+4))
                if value['outcome']!='ok':raise ValueError('representative native execution incomplete')
                return value
            measured('cold_native',execute);measured('warm_native',execute)
    config=dict(study=settings['study']['identity'],arm='C',seed=settings['study']['seed'],recipe=settings['recipe'],shards=bundle['parameterized'],
        data_mode='static',batch_size=32,target_tokens=25000000,milestones=[],seconds=max(60,deadline-time.monotonic()),device='cuda',checkpoint_policy='hourly/v1',
        pac=dict(bundle=bundle,purpose='research',source=settings['source'],environment=settings['environment'],initializer=settings['study']['initializer'],
            observation_seconds=min(7200,max(10,(deadline-time.monotonic())/2)),publication_seconds=30))
    run=run_torch(root,output/'private/production',config,stop_after_updates=64)
    if run['updates']!=64 or run['pac']['valid_rewards']<6:raise ValueError('production feedback cycle incomplete')
    samples['save'].append(run['timing']['max_checkpoint_seconds'])
    store=GeneralizationCheckpointStore(output/'private/production/checkpoints',run['compatibility'],25000000)
    loaded=measured('load',lambda:store.load(run['checkpoint']))
    recipe=json.loads(json.dumps(settings['recipe']));recipe['model']['seed']=settings['study']['seed']
    encoder,decoder,_=models(recipe,'cuda');encoder.load_state_dict(loaded['states']['weights']['encoder']);decoder.load_state_dict(loaded['states']['weights']['decoder'])
    optimizer=torch.optim.AdamW([*encoder.parameters(),*decoder.parameters()],lr=recipe['learning']['learning_rate'])
    step=torch_update(encoder,decoder,optimizer,device='cuda')
    for kind,row in zip(('short_update','long_update'),selected):
        family=batches.row_families[row['id']];batch=[row]+[batches.rows[r] for r in batches.families[family] if r!=row['id']][:31]
        for _ in range(2):measured(kind,lambda batch=batch:step(batch))
        probes=[batches.probe_rows[i] for i in batches.probes[family][:2]]
        measured('reward_probe',lambda:observe_targets(encoder,decoder,probes,device='cuda'))
        measured('grade',lambda:observe_targets(encoder,decoder,[row],device='cuda'))
        probe=batches.probes[family][0]
        measured('semantic',lambda:semantic_probe(encoder,decoder,batches.probe_rows[probe],batches.probe_truth[probe],device='cuda',seed=7,recipe=recipe,cache_root=output/'private/semantic',seconds=5))
    measured('common_panel',lambda:[observe_targets(encoder,decoder,[batches.probe_rows[i] for i in ids],device='cuda') for ids in batches.probes.values()])
    measured('restore',lambda:torch.set_rng_state(torch.get_rng_state()))
    measured('publication',lambda:write(output/'timing-sample.json',run['timing']))
    measured('cleanup',lambda:optimizer.zero_grad(set_to_none=True))
    minimum=min(charge_tokens([r['tokens']],bos=1,pad=0) for r in [*ordered,*literals.rows.values()])*32
    updates=(25000000+minimum-1)//minimum
    forecasts={arm:forecast(samples,updates=updates,arm=arm if arm!='A' else 'B',existing={}) for arm in settings['study']['arms']}
    result=seal(dict(status='PASS',version='pac-calibration/v1',source=settings['source'],environment=settings['environment'],
        production_recipe=settings['study']['model'],updates=64,tokens=run['tokens'],valid_rewards=run['pac']['valid_rewards'],
        forecasts=forecasts,samples=samples,forecast_updates=updates,weights_are_research_initializers=False))
    write(output/'calibration.json',result);return result


def verify_calibration(record,*,source,environment,arms,model=None):
    """Reconstruct every forecast from the complete bounded measured samples."""
    fields={'version','identity','status','source','environment','production_recipe','updates','tokens','valid_rewards',
            'forecasts','samples','forecast_updates','weights_are_research_initializers'}
    if (set(record)!=fields or seal(record)!=record or record['version']!='pac-calibration/v1' or record['status']!='PASS'
        or record['source']!=source or record['environment']!=environment or set(record['forecasts'])!=set(arms)
        or record['weights_are_research_initializers'] is not False or record['updates']!=64 or record['valid_rewards']<6
        or type(record['tokens']) is not int or record['tokens']<1 or type(record['forecast_updates']) is not int or record['forecast_updates']<1
        or model is not None and record['production_recipe']!=model):raise ValueError('complete compatible measured calibration required')
    for arm in arms:
        expected=forecast(record['samples'],updates=record['forecast_updates'],arm='B' if arm=='A' else arm,existing={})
        if expected!=record['forecasts'][arm]:raise ValueError('calibration forecast differs from actual samples')
    return record


def apply_calibration(study,runtime,record):
    """Raise and seal operational limits before any research score/update."""
    import copy
    from .identity import digest
    from .generalization_execution import bind_pac,pac_budget_projection
    from .identity import number
    verify_calibration(record,source=study['source'],environment=digest(dict(images=runtime['images'],engine=runtime['engine'],backend=study['backend'])),arms=study['arms'],model=study['model'])
    runtime=copy.deepcopy(runtime);study=copy.deepcopy(study);e=runtime['envelope']
    for name,key in [('learner','learner'),('resume','learner'),('generation','generation'),('preparation','preparation')]:
        work=max(e[name]['work'],max(v[key]['seconds'] for v in record['forecasts'].values()));number(work,positive=True)
        e[name]['work']=work
        samples=record['samples']
        e[name]['startup']=max(e[name]['startup'],3*max(samples['load'])+60)
        e[name]['publication']=max(e[name]['publication'],3*max(samples['save'])+60)
        e[name]['outer']=sum(e[name][k] for k in ('work','startup','publication','projection','cleanup'))
    e['diagnostics_seconds']=max(e['diagnostics_seconds'],max(v['diagnostics']['seconds'] for v in record['forecasts'].values()))
    from .generalization_execution import PAC_STAGES
    e['campaign']=max(e['campaign'],sum(e[n]['outer'] for n in PAC_STAGES)+(len(study['arms'])-1)*sum(e[n]['outer'] for n in ('learner','resume','generation','proof','export'))+3600)
    study['budget_graph']=pac_budget_projection(e,arms=study['arms'])
    return bind_pac(study,runtime)
