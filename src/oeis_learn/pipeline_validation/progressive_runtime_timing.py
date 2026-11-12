"""Bounded full-learner runtime sampling over original gradient data only."""
import os
from pathlib import Path
import sys
import time
import traceback

from .identity import confined, digest, seal
from .preparation_recovery import retain, revise
from .progressive_workstation_state import read
from .progressive_research_inputs import checked_file, _json
from oeis_learn.generalization.progressive.contracts import reference

VERSION='progressive-runtime-timing/v1'
TARGET=128000
LEARN_SECONDS=2700
CALIBRATION_SECONDS=900
HOST_SECONDS=7*LEARN_SECONDS+4*CALIBRATION_SECONDS+1200


def scopes():
    return ([dict(id=shape+'-b16-static',kind='learner',shape=shape,batch=16,arm='static')
             for shape in ('reference','smaller_width','shallower','larger')]
        + [dict(id='reference-b'+str(b)+'-static',kind='learner',shape='reference',batch=b,arm='static') for b in (32,64)]
        + [dict(id='reference-b16-refreshed',kind='learner',shape='reference',batch=16,arm='refreshed')]
        + [dict(id=shape+'-calibration',kind='calibration',shape=shape,batch=16,arm='static')
             for shape in ('reference','smaller_width','shallower','larger')])


def gather(root, donor, training_donor, output):
    import fcntl
    from .progressive_training_timing import gather as gather_training
    from .progressive_service_calibration import validate_historical_source
    root,output=Path(root),Path(output)
    original=gather_training(root,donor,output)
    source=read(confined(root,Path(donor)/'measure-services/services/service-inputs.json'))
    if source['identity']!=original['original_inputs']:
        raise ValueError('original service inputs changed')
    validate_historical_source(root,_json(checked_file(root,source['original_request'])))
    service_review=_json(checked_file(root,original['original_review']))
    folder=confined(root,training_donor)
    if not folder.is_relative_to(root/'run-state') or output.resolve().is_relative_to(folder):
        raise ValueError('separate original training donor required')
    with (folder/'session.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=read(folder/'journal.json');refs={}
        for name in ('original-inputs','timing-plan','training-measurements','runtime-review'):
            step=journal['steps'].get('measure-training/'+name,{})
            if step.get('status')!='COMPLETE':raise ValueError('complete original training diagnostic required: '+name)
            for ref in step['outputs']:
                checked_file(root,ref);refs[ref['path']]=ref
        def owned(name):
            path=folder/'measure-training/timing'/name
            if refs.get(str(path.relative_to(root)))!=reference(root,path):
                raise ValueError('original training publication is not journal-bound')
            return read(path)
        prior_inputs=owned('training-timing-inputs.json')
        prior=owned('measurement/producer.json')
        prior_request=owned('request.json')
        prior_package=_json(checked_file(folder/'measure-training/timing',prior_inputs['package']),maximum=256<<20)
        if (prior_request['package']!=prior_inputs['package'] or prior['package']!=prior_package['identity']
                or prior_request['source']!=prior_inputs['source']
                or prior_inputs['original_inputs']!=source['identity'] or prior_request['inputs']!=prior_inputs['identity']
                or prior['status']!='PASS' or prior['device']!='actual_HIP'):
            raise ValueError('same original successful HIP training diagnostic required')
        prior_ref=reference(root,folder/'measure-training/timing/measurement/producer.json')
    retain(output/'service-budget-review.json',service_review)
    retain(output/'service-inputs.json',source)
    retain(output/'prior-training.json',prior)
    value=seal(dict(version=VERSION,source=original['source'],original_inputs=source['identity'],
        package=original['package'],prior_training=prior_ref,
        service_inputs=reference(output,'service-inputs.json'),service_review=reference(output,'service-budget-review.json'),
        scope='DIAGNOSTIC_ONLY; original training support; no panel access'))
    retain(output/'runtime-inputs.json',value)
    return value


def plan(root, output, inputs, image, recipient):
    root,output=Path(root),Path(output)
    req=seal(dict(version=VERSION,source=inputs['source'],image=image,inputs=inputs['identity'],
        original_inputs=inputs['original_inputs'],package=inputs['package'],cases=scopes(),
        target_tokens=TARGET,learner_seconds=LEARN_SECONDS,calibration_seconds=CALIBRATION_SECONDS,
        seconds=HOST_SECONDS,scope='DIAGNOSTIC_ONLY; no scientific selection or qualification'))
    claim=confined(root,root/'run-state/discovery-runtime-timing'/(inputs['original_inputs'][7:]+'.json'),exists=False)
    claim.parent.mkdir(parents=True,exist_ok=True)
    owner=seal(dict(request=req['identity'],recipient=str(Path(recipient).relative_to(root))))
    if claim.exists():
        if read(claim)!=owner:raise ValueError('original runtime diagnostic already owned; no sibling allowance reset')
    else:
        import json
        with claim.open('x') as stream:
            json.dump(owner,stream);stream.flush();os.fsync(stream.fileno())
        fd=os.open(claim.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    retain(output/'request.json',req)
    return req


def validate_request(req):
    fields={'version','identity','source','image','inputs','original_inputs','package','cases','target_tokens',
            'learner_seconds','calibration_seconds','seconds','scope'}
    if (set(req)!=fields or seal(req)!=req or req['version']!=VERSION or req['cases']!=scopes()
        or req['target_tokens']!=TARGET or req['learner_seconds']!=LEARN_SECONDS
        or req['calibration_seconds']!=CALIBRATION_SECONDS or req['seconds']!=HOST_SECONDS
        or req['scope']!='DIAGNOSTIC_ONLY; no scientific selection or qualification'):
        raise ValueError('closed finite runtime diagnostic required')
    return req


def case_package(package, case, target=TARGET):
    from oeis_learn.generalization.progressive.models import resolve_profile
    from .progressive_training import resolve_update_policy
    p=resolve_profile(package['profile']['base_recipe'],shape=case['shape'],
                      seed=package['profile']['model']['seed'],dropout=0)
    policy=resolve_update_policy(p,schedule_horizon_tokens=400000000,
                                effective_batch=case['batch'],microbatch_size=16)
    value={k:v for k,v in package.items() if k not in {'identity','refresh_by_batch'}}
    # Only this separate diagnostic retimes the unchanged shards. Scientific
    # inputs retain their original cadence and schedule. No score chooses rows.
    minimum=min(len(r['tokens']) for shard in package['shards'] for r in shard)
    import math
    cadence=max(8,8*math.ceil(math.ceil(target/(minimum*case['batch']))/(len(package['shards'])*8)))
    return seal(dict(value,profile=p,update_policy=policy,checkpoint_policy='hourly/v1',
                     refresh_updates=cadence,scope='DIAGNOSTIC_ONLY',diagnostic_fault=None))


def run_case(root, index, *, device='cuda'):
    """One process per bounded case; canonical installed train role and calibrator."""
    from .progressive_diagnostic_training import prepare
    from .progressive_acceptance import invoke_cpu
    root=Path(root);req=validate_request(read(root/'request.json'));case=req['cases'][index]
    output=root/'cases'/str(index);began=time.monotonic()
    value=dict(version=VERSION,request=req['identity'],case=case,device='actual_HIP' if device=='cuda' else 'cpu',
               status='FAIL',segments=[],scope='DIAGNOSTIC_ONLY')
    try:
        package=case_package(_json(checked_file(root,req['package']),maximum=256<<20),case,target=req['target_tokens'])
        value['package']=package['identity']
        if device=='cuda':
            from .progressive_qualification import require_hip
            from .generalization_execution import verify_execution
            require_hip();verify_execution()
            os.environ['OEIS_PROGRESSIVE_ACCEPTANCE_DEVICE']='actual_HIP'
        elif device!='cpu':raise ValueError('explicit diagnostic device required')
        os.environ['OEIS_PROGRESSIVE_TIMING']='1'
        if case['kind']=='calibration':
            from .progressive_qualification import calibrate_iteration
            measured=calibrate_iteration(package,output/'private/calibration',source=req['source'],image=req['image'],
                                         device=device,checkpoint_bytes=16<<30,seconds=CALIBRATION_SECONDS-60)
            value['calibration']=measured
        else:
            work=output/'private/learner'
            campaign,tasks,channel=prepare(work,package,target_tokens=req['target_tokens'],seconds=1200,arm=case['arm'])
            retain(output/'acceptance-limits.json',campaign['payload']['storage'])
            for ordinal,task in enumerate(tasks):
                result,produced=invoke_cpu(work,campaign,task,channel,ordinal=ordinal)
                metrics=read(produced/'role/timing.json') if (produced/'role/timing.json').exists() else None
                process=read(produced/'process.json')
                row=dict(result=result,timing=metrics,process=process)
                value['segments'].append(row)
                retain(output/'segments'/str(ordinal)/'producer.json',seal(row))
                if result['mechanical_status']!='COMPLETE':
                    for trace in (produced/'role/private').glob('*traceback*'):
                        (output/f'train-{ordinal}-traceback.log').write_bytes(trace.read_bytes()[:4<<20])
                    raise ValueError('canonical learner incomplete: '+result['reason'])
                if metrics is None:raise ValueError('canonical learner timing missing')
            if value['segments'][1]['result']['metrics']['initialization']!='resume':
                raise ValueError('full-state successor restore was not exercised')
            if case['arm']=='refreshed' and max(s['timing']['maximum_shard'] for s in value['segments'])<1:
                raise ValueError('diagnostic did not consume refreshed frontier support')
        value['status']='PASS'
    except BaseException as exc:
        value.update(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        for ordinal,trace in enumerate(output.glob('private/learner/invocation-*/role/private/*traceback*')):
            (output/f'producer-{ordinal}-traceback.log').write_bytes(trace.read_bytes()[:4<<20])
        raise
    finally:
        value['elapsed_seconds']=time.monotonic()-began
        retain(output/'producer.json',seal(value))
    return value


def run_inside(root, seconds):
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from .progressive_qualification import hardware_facts
    from .supervision import launch
    root=Path(root);req=validate_request(read(root/'request.json'))
    if (req['source']!=digest(source_snapshot()) or req['image']!=os.environ.get('OEIS_QUALIFICATION_IMAGE')
            or not 0<seconds<=HOST_SECONDS):raise ValueError('current bounded runtime source/image required')
    retain(root/'hardware.json',hardware_facts())
    began=time.monotonic();rows=[];blocked=None
    previous=read(root/'runtime-cases.json') if (root/'runtime-cases.json').exists() else None
    if previous:blocked=previous.get('device_block')
    originals={r['index']:r['producer'] for r in previous['cases'] if 'producer' in r} if previous else {}
    if previous and previous['request']!=req['identity']:raise ValueError('original case journal differs')
    for index,case in enumerate(req['cases']):
        output=root/'cases'/str(index);path=output/'producer.json'
        if path.exists():
            if index in originals:checked_file(root,originals[index])
            else:blocked='UNJOURNALED_PREVIOUS_CASE'
            producer=read(path)
            if producer.get('request')!=req['identity'] or producer.get('case')!=case:
                raise ValueError('original runtime producer changed scope')
        elif (output/'started.json').exists():
            blocked='UNPROVEN_PREVIOUS_CASE_TERMINATION'
            producer=seal(dict(status='FAIL',request=req['identity'],case=case,error=blocked))
            retain(path,producer)
        elif blocked or seconds-(time.monotonic()-began)<60:
            rows.append(dict(index=index,status='BLOCKED',reason=blocked or 'allowance exhausted'))
            revise(root/'runtime-cases.json',seal(dict(version=VERSION,request=req['identity'],cases=rows,status='RUNNING',device_block=blocked)))
            continue
        else:
            retain(output/'started.json',seal(dict(request=req['identity'],case=case)))
            limit=LEARN_SECONDS if case['kind']=='learner' else CALIBRATION_SECONDS
            code='from oeis_learn.pipeline_validation.progressive_runtime_timing import run_case;import sys;run_case(sys.argv[1],int(sys.argv[2]))'
            try:
                launch([sys.executable,'-c',code,str(root),str(index)],output/'execution',
                       seconds=min(limit,seconds-(time.monotonic()-began)-30),cleanup_seconds=15)
            except Exception as exc:
                if not path.exists():retain(path,seal(dict(status='FAIL',request=req['identity'],case=case,
                    error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())))
                process=_json(output/'execution/process.json',maximum=1<<20) if (output/'execution/process.json').exists() else {}
                if process.get('cleanup_failure') or process.get('returncode') is None:
                    blocked='UNPROVEN_CASE_CLEANUP'
            producer=read(path)
            # Fresh allocation after each process catches a poisoned device before
            # the next owner. Cleanup failure blocks GPU work unconditionally.
            if not blocked:
                try:
                    left=seconds-(time.monotonic()-began)
                    if left<=30:raise TimeoutError('no device health allowance')
                    launch([sys.executable,'-c',
                        'from oeis_learn.pipeline_validation.progressive_qualification import require_hip; t=require_hip(); x=t.ones(1,device="cuda"); assert x.item()==1; t.cuda.synchronize()'],
                        output/'health',seconds=min(120,left-15),cleanup_seconds=15)
                except Exception:blocked='DEVICE_HEALTH_FAILED'
        rows.append(dict(index=index,status=producer['status'],producer=reference(root,path)))
        revise(root/'runtime-cases.json',seal(dict(version=VERSION,request=req['identity'],cases=rows,status='RUNNING',device_block=blocked)))
    value=seal(dict(version=VERSION,request=req['identity'],cases=rows,
        status='PASS' if len(rows)==len(req['cases']) and all(r['status']=='PASS' for r in rows) and not blocked else 'FAIL',
        device_block=blocked))
    revise(root/'runtime-cases.json',value)
    return value


def report(root, inputs):
    """Unpadded scenarios, not a fabricated whole-campaign expected duration."""
    root=Path(root);req=validate_request(read(root/'request.json'))
    if req['inputs']!=inputs['identity']:raise ValueError('original runtime inputs required')
    service=read(checked_file(root,inputs['service_review']))
    rows=[];deficits=[];calibration=[]
    batch=read(root/'runtime-cases.json') if (root/'runtime-cases.json').exists() else None
    if batch and batch['request']!=req['identity']:raise ValueError('original runtime batch required')
    results={r['index']:r for r in batch['cases']} if batch else {}
    for i,case in enumerate(req['cases']):
        ref=results.get(i,{}).get('producer')
        if not ref:
            deficits.append(case['id']);continue
        p=read(checked_file(root,ref))
        if p.get('request')!=req['identity'] or p.get('case')!=case:raise ValueError('runtime case binding differs')
        if p['status']!='PASS' or p.get('device')!='actual_HIP':
            deficits.append(case['id']);continue
        if case['kind']=='calibration':
            calibration.append(dict(case=case,seconds=p['elapsed_seconds'],measurements=p['calibration']['measurements']))
            continue
        segments=p['segments']
        windows=[w for s in segments for w in s['timing']['windows'] if not w['cold']]
        if not windows or not sum(w['tokens'] for w in windows):
            deficits.append(case['id']+':no-steady-window');continue
        seconds=sum(w['elapsed_us'] for w in windows)/1e6
        tokens=sum(w['tokens'] for w in windows)
        invocations=sum(s['process']['elapsed_us'] for s in segments)/1e6
        loops=sum(s['timing']['loop_us'] for s in segments)/1e6
        components={}
        for s in segments:
            for k,v in s['timing']['components'].items():
                c=components.setdefault(k,dict(calls=0,self_seconds=0,inclusive_seconds=0))
                c['calls']+=v['calls'];c['self_seconds']+=v['self_us']/1e6;c['inclusive_seconds']+=v['inclusive_us']/1e6
        snapshots=components.get('checkpoint_save',{})
        snapshot_seconds=(snapshots.get('inclusive_seconds',0)/snapshots['calls']) if snapshots.get('calls') else 0
        rows.append(dict(case=case,tokens_per_second=tokens/seconds,measured_tokens=tokens,
            measured_steady_seconds=seconds,invocation_seconds=invocations,loop_seconds=loops,
            fixed_seconds_per_task=max(0,invocations-loops)/len(segments),snapshot_seconds=snapshot_seconds,
            windows=windows,components=components,maximum_shard=max(s['timing']['maximum_shard'] for s in segments),
            device_memory=[s['timing'].get('device_memory') for s in segments],
            scope='short trained trajectory; dropout=0; microbatch=16; observed CPU/GPU wall time'))
    if batch and batch.get('device_block'):deficits.append(batch['device_block'])
    scenarios=[];stages={};training=service['runtime_estimate']['training_tasks']
    for task in training:
        domain=task['domain']
        if domain=='baseline': candidates=[r for r in rows if r['case']['shape']=='reference' and r['case']['batch']==16]
        elif domain in {'reference','smaller_width','shallower','larger'}:
            candidates=[r for r in rows if r['case']['shape']==domain and r['case']['batch']==16]
        elif domain=='learning':candidates=[r for r in rows if r['case']['shape']=='reference']
        else:candidates=rows
        if not candidates:continue
        values=[]
        for r in candidates:
            steady=task['incremental_tokens']/r['tokens_per_second']
            # Per-task startup/final publication is measured. Periodic publications
            # occur each hour; fixed global milestone pins are added separately.
            import math
            hourly=math.floor(steady/3600)*r['snapshot_seconds']
            values.append(steady+hourly+r['fixed_seconds_per_task'])
        scenarios.append(dict(task=task['task'],stage=task['stage'],incremental_tokens=task['incremental_tokens'],
                              sampled_low_seconds=min(values),sampled_high_seconds=max(values)))
    for stage,original in service['runtime_estimate']['stages'].items():
        selected=[s for s in scenarios if s['stage']==stage]
        original_tasks=[s for s in training if s['stage']==stage]
        evaluation=original['observed_generation_projection_seconds']
        scoring=original['positive_witness_scoring_projection_seconds']
        complete=len(selected)==len(original_tasks) and scoring is not None
        low=sum(s['sampled_low_seconds'] for s in selected) if complete else None
        high=sum(s['sampled_high_seconds'] for s in selected) if complete else None
        stages[stage]=dict(training_low_seconds=low,training_high_seconds=high,
            historical_generation_seconds=evaluation,historical_scoring_seconds=scoring,
            subtotal_low_seconds=None if low is None else low+evaluation+scoring,
            subtotal_high_seconds=None if high is None else high+evaluation+scoring)
    complete=not deficits and len(scenarios)==len(training) and all(s['subtotal_high_seconds'] is not None for s in stages.values())
    estimate=dict(status='SAMPLED_SCENARIO' if complete else 'INCOMPLETE',stages=stages,
        all_eligible_incremental_tokens=sum(t['incremental_tokens'] for t in training),
        selected_round_low_seconds=sum(s['subtotal_low_seconds'] for s in stages.values()) if complete else None,
        selected_round_high_seconds=sum(s['subtotal_high_seconds'] for s in stages.values()) if complete else None,
        calibration_1296_case_sample_projection_seconds=[1296*min(c['seconds'] for c in calibration),1296*max(c['seconds'] for c in calibration)] if calibration else None,
        full_campaign_expected_seconds=None,
        interpretation='All-eligible capability/diversity/tuning scenarios, not statistical confidence intervals or caps. '
            'Training uses observed model/batch cases; historical generation uses the slowest compatible random-model sample. '
            'Learned model generation can be faster or slower. Conditional stopping is not predicted.',
        unmeasured=['shape x batch x dropout interactions and long-run thermal behavior',
                    'remaining qualification, image builds, preparation publication and reporting',
                    'global milestone pins and exact calibration inputs beyond the four samples',
                    'later conditional scaling/OEIS/components/source branches and final assessment',
                    'future generated-program services not exercised by historical witnesses',
                    'calibration samples cover batch16/dropout0 only, not worst batch64 interactions'])
    value=seal(dict(version=VERSION,status='MEASURED' if complete else 'INCOMPLETE',request=req['identity'],
        cases=rows,calibration=calibration,deficits=deficits,training_tasks=scenarios,runtime_estimate=estimate,
        executable=False,scientific_budgets_changed=False,assessment_exposed=False))
    revise(root/'runtime-review.json',value)
    return value
