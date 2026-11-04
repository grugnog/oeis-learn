#!/usr/bin/env python3
"""Temporary current-source qualification and paired training throughput probe.

Uses canonical CLI stages. Never resumes old weights, evaluates protected tasks,
changes research budgets, or launches full research. Archives contain metadata only.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(HERE)]
import fresh_pilot as api
from oeis_learn.pipeline_validation.identity import confined,seal,write,file_hash
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact
from fresh_pilot_training_policy import assess,budget

LIMITS=dict(preflight=1200,prepare=3600,**{'train-static':3600,'train-refreshed':3600})


def original_job(root,previous,engine):
    """Historical bytes are candidate provenance, never current qualification."""
    from fresh_pilot_resume import document,command_receipt
    old=confined(root,previous);job=document(old/'private/smoke-job.json')
    if job['version']!='fresh-pilot-smoke-job/v1' or job['engine']!=engine:raise ValueError('original prepared job/engine required')
    prepared=old/'research/prepare';terminal,inventory=command_receipt(prepared,'prepare')
    if terminal['result']['manifest']!=job['manifest']:raise ValueError('original prepared manifest differs')
    base=confined(root,job['manifest']).parent
    cfg=json.loads(verify_artifact(root,job['blueprint']))
    verify_artifact(root,job['design'])
    from oeis_learn.pipeline_validation.generalization_runtime import load
    _,study,runtime=load(root,job['manifest'])
    if runtime['preparation'] is not None:raise ValueError('resolved original preparation required')
    for key,ref in [('recipe',runtime['recipe']),('descriptors',study['data']['descriptors']),('exclusions',study['data']['exclusions'])]:
        verify_artifact(base,ref)
        original=reference(confined(root,cfg[key]),root)
        if any(original[k]!=ref[k] for k in ('bytes','sha256')):raise ValueError('original '+key+' changed')
    for cell in cfg['cells']:
        if cell['role']=='training':continue
        path=confined(base,'cells/'+cell['name']+'/private/selected.json')
        if str(path.relative_to(prepared)) not in inventory:raise ValueError('protected source missing from original inventory')
    if cfg['initializer'] is not None or cfg['seeds']!=[20261016]:raise ValueError('original fresh paired design required')
    return job,terminal['identity']


def qualified(root,args,target):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.pipeline_validation.generalization_qualification_io import public_closure
    from oeis_learn.pipeline_validation.generalization_setup import assemble_acceptance
    from oeis_learn.container_runtime import Engine
    if args.qualified:
        folder=confined(root,args.qualified)
    else:
        import generalization_hip_serial as serial
        from oeis_learn.pipeline_validation.generalization_lifecycle import scope
        from oeis_learn.generalization.resources import CAPS
        if not args.cpu_image or not args.gpu_image:raise ValueError('current prebuilt CPU and GPU images required')
        engine=Engine(args.engine);images=[engine.image(args.cpu_image),engine.image(args.gpu_image)]
        setup=assemble_acceptance(root,imported=args.imported,images=images,output=str((target/'acceptance-inputs').relative_to(root)))
        folder=target/'acceptance'
        frozen=serial.plan(args.engine,setup['session']['manifest'],str(folder.relative_to(root)),image=images[0])
        for item in frozen:
            i=item['argv'].index('generalization');item['argv'][i:i]=['--memory-gib','64','--cpus','4']
        with scope(folder,CAPS,extra_roots=[target/'acceptance-inputs']):code=serial.execute(frozen,folder)
        if code:raise RuntimeError('current-source acceptance failed; no throughput learner launched')
        folder=folder/'report/qualified'
    authority=api.read(folder/'qualification.json')
    if seal(authority)!=authority or authority.get('version')!='generalization-qualification/v2':raise ValueError('original sealed qualification required')
    public_closure(folder,authority)
    if json.loads(verify_artifact(folder,authority['source_closure']))!=collect_snapshot(root):raise ValueError('qualification source changed; rebuild and requalify')
    if authority['resources']['memory_bytes']!=64<<30 or authority['resources']['cpus']!=4:raise ValueError('64 GiB/four CPU qualification required')
    engine=Engine(args.engine)
    if any(engine.image(i)!=i for i in authority['images']):raise ValueError('qualified images changed or unavailable')
    return folder,authority


def verify_plan(base,study,runtime,protocol):
    from fresh_pilot_smoke import batch_charges
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.pipeline_validation.identity import digest
    if seal(protocol)!=protocol or protocol['version']!='fresh-pilot-training-probe/v1':raise ValueError('frozen probe protocol required')
    for arm,count in (('static',1),('refreshed',2)):
        view=runtime['training'][arm]
        if view['batch_size']!=16 or len(view['shards'])!=count:raise ValueError('production probe batch/shard layout required')
        for i,ref in enumerate(view['shards']):
            rows=json.loads(verify_artifact(base,ref,max_bytes=64<<20))
            if (digest([r['source'] for r in rows])!=protocol['training_sources'][i] or
                any(r['tokens']!=encode_body(r['source']) for r in rows) or
                batch_charges(rows,256)!=protocol['batch_charges'][i]):raise ValueError('native probe selection/encoding differs')
    if (protocol['minimum_updates']!=24 or protocol['first_shard_tokens']!=sum(protocol['batch_charges'][0]) or
        budget(protocol['batch_charges'])!=study['budgets']['target_tokens'] or study['budgets']['target_tokens']!=protocol['target_tokens']):
        raise ValueError('probe exposure budget differs')


def execute(args,target,record):
    from fresh_pilot_smoke import reduced_blueprint
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.pipeline_validation.generalization_inputs import assemble_study
    from oeis_learn.pipeline_validation.generalization_runtime import load
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope,guard
    from oeis_learn.pipeline_validation.generalization_acceptance import training_proof
    from foundation_local import parser,Host
    root=api.ROOT
    rel=lambda p:str(p.relative_to(root))
    from fresh_pilot_resume import disk_preflight
    record['disk']=disk_preflight(root)
    job,record['original_preparation']=original_job(root,args.prepared_run,args.engine)
    record['acceptance_archive']=None if args.qualified else rel((target/'acceptance').with_suffix('.zip'))
    qualification,authority=qualified(root,args,target)
    record['qualification']=reference(qualification/'qualification.json',root)
    record['images']=authority['images']
    cfg,protocol=reduced_blueprint(root,job,target/'private/design',probe=True)
    record['protocol']=protocol
    write(target/'private/source.json',collect_snapshot(root))
    cfg.update(images=authority['images'],feature=rel(qualification/'feature.json'),
        source_closure=rel(target/'private/source.json'),evaluation=None)
    write(target/'private/current-blueprint.json',cfg)
    manifest=assemble_study(root,cfg,rel(target/'private/inputs'))['manifest']
    image=authority['images'][0]
    def command(stage,extra):
        output=target/'research'/stage
        code=api.invoke(api.canonical(args.engine,image,stage.split('-')[0],rel(output),extra),
            target/'private/logs'/stage,seconds=LIMITS[stage])
        result=api.read(output/'result.json') if (output/'result.json').is_file() else None
        status=api.terminal_status(code,result) if result is None or seal(result)==result else 'FAIL'
        record['stages'].append(dict(stage=stage,status=status,exit_code=code,
            receipt=reference(output/'result.json',root) if result is not None else None))
        write(target/'progress.json',seal(record),exclusive=False)
        if status!='PASS':raise RuntimeError('probe '+stage+' did not pass')
        return output,result
    command('preflight',['--manifest',manifest])
    _,result=command('prepare',['--manifest',manifest,'--qualification',rel(qualification)])
    manifest=result['result']['manifest'];base,study,runtime=load(root,manifest)
    verify_plan(base,study,runtime,protocol)
    host=Host(parser().parse_args(api.canonical(args.engine,image,'preflight','unused',['--manifest',manifest])[2:]))
    results={};initials=[];record['training']={};record['proofs']={}
    with scope(target/'research',runtime['storage']),guard(target/'research/private/disk'):
        for arm in ('static','refreshed'):
            output,result=command('train-'+arm,['--manifest',manifest,'--qualification',rel(qualification),'--arms',arm,'--seeds','20261016'])
            runs=result['result']['runs']
            if len(runs)!=1:raise ValueError('one paired probe learner required')
            training=runs[0]['result']
            if training.get('checkpoint_policy')!=protocol['checkpoint_policy']:raise ValueError('measured checkpoint policy differs from protocol')
            learner=confined(output/'training',runs[0]['root'])
            checkpoint,_=api.latest_checkpoint(learner,training,study=study['identity'],arm=arm)
            initials.append(checkpoint['initialization'])
            # Original metadata only: no weights, sources, terms or pending batches.
            record['training'][arm]={k:training[k] for k in ('status','tokens','charged_tokens','discarded_tokens','checkpoint_policy','pending_tokens','updates','timing')}
            from fresh_pilot_report import training_timing
            record['training'][arm]['timing']=training_timing(training['timing'])
            results[arm]=training
            proof=training_proof(host,root,target/('proof-'+arm),runtime,study,[(learner,training)])
            record['proofs'][arm]=dict(status=proof['status'])
            if proof['status']!='PASS':raise ValueError('finite changed-weight proof failed')
    if len(set(initials))!=1:raise ValueError('probe initial weights differ')
    record['initialization']=initials[0]
    record['forecast']=assess(results,protocol['first_shard_tokens'],research_milestones=protocol['research_milestones'])
    record['status']=record['forecast']['status']


def package(target,record):
    """Bounded aggregate-only ZIP; no recursive inclusion of private run trees."""
    from fresh_pilot_report import preparation_diagnostics,training_timing
    try:record['preparation']=preparation_diagnostics(target)
    except (OSError,ValueError,KeyError):record['preparation']=dict(status='UNAVAILABLE')
    # Partial worker timing survives even when a canonical command times out.
    record['partial_timings']={}
    for arm in ('static','refreshed'):
        folder=target/'research'/('train-'+arm)/'training'
        paths=list(folder.glob('*/execution/role/private/learner/timing.json'))
        if len(paths)==1:
            try:
                value=api.read(confined(folder,str(paths[0].relative_to(folder))),cap=1<<20)
                if value.get('version')=='generalization-training-timing/v1':record['partial_timings'][arm]=training_timing(value)
            except (OSError,ValueError):pass
    write(target/'result.json',seal(record),exclusive=False)
    archive=target/'report.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(target/'result.json','result.json')
        z.writestr('inventory.json',json.dumps(dict(version='training-probe-archive/v1',files=[reference(target/'result.json',target)]),sort_keys=True))
    with zipfile.ZipFile(archive) as z:
        import hashlib
        raw=z.read('result.json');expected=reference(target/'result.json',target)
        if len(raw)!=expected['bytes'] or 'sha256:'+hashlib.sha256(raw).hexdigest()!=expected['sha256']:raise ValueError('probe archive verification failed')
    return archive


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=('docker','podman'),default='podman')
    p.add_argument('--prepared-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--qualified');p.add_argument('--cpu-image');p.add_argument('--gpu-image')
    p.add_argument('--imported',default='run-state/generalization/uniform-import')
    args=p.parse_args(argv);root=api.ROOT;target=confined(root,args.output,exists=False)
    old=confined(root,args.prepared_run)
    if target.exists() or old==target or old in target.parents or target in old.parents:raise ValueError('fresh disjoint output required')
    target.mkdir(parents=True)
    began=time.monotonic()
    record=dict(version='fresh-pilot-training-probe-result/v1',status='FAIL',stages=[],scientific_status='UNMEASURED',
        automatic_research=False,qualification_evidence=False,helpers={n:file_hash(HERE/n) for n in ('fresh_pilot_training_probe.py','fresh_pilot_training_policy.py','fresh_pilot_smoke.py','fresh_pilot.py','fresh_pilot_resume.py','fresh_pilot_report.py')})
    try:execute(args,target,record)
    except (Exception,KeyboardInterrupt) as exc:
        record['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL'
        record['error']=type(exc).__name__+': '+str(exc)[:1024]
    record['elapsed_seconds']=time.monotonic()-began
    archive=package(target,record)
    print(json.dumps(dict(status=record['status'],archive=str(archive.relative_to(root)),scientific_status='UNMEASURED'),indent=2),flush=True)
    return api.EXIT_CODES[record['status']]


if __name__=='__main__':raise SystemExit(main())
