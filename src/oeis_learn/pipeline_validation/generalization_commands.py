"""Canonical host orchestration; data/model work stays in installed roles."""
import json
import shutil
from pathlib import Path
from .identity import confined,seal,write,digest
from .generalization_runtime import load,execution_refs,learner_config
from .generalization_preparation import reference
from .generalization_roles import project_and_invoke
from oeis_learn.generalization.contracts import verify_artifact,artifact_refs
from oeis_learn.generalization.interfaces import EXIT_CODES


def _document(root,ref):return json.loads(verify_artifact(root,ref,max_bytes=1<<20))


def _copy(destination,inputs):
    """Combine explicit projections, rejecting path collisions before a launch."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    refs={}
    for source,ref in inputs:
        payload=verify_artifact(source,ref)
        target=confined(destination,ref['path'],exists=False)
        if ref['path'] in refs and refs[ref['path']]!=ref:raise ValueError('conflicting projected input')
        if target.exists():
            if target.read_bytes()!=payload:raise ValueError('conflicting projected bytes')
        else:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        refs[ref['path']]=ref
    return list(refs.values())


def _validate(host,workspace,output,base,study,runtime,*,seconds=300):
    refs=execution_refs(base,study,runtime,preparation=runtime['preparation'] is not None)
    result=project_and_invoke(host,runtime['images'][0],workspace,'preparation',
        {'mechanical_sources':refs,'isolation_truth':[]},source_roots={'mechanical_sources':base,'isolation_truth':base},
        output=str((output/'validation').relative_to(workspace)),operation='validate-runtime',
        settings=dict(study=study,runtime=runtime),seconds=seconds)
    if result['status']!='PASS':raise ValueError('invalid frozen execution: '+result.get('error',result.get('reason','failed')))
    return result['plan']


def _qualification(host,workspace,output,base,study,runtime,args):
    from .generalization_qualification_io import consume_original
    if not args.qualification:raise ValueError('original qualification producer required')
    producer=confined(workspace,args.qualification)
    result=consume_original(host,workspace,producer,output/'qualification',required_suites=['bigint-cpu','bigint-gpu'],
        required_capabilities=['weight-load','finite-update','native-exact'],research=runtime['purpose']=='research',feature_receipt=runtime['feature'])
    current=_document(producer,result['authority']['source_closure'])
    if _document(base,study['source_closure'])!=current:raise ValueError('study differs from qualified source closure')
    if runtime['images']!=result['authority']['images']:raise ValueError('study image pair differs from qualification')
    return result


def _finish(output,study,operation,result,runtime,*,effective=None,original=None):
    from .generalization_lifecycle import inventory
    if effective is not None:write(output/'invocation.json',effective,exclusive=False)
    try:files=inventory(output)
    except ValueError as exc:
        result=dict(status='FAIL',reason='STORAGE_INVENTORY_FAILURE',error=str(exc));files=[]
    write(output/'summary.json',dict(selected_scope=result['status'],scientific_status='UNMEASURED',operation=operation),exclusive=False)
    files=[r for r in files if r['path']!='summary.json']+[reference(output/'summary.json',output)]
    record=seal(dict(version='generalization-command-result/v2',study=study['identity'],stage=operation,
        status=result['status'],runtime=runtime['identity'],result=result,invocation=effective,original=original,inventory=files))
    from .bundle import pack
    staging=output/'private/command-export'
    try:
        staging.mkdir(parents=True)
        write(staging/'summary.json',dict(selected_scope=result['status'],scientific_status='UNMEASURED',scope='command receipt only; not feature qualification'))
        write(staging/'command.json',dict(study=study['identity'],runtime=runtime['identity'],operation=operation,status=result['status'],receipt=record['identity'],original=original))
        archive=pack(staging,mandatory=['summary.json','command.json'])
        archive.rename(output.with_suffix('.zip'));shutil.rmtree(staging)
    except (Exception,KeyboardInterrupt) as exc:
        result=dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',reason='PACKAGE_FAILED',error=str(exc),work_result=result)
        write(output/'summary.json',dict(selected_scope=result['status'],scientific_status='UNMEASURED',operation=operation),exclusive=False)
        files=[r for r in files if r['path']!='summary.json']+[reference(output/'summary.json',output)]
        record=seal(dict(record,status=result['status'],result=result,inventory=files))
    write(output/'result.json',record)
    return EXIT_CODES[result['status']]


def _resume_inputs(workspace,value):
    path=confined(workspace,value);root=path.parent.parent
    pointer=reference(path,root);record=_document(root,pointer)
    if record.get('version')!='generalization-learner-state/v1' or record.get('kind')!='full_state':
        raise ValueError('full-state checkpoint required for resume')
    pins=json.loads(confined(root,'generalization-pins.json').read_text())
    pointers=[pointer,*pins.values()];refs={}
    for name in ('generalization-latest.json','generalization-pins.json'):
        ref=reference(root/name,root);refs[name]=ref
    for ref in pointers:
        refs[ref['path']]=ref
        for child in artifact_refs(_document(root,ref)):
            verify_artifact(root,child);refs[child['path']]=child
    return root,pointer,list(refs.values())


def train(host,workspace,output,base,study,runtime,plan,*,arms,seeds,resume=None,replay=None,stop_after_updates=None):
    if not arms or not seeds or len(set(arms))!=len(arms) or len(set(seeds))!=len(seeds):
        raise ValueError('explicit unique arm/seed subsets required')
    configs=[learner_config(study,runtime,plan['recipe'],arm,seed) for arm in arms for seed in seeds]
    if resume is not None and len(configs)!=1:raise ValueError('resume selects exactly one arm and seed')
    if stop_after_updates is not None and (runtime['purpose']!='acceptance' or type(stop_after_updates) is not int or stop_after_updates<1):
        raise ValueError('bounded acceptance-only interruption required')
    results=[]
    for config in configs:
        name=digest([config['arm'],config['seed']])[7:23];folder=output/name
        aux=folder/'private/inputs';inputs=[];resume_ref=None
        initializer=plan['initializer']['weights'] if plan['initializer'] else None
        if initializer is not None:inputs.append((base,initializer))
        if resume is not None:
            prior,resume_ref,refs=_resume_inputs(workspace,resume);inputs.extend((prior,ref) for ref in refs)
        settings=dict(config=config,resume=resume_ref,initializer=initializer)
        if stop_after_updates is not None:settings['stop_after_updates']=stop_after_updates
        training_refs=list(config['shards'])
        if replay is not None and config['arm']=='verified_replay':
            replay_root,pool_ref,tasks_ref=replay
            inputs.append((replay_root,pool_ref));training_refs.append(tasks_ref)
            settings['replay']=dict(pool=pool_ref,tasks=tasks_ref)
        refs=_copy(aux,inputs)
        result=project_and_invoke(host,runtime['images'][1],workspace,'learner',
            {'training_targets':training_refs,'verified_training_replay':refs},
            source_roots={'training_targets':base,'verified_training_replay':aux},
            output=str((folder/'execution').relative_to(workspace)),operation='train',settings=settings,
            seconds=config['seconds'],gpu=True)
        checkpoints=folder/'execution/role/private/learner/checkpoints'
        if (checkpoints/'retained.json').is_file():
            from .generalization_selection_io import remember_origin
            remember_origin(base,checkpoints,study,config['arm'])
        results.append(dict(arm=config['arm'],seed=config['seed'],result=result,
            root=str((folder/'execution/role/private/learner').relative_to(output))))
        if result['status']!='PASS':return dict(status=result['status'],runs=results)
    return dict(status='PASS',runs=results)


def _execute(args,workspace,host_factory):
    workspace=Path(workspace)
    base,study,runtime=load(workspace,args.manifest,getattr(args,'runtime',None))
    output=confined(workspace,args.output,exists=False)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh command output required')
    if args.image is not None and args.image!=runtime['images'][0]:raise ValueError('global image differs from frozen qualification image')
    if args.operation=='train':
        if not args.arms or not args.seeds or len(set(args.arms))!=len(args.arms) or len(set(args.seeds))!=len(args.seeds):raise ValueError('unique arm/seed subsets required')
        for arm in args.arms:
            for seed in args.seeds:learner_config(study,runtime,{},arm,seed)
    if args.operation=='evaluate':
        from .generalization_workflow import _checkpoints
        view=runtime['evaluation']
        if view is None:raise ValueError('frozen evaluation view required')
        candidates={c['identity'] for c,_ in _checkpoints(base,view['checkpoints'])}
        if not args.checkpoints or len(set(args.checkpoints))!=len(args.checkpoints) or set(args.checkpoints)-candidates:raise ValueError('undeclared checkpoint subset')
        if args.proposal_bank is not None and args.proposal_bank!=digest(view['bank_seeds']):raise ValueError('undeclared proposal bank')
    from .generalization_lifecycle import invocation
    effective=invocation(args,study,runtime)
    def finish(result,original=None):return _finish(output,study,args.operation,result,runtime,effective=effective,original=original)
    host=host_factory(args)
    if args.operation=='qualify':
        from .generalization_qualification_io import produce
        if getattr(args,'reuse_run',None):
            from .generalization_qualification_io import consume_original
            previous=confined(workspace,args.reuse_run)/'qualify'
            output.mkdir(parents=True)
            reused=consume_original(host,workspace,previous,output/'reuse',required_suites=['bigint-cpu','bigint-gpu'],
                required_capabilities=['weight-load','finite-update','native-exact'])
            # Preserve the original producer tree and exact references, never issue
            # a REUSED-only producer as new qualification evidence.
            from .generalization_qualification_io import public_closure
            refs=[reference(previous/'qualification.json',previous),*public_closure(previous,reused['authority'])]
            _copy(output,[(previous,r) for r in refs])
            result=dict(status='PASS',original=reused['original'],reused=True)
        else:result=produce(host,workspace,output,images=runtime['images'],contract=runtime['identity'])
        return finish(result)
    output.mkdir(parents=True)
    if args.operation=='preflight':
        from oeis_learn.generalization.resources import preflight
        resources=preflight(**runtime['storage'],available_bytes=shutil.disk_usage(workspace).free)
        plan=_validate(host,workspace,output,base,study,runtime)
        return finish(dict(status='PASS',resources=resources,plan=plan,qualification_required=True))
    _qualification(host,workspace,output,base,study,runtime,args)
    if getattr(args,'reuse_run',None):
        from .generalization_lifecycle import reuse
        previous=confined(workspace,args.reuse_run)
        if not (previous/'result.json').is_file():previous=confined(previous,args.operation)
        original=reuse(previous,output,effective)
        result=original['result']
        if args.operation=='prepare':
            result=dict(result,**{k:str((output/(workspace/result[k]).relative_to(previous)).relative_to(workspace)) for k in ('manifest','runtime')})
        return finish(result,original=original['identity'])
    if args.operation=='prepare':
        _validate(host,workspace,output,base,study,runtime)
        refs=execution_refs(base,study,runtime,preparation=True)
        authority=_document(base,runtime['preparation'])
        result=project_and_invoke(host,runtime['images'][0],workspace,'preparation',
            {'mechanical_sources':refs,'isolation_truth':[]},source_roots={'mechanical_sources':base,'isolation_truth':base},
            output=str((output/'preparation').relative_to(workspace)),operation='prepare',
            settings=dict(preparation=authority,proposed=study),seconds=min(600,authority['budgets']['seconds']))
        if result['status']=='PASS':
            from .generalization_runtime import publish_runtime
            prepared=output/'preparation/role/private/prepared'
            resolved,sidecar=publish_runtime(base,prepared,study,runtime)
            result=dict(result,manifest=str((prepared/'study.json').relative_to(workspace)),runtime=str((prepared/'runtime.json').relative_to(workspace)))
        return finish(result)
    if runtime['preparation'] is not None:raise ValueError('resolved prepared study/runtime required before model work')
    plan=_validate(host,workspace,output,base,study,runtime)
    if args.operation=='train':result=train(host,workspace,output/'training',base,study,runtime,plan,arms=args.arms,seeds=args.seeds,resume=args.resume)
    elif args.operation=='evaluate':
        from .generalization_workflow import evaluate
        result=evaluate(host,workspace,output/'evaluation',base,study,runtime,plan,checkpoints=args.checkpoints,bank=args.proposal_bank)
    elif args.operation=='replay':
        from .generalization_workflow import replay
        result=replay(host,workspace,output/'replay',base,study,runtime,plan)
    else:raise ValueError('unregistered generalization runtime command')
    return finish(result)


def main(args,workspace,host_factory):
    """Preserve a terminal result after launched-work failures and interruption."""
    output=confined(workspace,args.output,exists=False)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh command output required')
    from .generalization_lifecycle import scope
    base,_,limits=load(workspace,args.manifest,getattr(args,'runtime',None))
    try:
        with scope(base,limits['storage'],extra_roots=(output,)):return _execute(args,workspace,host_factory)
    except (Exception,KeyboardInterrupt) as exc:
        if not output.exists():raise
        from .generalization_lifecycle import invocation
        base,study,runtime=load(workspace,args.manifest,getattr(args,'runtime',None))
        return _finish(output,study,args.operation,dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',reason=type(exc).__name__,error=str(exc)),runtime,effective=invocation(args,study,runtime))
