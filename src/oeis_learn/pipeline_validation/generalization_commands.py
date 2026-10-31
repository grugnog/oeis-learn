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


def _validate(host,workspace,output,base,study,runtime):
    refs=execution_refs(base,study,runtime,preparation=runtime['preparation'] is not None)
    result=project_and_invoke(host,runtime['images'][0],workspace,'preparation',
        {'mechanical_sources':refs,'isolation_truth':[]},source_roots={'mechanical_sources':base,'isolation_truth':base},
        output=str((output/'validation').relative_to(workspace)),operation='validate-runtime',
        settings=dict(study=study,runtime=runtime),seconds=300)
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


def _finish(output,study,operation,result,runtime):
    record=seal(dict(version='generalization-command-result/v1',study=study['identity'],stage=operation,
        status=result['status'],runtime=runtime['identity'],result=result))
    write(output/'result.json',record)
    write(output/'summary.json',dict(selected_scope=result['status'],scientific_status='UNMEASURED',operation=operation))
    return EXIT_CODES[result['status']]


def _resume_inputs(workspace,value):
    path=confined(workspace,value);root=path.parent.parent
    pointer=reference(path,root);record=_document(root,pointer)
    if record.get('version')!='generalization-learner-state/v1' or record.get('kind')!='full_state':
        raise ValueError('full-state checkpoint required for resume')
    pins=json.loads(confined(root,'generalization-pins.json').read_text())
    pointers=[pointer,*pins.values()];refs={}
    for ref in pointers:
        refs[ref['path']]=ref
        for child in artifact_refs(_document(root,ref)):
            verify_artifact(root,child);refs[child['path']]=child
    return root,pointer,list(refs.values())


def train(host,workspace,output,base,study,runtime,plan,*,arms,seeds,resume=None,replay=None):
    if not arms or not seeds or len(set(arms))!=len(arms) or len(set(seeds))!=len(seeds):
        raise ValueError('explicit unique arm/seed subsets required')
    configs=[learner_config(study,runtime,plan['recipe'],arm,seed) for arm in arms for seed in seeds]
    if resume is not None and len(configs)!=1:raise ValueError('resume selects exactly one arm and seed')
    results=[]
    for config in configs:
        name=digest([config['arm'],config['seed']])[7:23];folder=output/name
        aux=folder/'private/inputs';inputs=[];resume_ref=None
        initializer=plan['initializer']['weights'] if plan['initializer'] else None
        if initializer is not None:inputs.append((base,initializer))
        if resume is not None:
            prior,resume_ref,refs=_resume_inputs(workspace,resume);inputs.extend((prior,ref) for ref in refs)
        settings=dict(config=config,resume=resume_ref,initializer=initializer)
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
        results.append(dict(arm=config['arm'],seed=config['seed'],result=result,
            root=str((folder/'execution/role/private/learner').relative_to(output))))
        if result['status']!='PASS':return dict(status=result['status'],runs=results)
    return dict(status='PASS',runs=results)


def main(args,workspace,host_factory):
    workspace=Path(workspace)
    base,study,runtime=load(workspace,args.manifest,getattr(args,'runtime',None))
    output=confined(workspace,args.output,exists=False)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh command output required')
    if args.image is not None and args.image!=runtime['images'][0]:raise ValueError('global image differs from frozen qualification image')
    if args.operation=='train':
        for arm in args.arms:
            for seed in args.seeds:learner_config(study,runtime,{},arm,seed)
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
        return _finish(output,study,args.operation,result,runtime)
    output.mkdir(parents=True)
    if args.operation=='preflight':
        from oeis_learn.generalization.resources import preflight
        resources=preflight(**runtime['storage'],available_bytes=shutil.disk_usage(workspace).free)
        plan=_validate(host,workspace,output,base,study,runtime)
        return _finish(output,study,'preflight',dict(status='PASS',resources=resources,plan=plan,qualification_required=True),runtime)
    _qualification(host,workspace,output,base,study,runtime,args)
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
        return _finish(output,study,'prepare',result,runtime)
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
    return _finish(output,study,args.operation,result,runtime)
