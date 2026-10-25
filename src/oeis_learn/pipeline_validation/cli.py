"""Public stdlib routing validates authority before touching an engine."""
import json
import shutil
from dataclasses import asdict
from .contracts import parse
from .identity import confined, digest, number, reference
from .registry import registry, replay_registry, replay_comparison_registry, replay_batching_registry, hardware_registry, native_capacity_registry, coschedule_registry, resolve
from .runner import execute
from .adapters import Adapters


def main(args, root, host_factory):
    if args.phase=='6':
        return study_main(args,root,host_factory)
    if args.study_manifest or args.final_plan:
        raise ValueError('engineering invocation cannot use study/final authority')
    path=confined(root,args.manifest or 'configs/foundation/pipeline_validation.json')
    contract=parse(json.loads(path.read_text()))
    maximum=contract['budgets']['global_seconds']
    if args.max_seconds is not None:
        if not 0<number(args.max_seconds)<=maximum:
            raise ValueError('--max-seconds may only lower the frozen ceiling')
        maximum=args.max_seconds
    primary=args.engine
    if primary=='auto':
        primary=next((n for n in ('docker','podman') if shutil.which(n)), 'docker')
    args.engine=primary
    engines=[primary]
    mode=args.engines or contract['engines']
    other='podman' if primary=='docker' else 'docker'
    if mode=='both' or mode=='available' and shutil.which(other):
        engines.append(other)
    args.pipeline_engines=engines
    if (contract['scope'].startswith('engineering-replay') or contract['scope']=='engineering-hardware') and args.phase not in {'all-ready','1','2'}:
        raise ValueError('replay scope registers only phases 1 and 2')
    if contract['scope']=='engineering-native-capacity' and args.phase not in {'all-ready','1','3'}:
        raise ValueError('native capacity scope registers only phases 1 and 3')
    if contract['scope']=='engineering-coschedule' and args.phase not in {'all-ready','1','2','4'}:
        raise ValueError('co-scheduling scope registers only phases 1,2 and4')
    select_registry={'engineering-coschedule':coschedule_registry,'engineering-native-capacity':native_capacity_registry,'engineering':registry,'engineering-replay':replay_registry,
        'engineering-replay-comparison':replay_comparison_registry,'engineering-replay-batching':replay_batching_registry,'engineering-hardware':hardware_registry}
    all_stages=select_registry[contract['scope']](engines,primary)
    stages=resolve(all_stages,args.phase)
    unavailable=[asdict(s) for s in all_stages if not s.available]
    for value in [args.run,args.inputs,*args.reuse]:
        if value:
            confined(root,value)
    for value in (args.run,args.inputs):
        if value and not confined(root,value).is_dir():raise ValueError('--run/--inputs must identify directories')
    for ref in [*contract['input_refs'],*([contract['weights_ref']] if contract['weights_ref'] else [])]:
        path=reference(root,ref)
        if set(path.relative_to(root).parts)&{'private','final'}:raise ValueError('engineering manifest cannot reference private/final payloads')
        bases=[confined(root,v) for v in (args.run,args.inputs) if v]
        if not any(base in path.parents for base in bases):raise ValueError('manifest input identity differs from selected run/inputs')
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    snapshot=collect_snapshot(root)
    if contract['source'] and contract['source']['sha256']!=digest(snapshot):raise ValueError('manifest source differs from current runtime snapshot')
    output=confined(root,args.output,exists=False)
    if args.scope_report:
        print(json.dumps(dict(selected=[asdict(s) for s in stages],unavailable=unavailable,
            ceiling_seconds=maximum,contract=contract['identity'],engines=engines,research='NOT_SELECTED',scope=contract['scope']),indent=2))
        return 0
    from .reuse import validate_import
    reuse_roots=[confined(root,path) for path in args.reuse]
    reuse_roots.extend(reference(root,ref).parent for ref in contract['reuse_refs'])
    if len(reuse_roots)>16:raise ValueError('at most sixteen prior invocations may be referenced')
    for producer in reuse_roots:
        validate_import(producer)
    from .closures import dependency_closure
    binding=dict(source=digest(snapshot),contract=contract['identity'],
        dependency_closure=dependency_closure(snapshot,prerequisites={}))
    return execute(stages,output,adapter=Adapters(args,root,host_factory,contract),binding=binding,
        cap=maximum,reserve=contract['budgets']['archive_seconds']+contract['budgets']['cleanup_seconds'],
        contract=contract,unavailable=unavailable,reuse_roots=reuse_roots)



def study_main(args,root,host_factory,*,collect=False):
    from pathlib import Path
    from .studies import parse_study
    from .identity import reference,write
    from .registry import Stage
    from .resources import observe,resolve
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    if args.manifest or not args.study_manifest:
        raise ValueError('phase6 requires a separate study manifest; engineering mixing rejected')
    source=digest(collect_snapshot(root))
    contract=json.loads(confined(root,args.study_manifest).read_text())
    final_plan=confined(root,args.final_plan) if args.final_plan else None
    if contract.get('version')=='pipeline-study-batch/v1':
        if collect:raise ValueError('nested study batches are not allowed')
        from .study_batch import main as batch_main
        return batch_main(args,root,host_factory,contract,source,final_plan)
    decision=parse_study(contract,root=root,source=source,final_plan=final_plan)
    if args.scope_report:
        print(json.dumps(decision,indent=2));return 0
    output=confined(root,args.output,exists=False)
    if decision['status']!='ADMITTED':
        stages=[Stage('study.admission',6,available=False,reason='missing gates: '+', '.join(decision['missing']))]
        if collect:return dict(stages=stages,adapter=lambda *_:None,ledger=None,decision=decision)
        return execute(stages,output,adapter=lambda *_: None,binding={'source':source,'study':contract['identity']},
            cap=contract['budgets']['wall_seconds'],reserve=contract['budgets']['cleanup_seconds']+contract['budgets']['archive_seconds'],contract=contract)
    # References are only resolved after every authority gate has passed.
    for arm in contract['arms']:
        for field in ('inputs','config','run','checkpoint'):
            if arm[field] is not None:
                reference(root,arm[field])
    runtime=json.loads(reference(root,contract['gates']['runtime']).read_text())
    if args.engine not in ('auto',runtime['engine']):
        raise ValueError('study engine differs from qualified runtime')
    if args.image and args.image != runtime['image']:
        raise ValueError('study image differs from qualified runtime')
    host_options=__import__('copy').copy(args)
    host_options.engine=runtime['engine'];host_options.image=runtime['image']
    from oeis_learn.container_runtime import Engine
    from .resources import execution_identity
    try:
        observations=observe()
        observations['limits'].append(Engine(runtime['engine']).facts()['memory_bytes'])
        observations['cpus']=min(observations['cpus'],args.cpus or 16)
        policy=resolve(**observations,lower=int(args.memory_gib*(1<<30)) if args.memory_gib else None)
        if execution_identity(policy)!=runtime['resources']:
            raise ValueError('study execution resources differ from the reviewed qualification envelope')
        host_options.memory_gib=policy['ceiling_bytes']/(1<<30)
        host_options.cpus=min(args.cpus or 16,policy['cpus'])
        host=host_factory(host_options)
    except (RuntimeError,OSError,ValueError) as exc:
        stages=[Stage('study.runtime',6,available=False,reason=str(exc))]
        if collect:return dict(stages=stages,adapter=lambda *_:None,ledger=None,decision=decision)
        return execute(stages,output,adapter=lambda *_:None,binding={'source':source,'study':contract['identity']},
            cap=contract['budgets']['wall_seconds'],reserve=contract['budgets']['cleanup_seconds']+contract['budgets']['archive_seconds'],contract=contract)
    from .studies import StudyLedger
    study_ledger=StudyLedger(root/'run-state/pipeline-studies',contract)
    stages=[Stage(arm['id'],6,seconds=arm.get('attempt',arm)['wall_seconds'],adapter='study') for arm in contract['arms']]
    def run_arm(stage,destination):
        from argparse import Namespace
        from foundation_bigint import execute as execute_bigint,resolve_recipe
        arm=next(a for a in contract['arms'] if a['id']==stage.id)
        reservation=study_ledger.reserve(arm)
        run=reference(root,arm['run']).parent if arm['run'] else destination/'run'
        config=reference(root,arm['config']) if arm['config'] else None
        inputs=reference(root,arm['inputs']).parent if arm['inputs'] else None
        if arm['checkpoint']:
            checkpoint=reference(root,arm['checkpoint'])
            pointer=json.loads((run/'checkpoint-identity.json').read_text())['selected']
            if confined(run/'checkpoints',pointer['path'])!=checkpoint:
                raise ValueError('selected checkpoint differs from admitted arm')
        options=Namespace(**vars(host_options))
        options.command=arm['operation'];options.run=str(run);options.inputs=str(inputs) if inputs else None
        options.config=str(config) if config else None;options.seed=arm['seed'];options.output=str(destination/'analysis') if arm['operation'] in {'diagnose','evaluate'} else None
        options.encoder=None;options.view='diverse';options.ablate_inputs='all' if contract['kind']=='encoder' else None
        options.split='final' if contract['kind']=='final' else 'development';options.final_plan=str(final_plan) if final_plan else None
        options.fitting='paired';options.explicit_options=set()
        options.max_updates=None;options.budget_seconds=None
        options.explicit_options=[]
        required_work=None
        if config:
            recipe=resolve_recipe(host,options,destination/'resolve')
            if contract['kind']=='representation' and (recipe['learning']['target_view']!=arm['view'] or recipe['learning']['curriculum'] or recipe['learning']['replay']):
                raise ValueError('representation arm changed its frozen target/control policy')
            if recipe['model']['seed']!=arm['seed'] or recipe['budgets']['training_seconds']+recipe['budgets'].get('preparation_seconds',0)+300>reservation['wall_seconds']:
                raise ValueError('recipe seed/budget differs from admitted arm')
        elif run.exists() and (run/'run.json').is_file():
            recipe=json.loads((run/'run.json').read_text())['recipe']
        else:
            raise ValueError('arm needs a reviewed recipe or a hash-bound completed run')
        from .studies import validate_recipe
        effective_arm=dict(arm,work=reservation['work'])
        denominator=json.loads(final_plan.read_text())['denominator'] if contract['kind']=='final' else None
        required_work=validate_recipe(contract,effective_arm,recipe,final_denominator=denominator)
        write(destination/'admission.json',dict(study=contract['identity'],arm=arm['id'],reservation=reservation,required_work=required_work))
        from .supervision import launch
        import sys
        descriptor=destination/'private/arm.json'
        write(descriptor,vars(options))
        try:
            from .disk import DiskWatchdog
            from foundation_local import state_path
            with DiskWatchdog(destination,destination/'resource',cap=reservation['disk_bytes'],extra_roots=[state_path(str(run))]):
                observations=launch([sys.executable,str(root/'scripts/pipeline_study_arm.py'),str(descriptor)],
                    destination/'supervision',seconds=reservation['wall_seconds']-5,cleanup_seconds=5)
        except BaseException:
            study_ledger.finish(arm['id'],'INTERRUPTED',attempt=reservation['attempt'])
            raise
        study_ledger.finish(arm['id'],'COMPLETE' if contract['kind']=='final' else 'INCONCLUSIVE',attempt=reservation['attempt'])
        return dict(study=contract['identity'],arm=arm['id'],observations=observations,
            result='execution ended; scientific outcome assembled separately',stage_state='PASS')
    def finalize(report_root,invocation,receipts,*,prefix=''):
        from .study_reports import study_report
        return study_report(contract,invocation,report_root,receipts,workspace=root,prefix=prefix,debits=study_ledger.reservations())
    run_arm.finalize=finalize
    if collect:return dict(stages=stages,adapter=run_arm,ledger=study_ledger,decision=decision)
    maximum=contract['budgets']['wall_seconds']
    if args.max_seconds is not None:
        if not 0<number(args.max_seconds)<=maximum:raise ValueError('study ceiling can only be lowered')
        maximum=args.max_seconds
    from oeis_learn.fresh.common import lock
    import uuid
    with lock(study_ledger.root/'controller.lock'):
        if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh study output required')
        study_ledger.begin_invocation(uuid.uuid4().hex)
        return execute(stages,output,adapter=run_arm,binding={'source':source,'study':contract['identity']},cap=maximum,
            reserve=contract['budgets']['cleanup_seconds']+contract['budgets']['archive_seconds'],contract=contract)
