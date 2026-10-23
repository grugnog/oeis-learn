"""Public stdlib routing validates authority before touching an engine."""
import json
import shutil
from dataclasses import asdict
from .contracts import parse
from .identity import confined, digest, number
from .registry import registry, resolve
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
    all_stages=registry(engines,primary)
    stages=resolve(all_stages,args.phase)
    unavailable=[asdict(s) for s in all_stages if not s.available]
    for value in [args.run,args.inputs,*args.reuse]:
        if value:
            confined(root,value)
    output=confined(root,args.output,exists=False)
    if args.scope_report:
        print(json.dumps(dict(selected=[asdict(s) for s in stages],unavailable=unavailable,
            ceiling_seconds=maximum,contract=contract['identity'],engines=engines,research='NOT_SELECTED'),indent=2))
        return 0
    from .reuse import validate_import
    reuse_roots=[confined(root,path) for path in args.reuse]
    for producer in reuse_roots:
        validate_import(producer)
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    snapshot=collect_snapshot(root)
    from .closures import dependency_closure
    binding=dict(source=digest(snapshot),contract=contract['identity'],
        dependency_closure=dependency_closure(snapshot,prerequisites={}))
    return execute(stages,output,adapter=Adapters(args,root,host_factory,contract),binding=binding,
        cap=maximum,reserve=contract['budgets']['archive_seconds']+contract['budgets']['cleanup_seconds'],
        contract=contract,unavailable=unavailable,reuse_roots=reuse_roots)



def study_main(args,root,host_factory):
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
    decision=parse_study(contract,root=root,source=source,final_plan=final_plan)
    if args.scope_report:
        print(json.dumps(decision,indent=2));return 0
    output=confined(root,args.output,exists=False)
    if decision['status']!='ADMITTED':
        stages=[Stage('study.admission',6,available=False,reason='missing gates: '+', '.join(decision['missing']))]
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
    policy=resolve(**observe(),lower=int(args.memory_gib*(1<<30)) if args.memory_gib else None)
    host_options.memory_gib=policy['ceiling_bytes']/(1<<30)
    host_options.cpus=min(args.cpus or 16,policy['cpus'])
    host=host_factory(host_options)
    from .studies import StudyLedger
    study_ledger=StudyLedger(root/'run-state/pipeline-studies',contract)
    stages=[Stage(arm['id'],6,seconds=arm['wall_seconds'],adapter='study') for arm in contract['arms']]
    def run_arm(stage,destination):
        from argparse import Namespace
        from foundation_bigint import execute as execute_bigint,resolve_recipe
        arm=next(a for a in contract['arms'] if a['id']==stage.id)
        study_ledger.reserve(arm)
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
        options.config=str(config) if config else None;options.seed=arm['seed'];options.output=str(destination/'analysis')
        options.encoder=None;options.view='diverse';options.ablate_inputs='all' if contract['kind']=='encoder' else None
        options.split='final' if contract['kind']=='final' else 'development';options.final_plan=str(final_plan) if final_plan else None
        options.fitting='paired';options.explicit_options=set()
        options.max_updates=None;options.budget_seconds=None
        options.explicit_options=[]
        if config:
            recipe=resolve_recipe(host,options,destination/'resolve')
            if recipe['model']['seed']!=arm['seed'] or recipe['budgets']['training_seconds']+recipe['budgets'].get('preparation_seconds',0)+300>arm['wall_seconds']:
                raise ValueError('recipe seed/budget differs from admitted arm')
        write(destination/'admission.json',dict(study=contract['identity'],arm=arm['id'],work_debit=arm['work'],disk_reservation=arm['disk_bytes']))
        from .supervision import launch
        import sys
        descriptor=destination/'private/arm.json'
        write(descriptor,vars(options))
        try:
            observations=launch([sys.executable,str(root/'scripts/pipeline_study_arm.py'),str(descriptor)],
                destination/'supervision',seconds=arm['wall_seconds']-5,cleanup_seconds=5)
        except BaseException:
            study_ledger.finish(arm['id'],'INTERRUPTED')
            raise
        study_ledger.finish(arm['id'],'INCONCLUSIVE')
        return dict(study=contract['identity'],arm=arm['id'],observations=observations,
            result='execution ended; scientific outcome requires reviewed report',stage_state='INCONCLUSIVE')
    maximum=contract['budgets']['wall_seconds']
    if args.max_seconds is not None:
        if not 0<number(args.max_seconds)<=maximum:raise ValueError('study ceiling can only be lowered')
        maximum=args.max_seconds
    return execute(stages,output,adapter=run_arm,binding={'source':source,'study':contract['identity']},cap=maximum,
        reserve=contract['budgets']['cleanup_seconds']+contract['budgets']['archive_seconds'],contract=contract)
