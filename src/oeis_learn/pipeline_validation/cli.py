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
        raise ValueError('study execution adapter unavailable; no research authority inferred')
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
    if args.reuse:
        # Receipt reader exists, but do not silently accept requests before imports are wired.
        raise ValueError('--reuse execution is not yet available; prior evidence was not imported')
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    binding=dict(source=digest(collect_snapshot(root)),contract=contract['identity'])
    return execute(stages,output,adapter=Adapters(args,root,host_factory,contract),binding=binding,
        cap=maximum,reserve=contract['budgets']['archive_seconds']+contract['budgets']['cleanup_seconds'],
        contract=contract,unavailable=unavailable)
