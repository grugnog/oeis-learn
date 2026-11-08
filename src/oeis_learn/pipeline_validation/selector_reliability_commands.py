"""One qualified training-only validation and controlled scheduling diagnostic."""
import copy
import json
import zipfile
import traceback
from pathlib import Path

from .identity import seal,write,confined,file_hash
from .generalization_preparation import reference
from .selector_validation_commands import probes,read,_invoke,check_ref,source,envelope
from .pilot_clocks import DRAIN,MATCHED
from oeis_learn.generalization.selector_run_diagnostics import summarize

CAP=512<<20
GENERATION_FILES=('selections.json','alternative-selections.json','selector-candidates.json',
    'selector-observations.json','evaluation.json','attempts/portfolio-clocks.json','proposal-bank.json')


def register(commands):
    parser=commands.add_parser('pac-reliability',description=__doc__)
    parser.add_argument('--campaign',default='run-state/generalization/pac-012-bcd-r1-002')
    parser.add_argument('--output',required=True)


def collect(output):
    """Closed metadata inventory; never recurse into model/private/target trees."""
    output=Path(output);paths=[];missing=[]
    wanted=[output/n for n in ('intent.json','benchmark-plan.json','result.json','summary.md','error.txt','traceback.txt')]
    from .generalization_diagnostics import collect as controls,EXPORTED
    for name in ('probes','benchmarks'):
        root=output/name
        if not root.is_dir():continue
        destination=output/'controls'/name
        try:
            index=controls(root,destination)
            wanted.append(destination/'index.json')
            wanted.extend(destination/r['export_path'] for r in index['files'] if r['status'] in EXPORTED)
        except (OSError,ValueError,RuntimeError) as exc:
            path=output/('controls-'+name+'-error.txt')
            path.write_text(type(exc).__name__+': '+str(exc)+'\n');wanted.append(path)
    p=output/'probes'
    wanted.extend(p/n for n in ('plan.json','result.json','summary.md','failure.json','traceback.txt',
        'qualification/qualification.json','qualification/source.json','qualification/pipeline.zip',
        'publication/role/visible/prompts.json'))
    for target in ('qualification','gpu'):
        wanted.extend(p/'build'/target/n for n in ('process.json','cleanup.json','stdout.log','stderr.log'))
    roots=[p/a/'generation/role' for a in ('B','D')]
    for a in ('B','D'):wanted.append(p/a/'scoring/role/scoring.json')
    roots.extend(output/'benchmarks'/a/str(i)/'role' for a in ('B','D') for i in range(4))
    for root in roots:
        wanted.extend([root/'result.json',root/'benchmark-telemetry.json'])
        wanted.extend(root/'generation'/n for n in GENERATION_FILES)
        wanted.extend(root/'telemetry'/n for n in ('capability.json','telemetry.json','events.jsonl','samples.jsonl'))
    control_roots=[*roots,p/'publication/role',*(p/a/'scoring/role' for a in ('B','D'))]
    for root in control_roots:
        wanted.append(root/'result.json')
        wanted.extend(root/'supervision'/n for n in ('process.json','cleanup.json','stdout.log','stderr.log'))
    wanted=list(dict.fromkeys(wanted))
    size=0
    for path in wanted:
        relative=str(path.relative_to(output))
        if path.is_symlink():raise ValueError('symlink diagnostic artifact')
        if not path.is_file():missing.append(relative);continue
        # Check every ancestor, including generated directories.
        confined(output,relative)
        size+=path.stat().st_size
        if size>CAP:raise ValueError('diagnostic metadata exceeds 512 MiB cap')
        paths.append(path)
    manifest=seal(dict(version='pac-selector-diagnostic-inventory/v1',
        scientific_status='DIAGNOSTIC_ONLY',files=[dict(path=str(p.relative_to(output)),
        bytes=p.stat().st_size,sha256=file_hash(p)) for p in paths],missing=missing,
        byte_cap=CAP,omitted='weights, private targets/caches, original corpora; unrecorded hardware counters unavailable'))
    write(output/'inventory.json',manifest)
    with zipfile.ZipFile(output/'diagnostics.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in [output/'inventory.json',*paths]:archive.write(path,str(path.relative_to(output)))
    return output/'diagnostics.zip'


def run(args,workspace,host_factory):
    output=confined(workspace,args.output,exists=False)
    if output.exists():raise ValueError('fresh reliability output required')
    output.mkdir(parents=True);args._selector_output=output
    # Freeze ordering/subset rule before qualification or any probe outcomes.
    write(output/'intent.json',seal(dict(version='pac-selector-reliability-intent/v1',
        source=source(workspace),training_tokens=0,assessment='NOT_EXECUTED',
        subset='first 16 sorted training-probe IDs; no outcome-based selection',
        order=[DRAIN,MATCHED,MATCHED,DRAIN],bank='excluded from scheduling benchmark',
        scientific_status='DIAGNOSTIC_ONLY',freeze_eligible=False)))
    primary_error=None
    try:
        child=copy.copy(args);child.output=str((output/'probes').relative_to(workspace));child.preflight_only=False
        try:probes(child,workspace,host_factory)
        finally:args._selector_output=getattr(child,'_selector_output',output)
        args._selector_output=output
        plan=read(output/'probes/plan.json',sealed=True)
        validation=read(output/'probes/result.json',sealed=True)
        host=host_factory(args)
        if host.engine.name!=plan['engine'] or source(workspace)!=plan['source']:
            raise ValueError('qualified probe source/engine changed before benchmark')
        benchmark_plan=seal(dict(plan,phase='benchmark',
            budgets=dict(plan.get('budgets',{}),generation=envelope(4*16*60+600)),
            benchmark_intent=read(output/'intent.json',sealed=True)['identity']))
        write(output/'benchmark-plan.json',benchmark_plan)
        views=output/'probes/publication/role'
        prompts=reference(views/'visible/prompts.json',workspace)
        recipe=read(check_ref(workspace,plan['inputs']['recipe']))
        runs=[]
        for arm in ('B','D'):
            selected=plan['inputs']['checkpoints'][arm];descriptor=selected['descriptor']
            check_ref(confined(workspace,selected['root']),descriptor['weights'])
            for i,policy in enumerate((DRAIN,MATCHED,MATCHED,DRAIN)):
                target=output/'benchmarks'/arm/str(i)
                value=_invoke(host,workspace,benchmark_plan,target,'generator','selector-benchmark',
                    {'visible_prompts':[prompts],'frozen_weights':[descriptor['weights']]},
                    {'visible_prompts':workspace,'frozen_weights':confined(workspace,selected['root'])},
                    dict(checkpoint=descriptor['weights'],prompts=prompts,recipe=recipe,
                         study=benchmark_plan['identity'],source=descriptor['identity'],device='cuda',
                         bank_seeds=None,benchmark_policy=policy,
                         render_node=getattr(host,'policy',{}).get('render_node')),'generation',gpu=True)
                role_root=target/'role';bound=value['benchmark_artifacts']
                records=read(check_ref(role_root,bound['records']),cap=512<<20)
                clock=read(check_ref(role_root,bound['clock']),sealed=True)
                if clock['identity']!=value['clock']:raise ValueError('benchmark clock identity changed')
                runs.append(dict(arm=arm,order=i,policy=policy,metrics=summarize(records,clock),
                    telemetry=read(check_ref(role_root,bound['telemetry']),sealed=True)))
        result=seal(dict(version='pac-selector-reliability-result/v1',status='PASS',
            scientific_status='DIAGNOSTIC_ONLY',freeze_eligible=False,
            probe_status=validation['scientific_status'],probe_result=validation['identity'],
            benchmark_plan=benchmark_plan['identity'],runs=runs,
            interpretation='paired scheduling counterfactual; behavior hashes may differ; no automatic schedule/selector adoption'))
        write(output/'result.json',result)
        lines=['# Training-only reliability diagnostic','',
            'Probe technical status: '+validation['scientific_status'],
            'Schedule results are diagnostic only; compare attempt behavior and completion alongside time.','',
            '| Arm | Order | Policy | Portfolio seconds | Decode seconds | Cap batches | Behavior |',
            '|---|---:|---|---:|---:|---:|---|']
        for r in runs:
            m=r['metrics'];decode='unavailable' if m['decode_seconds'] is None else f"{m['decode_seconds']:.3f}"
            lines.append(f"| {r['arm']} | {r['order']} | {r['policy']} | {m['portfolio_span_seconds']:.3f} | {decode} | {m['cap_batches']} | {m['attempt_behavior']} |")
        (output/'summary.md').write_text('\n'.join(lines)+'\n')
    except Exception as exc:
        primary_error=exc
        (output/'error.txt').write_text(type(exc).__name__+': '+str(exc)+'\n')
        (output/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        try:
            archive=collect(output)
            print('Comprehensive diagnostic archive: '+str(archive),flush=True)
        except (OSError,ValueError,RuntimeError) as exc:
            (output/'collection-error.txt').write_text(type(exc).__name__+': '+str(exc)+'\n')
            print('Diagnostic collection failed: '+str(exc),flush=True)
            if primary_error is None:raise
    print(json.dumps(dict(status='PASS',scientific_status='DIAGNOSTIC_ONLY',
        probe_status=validation['scientific_status'],freeze_eligible=False,report=str(output/'diagnostics.zip'))))
    return 0


def main(args,workspace,host_factory):
    try:return run(args,Path(workspace),host_factory)
    except (ValueError,KeyError,OSError,RuntimeError,TimeoutError) as exc:
        print(json.dumps(dict(status='FAIL',error=type(exc).__name__,message=str(exc))),flush=True)
        traceback.print_exc()
        return 1
