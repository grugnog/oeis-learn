#!/usr/bin/env python3
"""Temporary same-checkpoint continuation matrix; no research/qualification claim."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents)>3 else None
BRANCHES=('continue-a','continue-b','split-first','split-rest')


def plan(args):
    stages=[]
    for name in ('compatibility',*BRANCHES,'compare'):
        target=Path(args.output)/name
        stages.append(dict(name=name,operation=name,target=str(target),
            seconds=600 if name=='compatibility' else 300 if name=='compare' else 780,
            needs=[] if name=='compatibility' else ['compatibility']+(['split-first'] if name=='split-rest' else []),
            argv=[sys.executable,str(Path(__file__).resolve()),'--engine',args.engine,
                  '--run',args.run,'--output',str(target),'--internal',name]))
    return stages


def branch(name,context,previous):
    return (previous('split-first') if name=='split-rest' else context['paused'],
            context['paused_updates']+1 if name=='split-first' else None)


def compare_states(states):
    from inspect_resume import differences
    from itertools import combinations
    result={}
    for a,b in combinations(states,2):
        left,right=states[a],states[b]
        components={k:differences(left['states'][k],right['states'][k])
                    for k in ('weights','optimizer','rng','data','replay')}
        components['ledger']=differences(left['ledger'].snapshot(),right['ledger'].snapshot())
        result[a+':'+b]=dict(equal=all(v['equal'] for v in components.values()),components=components)
    return result


def load_context(run):
    from inspect_resume import saved_runs
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.generalization_runtime import load
    run=confined(ROOT,run);saved=saved_runs(run)
    meta=json.loads((run/'diagnostic.json').read_text())['metadata']
    original=confined(ROOT,meta['previous'])
    base,study,runtime=load(ROOT,str(run/'replay-resume/private/resolved/study.json'))
    if runtime['purpose']!='acceptance' or runtime['images']!=meta['images']:
        raise ValueError('unchanged acceptance image/input authority required')
    seed=study['seeds']
    if len(seed)!=1:raise ValueError('single saved seed required')
    paused=json.loads((saved['paused']/'training.json').read_text())
    full=json.loads((saved['full']/'training.json').read_text())
    if full['updates']<paused['updates']+2:raise ValueError('at least two remaining updates required')
    pool=confined(run,'replay-resume/case/private/replay/'+str(seed[0])+'/search/private')
    return dict(run=run,original=original,base=base,study=study,runtime=runtime,seed=seed[0],
                paused=saved['paused'],saved=saved,pool=pool,
                paused_updates=paused['updates'],final_updates=full['updates'])


def learner_output(stage):
    from oeis_learn.pipeline_validation.identity import confined
    row=json.loads((stage/'result.json').read_text())
    if row['status']!='PASS':raise ValueError('successful producer required')
    return confined(ROOT,row['learner'])


def train_branch(name,context,host,output):
    from oeis_learn.pipeline_validation.generalization_commands import _validate,train
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.identity import confined
    base,study,runtime=context['base'],context['study'],context['runtime']
    start,stop=branch(name,context,lambda n:learner_output(output.parent/n))
    prior=json.loads((start/'training.json').read_text())
    pointer=confined(start/'checkpoints',prior['checkpoint']['path'])
    case=output/'replay-resume'
    execution=_validate(host,ROOT,case,base,study,runtime,seconds=60)
    host.select_gpu()
    destination=case/'case/private/training'
    result=train(host,ROOT,destination,base,study,runtime,execution,
        arms=['verified_replay'],seeds=[context['seed']],resume=str(pointer.relative_to(ROOT)),
        replay=(context['pool'],reference(context['pool']/'pool.json',context['pool']),runtime['replay']['tasks']),
        stop_after_updates=stop)
    if len(result['runs'])!=1:raise ValueError('one learner result required')
    row=result['runs'][0];actual=row['result']
    expected_status='PARTIAL' if stop is not None else 'PASS'
    if actual['status']!=expected_status or actual['updates']!=(stop or context['final_updates']):
        raise ValueError('expected continuation boundary not reached')
    if actual['compatibility']!=prior['compatibility'] or actual['device']!='cuda':
        raise ValueError('continuation changed contract or device')
    return dict(status='PASS',learner=str((destination/row['root']).relative_to(ROOT)),
                start_checkpoint=prior['checkpoint']['sha256'],checkpoint=actual['checkpoint']['sha256'],
                updates=actual['updates'],learner_status=actual['status'],feature_qualified=False)


def comparison_container():
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from diagnose_hip import save
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.contracts import verify_artifact
    states={};refs={};missing=[]
    for name in ('a','b','split','original','old-resumed'):
        root=Path('/saved')/name
        if not root.exists():missing.append(name);continue
        result=json.loads((root/'training.json').read_text())
        descriptor=json.loads(verify_artifact(root/'checkpoints',result['checkpoint'],max_bytes=1<<20))
        store=GeneralizationCheckpointStore(root/'checkpoints',result['compatibility'],descriptor['ledger']['cap'])
        states[name]=store.load(result['checkpoint'])
        refs[name]=dict(checkpoint=result['checkpoint']['sha256'],compatibility=result['compatibility'])
    if len({r['compatibility'] for r in refs.values()})!=1:raise ValueError('incompatible comparison contracts')
    comparisons=compare_states(states)
    complete=all(n in states for n in ('a','b','split'))
    repeat_equal=comparisons.get('a:b',{}).get('equal')
    split_equal=comparisons.get('a:split',{}).get('equal')
    report=dict(version='generalization-resume-matrix/v1',status='PASS',feature_qualified=False,
        scientific_status='UNMEASURED',complete=complete,missing=missing,checkpoints=refs,
        repeat_equal=repeat_equal,split_equal=split_equal,comparisons=comparisons,
        observation='incomplete_matrix' if not complete else 'repeat_divergence' if not repeat_equal else
                    'extra_restart_divergence' if not split_equal else 'new_continuations_agree',
        note='Exact measurements only. PASS means comparison completed, not feature qualification or a diagnosed backend cause.')
    save(Path('/output/comparison.json'),report)


def compare(host,context,output):
    import shutil,uuid
    from inspect_resume import command
    from oeis_learn.pipeline_validation.supervision import launch,reclaim
    inputs={'original':context['saved']['full'],'old-resumed':context['saved']['resumed']}
    for key,name in [('a','continue-a'),('b','continue-b'),('split','split-rest')]:
        stage=output.parent/name
        if (stage/'result.json').is_file() and json.loads((stage/'result.json').read_text()).get('status')=='PASS':
            inputs[key]=learner_output(stage)
    helpers=output/'private/helper';helpers.mkdir(parents=True)
    for filename in ('probe_resume.py','inspect_resume.py','diagnose_hip.py'):
        shutil.copyfile(Path(__file__).with_name(filename),helpers/filename)
    target=output/'private/comparison';target.mkdir()
    container='resume-matrix-'+uuid.uuid4().hex
    argv=command(host,context['runtime']['images'][1],helpers,target,inputs,container)
    # Reuse the reviewed CPU command constructor; change only the helper entrypoint.
    argv[argv.index('/helper/inspect_resume.py')]='/helper/probe_resume.py'
    cleanup=dict(deadline=time.monotonic()+270,allowance_seconds=30)
    try:
        launch(argv,output/'supervision',seconds=240,stream_cap=1<<20,cleanup_seconds=30,cleanup_state=cleanup)
    finally:reclaim(host.engine,container,output/'supervision',cleanup)
    report=json.loads((target/'comparison.json').read_text())
    from diagnose_hip import save
    save(output/'comparison.json',report)
    return dict(status='PASS',complete=report['complete'],observation=report['observation'],feature_qualified=False)


def internal(args):
    from diagnose_hip import host_for
    from oeis_learn.pipeline_validation.identity import confined
    context=load_context(args.run)
    output=confined(ROOT,args.output,exists=False);output.mkdir(parents=True,exist_ok=False)
    host=host_for(args.engine)
    if args.internal=='compatibility':
        from oeis_learn.pipeline_validation.generalization_commands import _qualification
        from types import SimpleNamespace
        q=_qualification(host,ROOT,output,context['base'],context['study'],context['runtime'],
                         SimpleNamespace(qualification=str(context['original']/'qualify')))
        return dict(status='PASS',original=q['original'],paused_updates=context['paused_updates'],
                    final_updates=context['final_updates'],feature_qualified=False)
    if args.internal=='compare':return compare(host,context,output)
    return train_branch(args.internal,context,host,output)


def export(output):
    from diagnose_hip import save
    from oeis_learn.pipeline_validation.generalization_diagnostics import collect,control_fields,terminal_fields
    from oeis_learn.pipeline_validation.bundle import pack,verify
    from shutil import copyfile
    destination=output/'export';destination.mkdir()
    record=json.loads((output/'diagnostic.json').read_text())
    save(destination/'diagnostic.json',record)
    save(destination/'summary.json',dict(selected_scope=record['status'],feature_qualified=False,
        scientific_status='UNMEASURED',scope='same-checkpoint continuation diagnostics; comparison completion is not equality'))
    comparison=output/'compare/comparison.json'
    if comparison.is_file():copyfile(comparison,destination/'comparison.json')
    for stage in record['plan']:
        name=stage['name'];root=output/name
        if root.is_dir():
            collect(root,destination/(name+'-diagnostics'))
            result=root/'result.json'
            if result.is_file():save(destination/'controls'/name/'result.json',terminal_fields(json.loads(result.read_text())))
        for control_root,label in [(output/'private/logs'/name,'outer'),(root/'supervision','comparison')]:
            for filename in ('process.json','cleanup.json'):
                path=control_root/filename
                if path.is_file():save(destination/'controls'/name/label/filename,control_fields(json.loads(path.read_text())))
            for filename in ('stdout.log','stderr.log'):
                path=control_root/filename
                if path.is_file():
                    target=destination/'logs'/name/label/filename;target.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('rb') as stream:stream.seek(max(0,path.stat().st_size-65536));target.write_bytes(stream.read(65536))
    mandatory=['summary.json','diagnostic.json']+(['comparison.json'] if comparison.is_file() else [])
    archive=pack(destination,mandatory=mandatory,cap=64<<20,optional_cap=48<<20)
    target=output.with_suffix('.zip')
    if target.exists():raise ValueError('matrix archive already exists')
    archive.rename(target);verify(target)
    return target


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=['docker','podman'],required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--internal',choices=['compatibility',*BRANCHES,'compare'],help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
    from diagnose_hip import save,execute,run
    from oeis_learn.pipeline_validation.identity import confined,file_hash
    if args.internal:
        output=confined(ROOT,args.output,exists=False)
        try:result=internal(args)
        except Exception as exc:result=dict(status='FAIL',reason=type(exc).__name__,error=str(exc)[:2048])
        save(output/'result.json',result)
        return 0 if result['status']=='PASS' else 1
    context=load_context(args.run)
    output=confined(ROOT,args.output,exists=False)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh matrix output and archive required')
    stages=plan(args)
    if args.dry_run:
        print(json.dumps(dict(stages=stages,ceiling_seconds=sum(s['seconds'] for s in stages),feature_qualified=False),indent=2));return 0
    output.mkdir(parents=True)
    from oeis_learn.generalization.resources import CAPS
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope
    metadata=dict(source_run=args.run,images=context['runtime']['images'],helpers={
        name:file_hash(Path(__file__).with_name(name)) for name in
        ('probe_resume.py','test_probe_resume.py','inspect_resume.py','diagnose_hip.py')})
    with scope(output,CAPS):record=execute(stages,output,runner=run,metadata=metadata)
    archive=export(output)
    print(json.dumps(dict(status=record['status'],archive=str(archive.relative_to(ROOT)),feature_qualified=False),indent=2))
    return 0 if record['status']=='PASS' else 1


if __name__=='__main__':
    if sys.argv[1:]==['--container']:comparison_container()
    else:
        try:raise SystemExit(main())
        except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
