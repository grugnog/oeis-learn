#!/usr/bin/env python3
"""Temporary fresh-process replay of trusted captured Linear operands; no training."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents)>3 else None
POLICIES=('observe','rocblas','no-lt')


def schedule():return [(policy,trial) for trial in range(1,4) for policy in POLICIES]


def summarize(rows):
    groups={}
    for policy in POLICIES:
        actual=[r['result'] for r in rows if r['policy']==policy and r['status']=='PASS']
        variants={}
        for name in ('linear','flat_linear','mm_bias','linear_no_bias'):
            observed=[r['variants'][name] for r in actual if name in r['variants']]
            if observed:
                variants[name]=dict(processes=len(observed),cross_process_equal=len(observed)==3 and
                    len({v['output_sha256'] for v in observed})==1,
                    within_process_equal=all(v['repeat_equal'] for v in observed))
        groups[policy]=variants
    return dict(complete=len(rows)==9 and all(r['status']=='PASS' for r in rows),policies=groups,
        feature_qualified=False,scientific_status='UNMEASURED',
        note='Only captured operands are replayed. A clean result is not proof of full training reproducibility.')


def saved_operand(run):
    from oeis_learn.pipeline_validation.identity import confined,file_hash
    metadata=json.loads((run/'diagnostic.json').read_text())['metadata']
    stage=run/'old-traced';result=json.loads((stage/'result.json').read_text())
    if result['status']!='PASS':raise ValueError('successful operand producer required')
    learner=confined(ROOT,result['learner'])
    if not learner.is_relative_to(stage):raise ValueError('learner escaped producer stage')
    expected=next(r['linear_operand_sha256'] for r in result['trace'] if r.get('phase')=='forward' and 'linear_operand_sha256' in r)
    operand=confined(stage,str((learner.parent.parent/'linear-operands.pt').relative_to(stage)))
    if operand.stat().st_size>64<<20 or file_hash(operand)!=expected:raise ValueError('captured operand hash/size mismatch')
    return operand,expected,metadata['images'][1]


def command(host,image,helper,inputs,output,name,policy,expected):
    from oeis_learn.container_runtime import private_runtime_environment
    environment=private_runtime_environment(output,destination='/output')
    environment.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HIPBLASLT_LOG_MASK='32')
    return host.engine.command(image=image,name=name,
        mounts=[(helper,'/helper',False),(inputs,'/inputs',False),(output,'/output',True)],
        argv=['python','/helper/replay_linear.py','--container',policy,expected],
        gpu=True,render_node=host.policy.get('render_node'),memory_bytes=host.policy['memory_bytes'],
        cpus=host.policy['cpus'],network=False,tmp_bytes=2<<30,environment=environment)


def worker(policy,expected):
    import os,hashlib
    if policy not in POLICIES:raise ValueError('closed replay policy required')
    # Configuration precedes any Torch import/BLAS handle creation.
    from trace_resume import configure_policy,probe_linear
    if policy!='observe':configure_policy(policy)
    import torch
    from oeis_learn.pipeline_validation.sizing_execution import execution_context,execution_observation
    from diagnose_hip import save
    path=Path('/inputs/linear-operands.pt')
    if path.stat().st_size>64<<20 or 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('operand verification failed before deserialization')
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('actual HIP required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    values=torch.load(path,map_location='cpu',weights_only=True)
    with execution_context(),torch.no_grad():
        result=probe_linear({k:v.to('cuda') for k,v in values.items()})
        result.update(torch=str(torch.__version__),hip=torch.version.hip,
            device=torch.cuda.get_device_name(),architecture=torch.cuda.get_device_properties(0).gcnArchName,
            execution=execution_observation(),blas_preference=str(torch.backends.cuda.preferred_blas_library()),
            environment={k:os.environ.get(k) for k in ('DISABLE_ADDMM_CUDA_LT','TORCH_BLAS_PREFER_HIPBLASLT',
                'ROCBLAS_USE_HIPBLASLT','ROCBLAS_DEFAULT_ATOMICS_MODE','HIPBLASLT_LOG_MASK')},
            operand_sha256=expected,status='PASS')
    save(Path('/output/result.json'),result)


def run_one(host,image,helper,inputs,output,policy,expected):
    import uuid
    from oeis_learn.pipeline_validation.supervision import launch,reclaim
    from oeis_learn.pipeline_validation.generalization_lifecycle import guard
    from diagnose_hip import uncontained
    name='linear-replay-'+uuid.uuid4().hex
    output.mkdir();cleanup=dict(deadline=time.monotonic()+180,allowance_seconds=30)
    argv=command(host,image,helper,inputs,output,name,policy,expected)
    status='FAIL';reason=None
    try:
        try:
            with guard(output),host.gpu_lock():
                launch(argv,output/'supervision',seconds=150,stream_cap=1<<20,cleanup_seconds=30,cleanup_state=cleanup)
        finally:reclaim(host.engine,name,output/'supervision',cleanup)
        result=json.loads((output/'result.json').read_text());status=result['status']
    except KeyboardInterrupt:status='INTERRUPTED';result={}
    except Exception as exc:reason=type(exc).__name__;result={}
    unsafe=uncontained(output)
    return dict(status=status,result=result,reason=reason,containment='UNACKNOWLEDGED' if unsafe else 'ACKNOWLEDGED')


def export(output):
    from diagnose_hip import save
    from oeis_learn.pipeline_validation.generalization_diagnostics import control_fields
    from oeis_learn.pipeline_validation.bundle import pack,verify
    import shutil
    target=output/'export';target.mkdir()
    for name in ('summary.json','results.json'):shutil.copyfile(output/name,target/name)
    for trial in output.glob('trial-*'):
        for name in ('process.json','cleanup.json'):
            path=trial/'supervision'/name
            if path.is_file():save(target/trial.name/name,control_fields(json.loads(path.read_text())))
        for name in ('stdout.log','stderr.log'):
            path=trial/'supervision'/name
            if path.is_file():
                with path.open('rb') as stream:stream.seek(max(0,path.stat().st_size-65536));data=stream.read(65536)
                dest=target/trial.name/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    archive=pack(target,mandatory=['summary.json','results.json'],cap=16<<20,optional_cap=12<<20)
    destination=output.with_suffix('.zip')
    if destination.exists():raise ValueError('fresh replay archive required')
    archive.rename(destination);verify(destination);return destination


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=['docker','podman'],required=True)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
    from diagnose_hip import save,host_for
    from oeis_learn.pipeline_validation.identity import confined,file_hash
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope
    from oeis_learn.generalization.resources import CAPS
    import shutil
    run=confined(ROOT,args.run);output=confined(ROOT,args.output,exists=False)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh output and archive required')
    operand,expected,image=saved_operand(run)
    host=host_for(args.engine)
    if host.engine.image(image)!=image:raise ValueError('original immutable learner image required')
    host.select_gpu();output.mkdir(parents=True)
    rows=[]
    with scope(output,CAPS):
        inputs=output/'private/inputs';inputs.mkdir(parents=True)
        shutil.copyfile(operand,inputs/'linear-operands.pt')
        if file_hash(inputs/'linear-operands.pt')!=expected:raise ValueError('copied operand mismatch')
        helper=output/'private/helper';helper.mkdir()
        names=('replay_linear.py','trace_resume.py','diagnose_hip.py')
        for name in names:shutil.copyfile(Path(__file__).with_name(name),helper/name)
        metadata=dict(image=image,operand_sha256=expected,helpers={n:file_hash(Path(__file__).with_name(n)) for n in (*names,'test_replay_linear.py')})
        for policy,trial in schedule():
            print('linear replay: '+policy+' '+str(trial),flush=True)
            row=dict(policy=policy,trial=trial,**run_one(host,image,helper,inputs,output/('trial-'+str(trial)+'-'+policy),policy,expected))
            rows.append(row);save(output/'results.json',dict(metadata=metadata,rows=rows))
            if row['status']=='INTERRUPTED' or row['containment']=='UNACKNOWLEDGED':break
        summary=summarize(rows);summary['selected_scope']='PASS' if summary['complete'] else 'FAIL'
        save(output/'summary.json',summary)
    archive=export(output)
    print(json.dumps(dict(status=summary['selected_scope'],archive=str(archive.relative_to(ROOT)),feature_qualified=False),indent=2))
    return 0 if summary['complete'] else 1


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--container':
        try:worker(sys.argv[2],sys.argv[3])
        except Exception as exc:
            from diagnose_hip import save
            save(Path('/output/result.json'),dict(status='FAIL',reason=type(exc).__name__,error=str(exc)[:2048]))
            raise SystemExit(1)
    else:
        try:raise SystemExit(main())
        except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
