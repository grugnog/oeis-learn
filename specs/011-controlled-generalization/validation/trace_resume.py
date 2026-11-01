#!/usr/bin/env python3
"""Temporary paired continuation controls and numerical tracing; never qualification."""
import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents)>3 else None
RUNS=('old-control','new-control','old-traced','new-traced')


def configure_policy(policy):
    """Experimental intervention confined to traced diagnostic learner processes."""
    if policy=='observe':return
    if policy!='rocblas':raise ValueError('unregistered diagnostic policy')
    import os
    # Set handle-creation defaults before importing Torch or creating BLAS handles.
    os.environ.update(ROCBLAS_USE_HIPBLASLT='0',ROCBLAS_DEFAULT_ATOMICS_MODE='0')
    import torch
    torch.backends.cuda.preferred_blas_library('cublas')
    if str(torch.backends.cuda.preferred_blas_library()).split('.')[-1].lower()!='cublas':
        raise RuntimeError('requested hipBLAS preference was not accepted')


def install_trace(path,policy='observe'):
    """Observe the existing update; CPU readbacks synchronize and may affect timing."""
    import hashlib,json,os,random
    from pathlib import Path
    import numpy as np
    import torch
    from oeis_learn.pipeline_validation import generalization_training as training
    from oeis_learn.pipeline_validation.sizing_execution import execution_observation
    from oeis_learn.rl import sft_trainer as sft
    path=Path(path);count=0
    def fingerprint(value):
        h=hashlib.sha256()
        def put(value):
            if isinstance(value,torch.Tensor):
                h.update(json.dumps(['tensor',str(value.dtype),list(value.shape)]).encode())
                flat=value.detach().reshape(-1)
                for offset in range(0,flat.numel(),1<<20):
                    h.update(flat[offset:offset+(1<<20)].contiguous().cpu().view(torch.uint8).numpy().tobytes())
            elif isinstance(value,np.ndarray):
                h.update(json.dumps(['array',str(value.dtype),list(value.shape)]).encode());h.update(value.tobytes())
            elif isinstance(value,dict):
                h.update(b'dict')
                for key in sorted(value,key=lambda x:(type(x).__name__,str(x))):put(key);put(value[key])
                h.update(b'end')
            elif isinstance(value,(tuple,list)):
                h.update(type(value).__name__.encode())
                for child in value:put(child)
                h.update(b'end')
            else:h.update(json.dumps([type(value).__name__,value],allow_nan=False).encode())
        put(value);return 'sha256:'+h.hexdigest()
    def emit(phase,update=0,**fields):
        nonlocal count
        if count>=32:raise ValueError('bounded trace record cap')
        record=dict(phase=phase,update=update,**fields)
        text=json.dumps(record,allow_nan=False)
        if len(text)>8192:raise ValueError('bounded trace record bytes')
        with path.open('a') as stream:stream.write(text+'\n')
        count+=1
    original=training.torch_update
    def factory(encoder,decoder,optimizer,*,device,scheduler=None):
        update=original(encoder,decoder,optimizer,device=device,scheduler=scheduler)
        emit('execution',torch=str(torch.__version__),hip=torch.version.hip,device=device,
            diagnostic_policy=policy,blas_preference=str(torch.backends.cuda.preferred_blas_library()),
            device_name=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
            execution=execution_observation(),threads=torch.get_num_threads(),
            environment={k:os.environ.get(k) for k in ('PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG',
                'HIPBLAS_WORKSPACE_CONFIG','ROCBLAS_DEFAULT_ATOMICS_MODE','ROCBLAS_USE_HIPBLASLT')})
        ordinal=0
        def snapshot():
            return dict(weights=fingerprint(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict())),
                optimizer=fingerprint(optimizer.state_dict()),rng=fingerprint(dict(python=random.getstate(),
                numpy=np.random.get_state(),torch=torch.get_rng_state(),
                device=torch.cuda.get_rng_state_all() if device=='cuda' else [])))
        def observed(batch):
            nonlocal ordinal
            ordinal+=1
            if ordinal>4:raise ValueError('trace limited to four diagnostic updates')
            logits=sft.teacher_forced_logits;clip=torch.nn.utils.clip_grad_norm_
            def forward(*args,**kwargs):
                result=logits(*args,**kwargs)
                emit('forward',ordinal,logits=fingerprint(result));return result
            def gradients(parameters,*args,**kwargs):
                parameters=list(parameters)
                emit('backward',ordinal,gradients=fingerprint([p.grad for p in parameters]))
                result=clip(parameters,*args,**kwargs)
                emit('clip',ordinal,gradients=fingerprint([p.grad for p in parameters]),norm=fingerprint(result))
                return result
            try:
                emit('before',ordinal,batch=fingerprint(batch),
                    buffers=fingerprint(dict(encoder=dict(encoder.named_buffers()),decoder=dict(decoder.named_buffers()))),
                    modes=fingerprint([[m.training for m in model.modules()] for model in (encoder,decoder)]),**snapshot())
                sft.teacher_forced_logits=forward;torch.nn.utils.clip_grad_norm_=gradients
                result=update(batch)
                emit('after',ordinal,**snapshot())
                return result
            finally:
                sft.teacher_forced_logits=logits;torch.nn.utils.clip_grad_norm_=clip
        return observed
    training.torch_update=factory
    def undo():training.torch_update=original
    return undo


@contextmanager
def instrument(policy='observe'):
    from oeis_learn.pipeline_validation import generalization_roles as roles
    original=roles.ROLE_PROGRAM
    # Installed role code and its canonical mounts/deadlines are unchanged. Only
    # this diagnostic process receives the separately hashed observer bootstrap.
    if policy not in {'observe','rocblas'}:raise ValueError('unregistered diagnostic policy')
    roles.ROLE_PROGRAM=(inspect.getsource(configure_policy)+'\n'+inspect.getsource(install_trace)+
        "\nimport sys,json\nif sys.argv[-2]=='learner' and json.loads(sys.argv[-1])['operation']=='train':\n"
        f"    configure_policy({policy!r})\n    install_trace('/output/learner/update-trace.jsonl',{policy!r})\n"+original)
    try:yield
    finally:roles.ROLE_PROGRAM=original


def plan(args):
    result=[]
    for name in ('compatibility',*RUNS,'compare'):
        result.append(dict(name=name,operation=name,target=str(Path(args.output)/name),
            seconds=600 if name=='compatibility' else 300 if name=='compare' else 780,
            needs=[] if name=='compatibility' else ['compatibility'],
            argv=[sys.executable,str(Path(__file__).resolve()),'--engine',args.engine,'--run',args.run,
                  '--matrix',args.matrix,'--output',str(Path(args.output)/name),'--internal',name,
                  '--policy',getattr(args,'policy','observe')]))
    return result


def first_difference(left,right):
    from itertools import zip_longest
    for index,(a,b) in enumerate(zip_longest(left,right)):
        if a==b:continue
        if a is None or b is None:return dict(equal=False,index=index,phase='missing_record')
        return dict(equal=False,index=index,phase=a['phase'] if a['phase']==b['phase'] else 'phase_order',
            update=a['update'],fields=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k)))
    return dict(equal=True)


def read_trace(path,*,complete=True):
    if path.stat().st_size>64<<10:raise ValueError('trace byte cap')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if not rows or len(rows)>32 or rows[0].get('phase')!='execution':raise ValueError('trace header/record cap')
    expected=[('execution',0)]+[(phase,update) for update in (1,2)
                              for phase in ('before','forward','backward','clip','after')]
    actual=[(r.get('phase'),r.get('update')) for r in rows]
    if actual!=(expected if complete else expected[:len(actual)]):raise ValueError('ordered diagnostic update trace required')
    return rows


def context(args):
    import probe_resume
    from inspect_resume import matrix_pause
    from oeis_learn.pipeline_validation.identity import confined
    value=probe_resume.load_context(args.run)
    if value['paused_updates']!=1 or value['final_updates']!=3:raise ValueError('registered one-to-three-update diagnostic required')
    matrix=confined(ROOT,args.matrix)
    value['new_paused']=matrix_pause(matrix,value['run'],value['runtime']['images'])
    return value


def internal(args):
    from contextlib import nullcontext
    import probe_resume
    from diagnose_hip import host_for,save
    from oeis_learn.pipeline_validation.identity import confined,digest
    ctx=context(args);output=confined(ROOT,args.output,exists=False);output.mkdir(parents=True,exist_ok=False)
    host=host_for(args.engine)
    if args.internal=='compatibility':
        from oeis_learn.pipeline_validation.generalization_commands import _qualification
        from types import SimpleNamespace
        q=_qualification(host,ROOT,output,ctx['base'],ctx['study'],ctx['runtime'],
            SimpleNamespace(qualification=str(ctx['original']/'qualify')))
        return dict(status='PASS',original=q['original'])
    if args.internal=='compare':return compare(host,ctx,output)
    ctx=dict(ctx,paused=ctx['new_paused'] if args.internal.startswith('new-') else ctx['paused'])
    traced=args.internal.endswith('-traced')
    try:
        with (instrument() if getattr(args,'policy','observe')=='observe' else instrument(args.policy)) if traced else nullcontext():
            result=probe_resume.train_branch('continue-a',ctx,host,output)
    finally:
        trace=output/'replay-resume/case/private/training'/digest(['verified_replay',ctx['seed']])[7:23]/'execution/role/update-trace.jsonl'
        if traced and trace.is_file():save(output/'trace.json',read_trace(trace,complete=False))
    if args.internal.endswith('-traced'):
        learner=confined(ROOT,result['learner'])
        result['trace']=read_trace(learner.parent.parent/'update-trace.jsonl')
    return result


def comparison_container(saved=Path('/saved'),output=Path('/output'),trace_path=Path('/helper/traces.json')):
    import torch
    from probe_resume import load_saved,compare_states
    from diagnose_hip import save
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    states={};refs={};missing=[]
    for name in (*RUNS,'original','old-resumed'):
        root=saved/name
        if root.exists():states[name],refs[name]=load_saved(root)
        else:missing.append(name)
    if len({r['compatibility'] for r in refs.values()})!=1:raise ValueError('comparison contracts differ')
    comparisons=compare_states(states)
    traces=json.loads(trace_path.read_text())
    complete=all(n in states for n in RUNS)
    difference=first_difference(traces['old-traced'][1:],traces['new-traced'][1:]) if all(n in traces for n in ('old-traced','new-traced')) else None
    def equal(a,b):return comparisons.get(a+':'+b,{}).get('equal')
    policies={r[0].get('diagnostic_policy','observe') for r in traces.values() if r}
    intervention='rocblas' in policies
    save(output/'comparison.json',dict(status='PASS',feature_qualified=False,scientific_status='UNMEASURED',
        starting_completed_updates=1,trace_update_numbers='continuation ordinals: 1 is overall update 2',
        complete=complete,missing=missing,checkpoints=refs,comparisons=comparisons,traces=traces,
        control_equal=equal('old-control','new-control'),traced_equal=equal('old-traced','new-traced'),
        diagnostic_policy='rocblas' if intervention else 'observe',
        observer_effect=None if intervention else dict(old_equal=equal('old-control','old-traced'),new_equal=equal('new-control','new-traced')),
        intervention_effect=dict(old_equal=equal('old-control','old-traced'),new_equal=equal('new-control','new-traced')) if intervention else None,
        first_trace_difference=difference,
        note='Readbacks synchronize execution and may change timing. First observed difference is not a backend diagnosis.'))


def compare(host,ctx,output):
    import shutil,uuid
    import probe_resume
    from inspect_resume import command
    from diagnose_hip import save
    from oeis_learn.pipeline_validation.supervision import launch,reclaim
    inputs={'original':ctx['saved']['full'],'old-resumed':ctx['saved']['resumed']};traces={}
    for name in RUNS:
        stage=output.parent/name;path=stage/'result.json'
        if path.is_file():
            result=json.loads(path.read_text())
            if 'trace' in result:traces[name]=result['trace']
            if result['status']=='PASS':
                inputs[name]=probe_resume.learner_output(stage)
    helpers=output/'private/helper';helpers.mkdir(parents=True)
    for name in ('trace_resume.py','probe_resume.py','inspect_resume.py','diagnose_hip.py'):
        shutil.copyfile(Path(__file__).with_name(name),helpers/name)
    save(helpers/'traces.json',traces)
    target=output/'private/comparison';target.mkdir()
    container='resume-trace-'+uuid.uuid4().hex
    argv=command(host,ctx['runtime']['images'][1],helpers,target,inputs,container)
    argv[argv.index('/helper/inspect_resume.py')]='/helper/trace_resume.py'
    cleanup=dict(deadline=time.monotonic()+270,allowance_seconds=30)
    try:launch(argv,output/'supervision',seconds=240,stream_cap=1<<20,cleanup_seconds=30,cleanup_state=cleanup)
    finally:reclaim(host.engine,container,output/'supervision',cleanup)
    report=json.loads((target/'comparison.json').read_text());save(output/'comparison.json',report)
    return dict(status='PASS',complete=report['complete'],feature_qualified=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=['docker','podman'],required=True)
    p.add_argument('--run',required=True);p.add_argument('--matrix',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--policy',choices=['observe','rocblas'],default='observe')
    p.add_argument('--internal',choices=['compatibility',*RUNS,'compare'],help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
    from diagnose_hip import save,execute,run
    from oeis_learn.pipeline_validation.identity import confined,file_hash
    output=confined(ROOT,args.output,exists=False)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh trace output required')
    if args.internal:
        try:result=internal(args)
        except Exception as exc:result=dict(status='FAIL',reason=type(exc).__name__,error=str(exc)[:2048])
        if (output/'trace.json').is_file():result['trace']=json.loads((output/'trace.json').read_text())
        save(output/'result.json',result);return 0 if result['status']=='PASS' else 1
    ctx=context(args);stages=plan(args)
    if args.dry_run:print(json.dumps(dict(stages=stages,ceiling_seconds=4020),indent=2));return 0
    from oeis_learn.generalization.resources import CAPS
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope
    import probe_resume
    output.mkdir(parents=True)
    metadata=dict(source_run=args.run,matrix=args.matrix,policy=args.policy,images=ctx['runtime']['images'],helpers={
        n:file_hash(Path(__file__).with_name(n)) for n in ('trace_resume.py','test_trace_resume.py',
        'probe_resume.py','inspect_resume.py','diagnose_hip.py')})
    with scope(output,CAPS):record=execute(stages,output,runner=run,metadata=metadata)
    # Reuse the existing diagnostic exporter, with this helper's branch names.
    prior=probe_resume.BRANCHES
    try:probe_resume.BRANCHES=RUNS;archive=probe_resume.export(output)
    finally:probe_resume.BRANCHES=prior
    print(json.dumps(dict(status=record['status'],archive=str(archive.relative_to(ROOT)),feature_qualified=False),indent=2))
    return 0 if record['status']=='PASS' else 1


if __name__=='__main__':
    if sys.argv[1:]==['--container']:comparison_container()
    else:
        try:raise SystemExit(main())
        except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
