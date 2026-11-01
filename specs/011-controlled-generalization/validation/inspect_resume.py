#!/usr/bin/env python3
"""Read-only CPU forensics of trusted local learner state; no training or HIP."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents)>3 else None


def differences(left,right,*,limit=128):
    """Same exact equality as acceptance; disclose statistics, never values."""
    import torch
    import numpy as np
    count=0;leaves=0;details=[]
    def difference(path,reason,**extra):
        nonlocal count
        count+=1
        if len(details)<limit:details.append(dict(path=path,reason=reason,**extra))
    def walk(a,b,path):
        nonlocal leaves
        leaves+=1
        if leaves>100000:raise ValueError('state comparison node cap')
        if isinstance(a,torch.Tensor):
            if not isinstance(b,torch.Tensor):difference(path,'type');return
            if a.shape!=b.shape: difference(path,'shape',left_shape=list(a.shape),right_shape=list(b.shape));return
            if a.dtype!=b.dtype:difference(path,'dtype',left_dtype=str(a.dtype),right_dtype=str(b.dtype));return
            if torch.equal(a,b):return
            changed=0;nonfinite=0;maximum=0.
            av=a.detach().reshape(-1);bv=b.detach().reshape(-1)
            for offset in range(0,av.numel(),1<<20):
                x=av[offset:offset+(1<<20)].to(torch.float64)
                y=bv[offset:offset+(1<<20)].to(torch.float64)
                changed+=int((x!=y).sum());finite=torch.isfinite(x)&torch.isfinite(y)
                nonfinite+=int((~finite).sum())
                if finite.any():maximum=max(maximum,float((x[finite]-y[finite]).abs().max()))
            difference(path,'tensor_values',shape=list(a.shape),dtype=str(a.dtype),different_elements=changed,
                       elements=a.numel(),nonfinite_pairs=nonfinite,max_abs_finite=maximum)
        elif isinstance(a,np.ndarray):
            if not isinstance(b,np.ndarray) or a.shape!=b.shape or a.dtype!=b.dtype:
                difference(path,'array_type_shape_or_dtype')
            elif not np.array_equal(a,b):difference(path,'array_values',different_elements=int(np.count_nonzero(a!=b)),elements=a.size)
        elif isinstance(a,dict):
            if not isinstance(b,dict):difference(path,'type');return
            if set(a)!=set(b):difference(path,'dictionary_keys')
            for key in sorted(set(a)&set(b),key=str):
                # Keys can contain arbitrary program/task strings. Hash those.
                import re,hashlib
                label=str(key)
                if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,96}',label):label='key-'+hashlib.sha256(label.encode()).hexdigest()
                walk(a[key],b[key],path+'.'+label)
        elif isinstance(a,(list,tuple)):
            if type(a)!=type(b) or len(a)!=len(b):difference(path,'sequence_type_or_length');return
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'['+str(i)+']')
        elif type(a)!=type(b) or a!=b:difference(path,'scalar_value_or_type')
    walk(left,right,'state')
    return dict(equal=count==0,different_leaves=count,details=details,omitted_details=max(0,count-len(details)))


def inspect_saved(mounts,output):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.pipeline_validation.identity import file_hash,seal
    from diagnose_hip import save
    loaded={};summaries={};initials={}
    for name,folder in mounts.items():
        folder=Path(folder)
        result=json.loads((folder/'training.json').read_text())
        checkpoint=result['checkpoint'];root=folder/'checkpoints'
        descriptor=json.loads(verify_artifact(root,checkpoint,max_bytes=1<<20))
        store=GeneralizationCheckpointStore(root,result['compatibility'],descriptor['ledger']['cap'])
        state=store.load(checkpoint)
        summaries[name]=dict(checkpoint=checkpoint['sha256'],training_sha256=file_hash(folder/'training.json'),
                             compatibility=result['compatibility'],ledger=state['ledger'].snapshot(),
                             learner_seconds=state['states']['scheduler']['learner_seconds'])
        if name!='paused':loaded[name]=state
        roster=json.loads((root/'retained.json').read_text())
        if seal(roster)!=roster:raise ValueError('retained roster identity mismatch')
        first=next(c for c in roster['checkpoints'] if 'initial' in c['pins'])
        import io
        initials[name]=torch.load(io.BytesIO(verify_artifact(root,first['weights'])),map_location='cpu',weights_only=True)
    if len({v['compatibility'] for v in summaries.values()})!=1:
        raise ValueError('saved runs have different compatibility contracts')
    a,b=loaded['full'],loaded['resumed']
    components={key:differences(a['states'][key],b['states'][key]) for key in ('weights','optimizer','rng','data','replay')}
    components['ledger']=differences(a['ledger'].snapshot(),b['ledger'].snapshot())
    report=dict(version='generalization-resume-inspection/v1',status='PASS',
                feature_qualified=False,scientific_status='UNMEASURED',device='cpu',
                comparison_equal=all(v['equal'] for v in components.values()),
                runs=summaries,components=components,
                initial_full_paused=differences(initials['full'],initials['paused']),
                initial_full_resumed=differences(initials['full'],initials['resumed']),
                note='PASS means inspection completed; comparison_equal reports resume equality. No tolerance applied.')
    save(Path(output)/'inspection.json',report)
    return report


def saved_runs(run):
    from oeis_learn.pipeline_validation.identity import confined,seal
    run=Path(run)
    root=confined(run,'replay-resume/case/private/replay')
    record=json.loads((root/'replay.json').read_text())
    if seal(record)!=record or len(record['runs'])!=1:raise ValueError('one sealed replay run required')
    seed=record['runs'][0]
    if type(seed['seed']) is not int or not 0<=seed['seed']<2**32:raise ValueError('bounded seed required')
    full=next(r for r in seed['paired']['runs'] if r['arm']=='verified_replay')
    folder=confined(root,str(seed['seed']))
    paths={name:confined(folder,kind+'/'+full['root']) for name,kind in
           [('full','training'),('paused','paused'),('resumed','resumed')]}
    for name,path in paths.items():
        if not (path/'training.json').is_file():raise ValueError('missing saved learner: '+name)
    return paths


def command(host,image,helper,output,runs,name):
    from oeis_learn.container_runtime import private_runtime_environment
    environment=private_runtime_environment(output,destination='/output')
    environment.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return host.engine.command(image=image,name=name,
        mounts=[(helper,'/helper',False),*[(p,'/saved/'+n,False) for n,p in runs.items()],(output,'/output',True)],
        argv=['python','/helper/inspect_resume.py','--container'],gpu=False,
        memory_bytes=host.policy['memory_bytes'],cpus=host.policy['cpus'],network=False,
        tmp_bytes=2<<30,environment=environment)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=['docker','podman'],required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
    from diagnose_hip import save,host_for
    from oeis_learn.pipeline_validation.identity import confined,file_hash
    from oeis_learn.pipeline_validation.supervision import launch,reclaim
    from oeis_learn.pipeline_validation.bundle import pack,verify
    run=confined(ROOT,args.run);output=confined(ROOT,args.output,exists=False)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh output and archive required')
    runs=saved_runs(run)
    ledger=json.loads((run/'diagnostic.json').read_text())
    image=ledger['metadata']['images'][1]
    host=host_for(args.engine)
    if host.engine.image(image)!=image:raise ValueError('original immutable image required')
    import shutil,uuid
    output.mkdir(parents=True);private=output/'private';private.mkdir()
    helpers=private/'helper';helpers.mkdir()
    for filename in ('inspect_resume.py','diagnose_hip.py'):shutil.copyfile(Path(__file__).with_name(filename),helpers/filename)
    target=private/'container';target.mkdir()
    name='resume-inspection-'+uuid.uuid4().hex
    argv=command(host,image,helpers,target,runs,name)
    status='FAIL';cleanup=dict(deadline=time.monotonic()+300,allowance_seconds=30)
    try:
        try:
            launch(argv,private/'logs',seconds=270,stream_cap=1<<20,cleanup_seconds=30,cleanup_state=cleanup)
        finally:reclaim(host.engine,name,private/'logs',cleanup)
        result=json.loads((target/'inspection.json').read_text())
        save(output/'inspection.json',result);status=result['status']
    except (Exception,KeyboardInterrupt) as exc:
        save(output/'error.json',dict(reason=type(exc).__name__,error=str(exc)[:2048]))
        status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL'
    from oeis_learn.pipeline_validation.generalization_diagnostics import control_fields
    for filename in ('process.json','cleanup.json'):
        path=private/'logs'/filename
        if path.is_file():save(output/filename,control_fields(json.loads(path.read_text())))
    for filename in ('stdout.log','stderr.log'):
        path=private/'logs'/filename
        if path.is_file():
            with path.open('rb') as stream:stream.seek(max(0,path.stat().st_size-65536));data=stream.read(65536)
            (output/filename).write_bytes(data)
    save(output/'summary.json',dict(selected_scope=status,feature_qualified=False,scientific_status='UNMEASURED',
         scope='read-only saved-state inspection; no accelerator or training',image=image,
         helper_sha256=file_hash(Path(__file__)),diagnostic_helper_sha256=file_hash(Path(__file__).with_name('diagnose_hip.py')),
         tests_sha256=file_hash(Path(__file__).with_name('test_inspect_resume.py')),
         input_diagnostic_sha256=file_hash(run/'diagnostic.json')))
    archive=pack(output,mandatory=['summary.json'],cap=4<<20,optional_cap=2<<20);verify(archive)
    print(json.dumps(dict(status=status,archive=str(archive.relative_to(ROOT))),indent=2))
    return 0 if status=='PASS' else 1


if __name__=='__main__':
    if sys.argv[1:]==['--container']:
        inspect_saved({n:Path('/saved')/n for n in ('full','paused','resumed')},Path('/output'))
    else:
        try:raise SystemExit(main())
        except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
