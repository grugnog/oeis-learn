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
    totals=dict(tensor_different_elements=0,tensor_nonfinite_pairs=0,max_tensor_abs_finite=0.)
    def difference(path,reason,**extra):
        nonlocal count
        count+=1
        if reason=='tensor_values':
            totals['tensor_different_elements']+=extra['different_elements']
            totals['tensor_nonfinite_pairs']+=extra['nonfinite_pairs']
            totals['max_tensor_abs_finite']=max(totals['max_tensor_abs_finite'],extra['max_abs_finite'])
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
    return dict(equal=count==0,different_leaves=count,details=details,omitted_details=max(0,count-len(details)),**totals)


def representation(value):
    """Value-free layout/order/alias description; never export memory addresses."""
    import hashlib
    import numpy as np
    import torch
    rows={};storages={};nodes=0
    def key(value):
        return hashlib.sha256((type(value).__name__+':'+str(value)).encode()).hexdigest()
    def walk(value,path):
        nonlocal nodes
        nodes+=1
        if nodes>100000:raise ValueError('representation node cap')
        row=dict(type=type(value).__module__+'.'+type(value).__qualname__)
        rows[path]=row
        if isinstance(value,torch.Tensor):
            if value.device.type!='cpu' or value.layout!=torch.strided:
                raise ValueError('CPU strided inspection tensor required')
            storage=value.untyped_storage()
            # IDs are used only inside this process to assign stable alias classes.
            alias=storages.setdefault(storage._cdata,len(storages))
            row.update(shape=list(value.shape),dtype=str(value.dtype),stride=list(value.stride()),
                storage_offset=value.storage_offset(),storage_bytes=storage.nbytes(),alias=alias,
                requires_grad=value.requires_grad,is_conj=value.is_conj(),is_neg=value.is_neg())
        elif isinstance(value,np.ndarray):
            row.update(shape=list(value.shape),dtype=str(value.dtype),strides=list(value.strides))
        elif isinstance(value,dict):
            row['key_order']=[key(k) for k in value]
            for k in sorted(value,key=key):walk(value[k],path+'.'+key(k))
        elif isinstance(value,(tuple,list)):
            row['length']=len(value)
            for i,item in enumerate(value):walk(item,path+'.'+str(i))
    walk(value,'state')
    return rows


def inspect_pause_pair(old,new):
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.contracts import verify_artifact
    states=[];records=[]
    for root in (old,new):
        result=json.loads((root/'training.json').read_text())
        ref=result['checkpoint'];record=json.loads(verify_artifact(root/'checkpoints',ref))
        store=GeneralizationCheckpointStore(root/'checkpoints',result['compatibility'],record['ledger']['cap'])
        state=store.load(ref)
        if state['ledger'].updates!=1:raise ValueError('one-update pause required')
        states.append(state)
        records.append(dict(checkpoint=ref['sha256'],compatibility=result['compatibility'],
            serialized={k:({n:r['sha256'] for n,r in v.items()} if k=='rng' else v['sha256'])
                        for k,v in record['states'].items() if k!='data'}))
    if records[0]['compatibility']!=records[1]['compatibility']:raise ValueError('pause contracts differ')
    left,right=states
    components={k:differences(left['states'][k],right['states'][k])
                for k in ('weights','optimizer','rng','data','replay')}
    components['ledger']=differences(left['ledger'].snapshot(),right['ledger'].snapshot())
    old_layout=representation({k:left['states'][k] for k in components if k!='ledger'})
    new_layout=representation({k:right['states'][k] for k in components if k!='ledger'})
    metadata=differences(old_layout,new_layout)
    changed=[p for p in sorted(set(old_layout)|set(new_layout)) if old_layout.get(p)!=new_layout.get(p)]
    layout_rows=[]
    for path in changed[:16]:
        row=dict(path=path,old=old_layout.get(path),new=new_layout.get(path))
        if len(json.dumps(row))<=32768:layout_rows.append(row)
    # Include excluded state categories explicitly, without mistaking provenance
    # or elapsed-time differences for a numerical mismatch or its cause.
    excluded={k:differences(left['states'][k],right['states'][k])
              for k in ('scheduler','history','pins','selection')}
    return dict(values_equal=all(c['equal'] for c in components.values()),components=components,
        representation_equal=metadata['equal'],representation=metadata,layout_changes=layout_rows,
        omitted_layout_changes=len(changed)-len(layout_rows),excluded=excluded,checkpoints=records,
        note='CPU-loaded representations only; historical device allocations/backend state are unavailable. No causal conclusion.')


def matrix_pause(matrix,run,images):
    from oeis_learn.pipeline_validation.identity import confined
    metadata=json.loads((matrix/'diagnostic.json').read_text())['metadata']
    if metadata['start']!='fresh' or metadata['images']!=images or confined(ROOT,metadata['source_run'])!=run:
        raise ValueError('fresh matrix must bind the original diagnostic and images')
    row=json.loads((matrix/'split-first/result.json').read_text())
    if row['status']!='PASS' or row['learner_status']!='PARTIAL' or row['updates']!=1:
        raise ValueError('completed fresh one-update pause required')
    learner=confined(ROOT,row['learner'])
    if not learner.is_relative_to(matrix/'split-first'):raise ValueError('matrix learner escaped producer stage')
    result=json.loads((learner/'training.json').read_text())
    if result['updates']!=1 or result['status']!='PARTIAL' or result['checkpoint']['sha256']!=row['checkpoint']:
        raise ValueError('matrix checkpoint result binding mismatch')
    return learner


def inspect_saved(mounts,output):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.pipeline_validation.identity import file_hash,seal
    from diagnose_hip import save
    loaded={};summaries={};initials={}
    for name in ('full','paused','resumed'):
        folder=mounts[name]
        folder=Path(folder)
        result=json.loads((folder/'training.json').read_text())
        checkpoint=result['checkpoint'];root=folder/'checkpoints'
        descriptor=json.loads(verify_artifact(root,checkpoint,max_bytes=1<<20))
        store=GeneralizationCheckpointStore(root,result['compatibility'],descriptor['ledger']['cap'])
        state=store.load(checkpoint)
        summaries[name]=dict(checkpoint=checkpoint['sha256'],training_sha256=file_hash(folder/'training.json'),
                             compatibility=result['compatibility'],ledger=state['ledger'].snapshot(),
                             learner_seconds=state['states']['scheduler']['learner_seconds'])
        loaded[name]=state
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
    boundary=pause_boundary(Path(mounts['full']),loaded['paused'])
    report=dict(version='generalization-resume-inspection/v2',status='PASS',
                feature_qualified=False,scientific_status='UNMEASURED',device='cpu',
                comparison_equal=all(v['equal'] for v in components.values()),
                runs=summaries,components=components,pause_boundary=boundary,
                initial_full_paused=differences(initials['full'],initials['paused']),
                initial_full_resumed=differences(initials['full'],initials['resumed']),
                note='PASS means inspection completed; comparison_equal reports resume equality. No tolerance applied.')
    if 'matrix-paused' in mounts:
        # Release prior full states before loading the extra pair.
        del loaded,a,b,state,initials
        report['pause_pair']=inspect_pause_pair(Path(mounts['paused']),Path(mounts['matrix-paused']))
    save(Path(output)/'inspection.json',report)
    return report


def pause_boundary(full,paused):
    """Compare retained pre/post-save points at the same completed update count.

    A pre-update save may already hold the NEXT batch/debit/replay cursor. Compare
    weights, optimizer and RNG only; never pretend its data/ledger equals pause.
    Inspect every retained match rather than selecting a favorable checkpoint.
    """
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.pipeline_validation.identity import seal
    root=full/'checkpoints'
    result=json.loads((full/'training.json').read_text())
    roster=json.loads((root/'retained.json').read_text())
    if seal(roster)!=roster or len(roster['checkpoints'])>128:raise ValueError('bounded sealed retained roster required')
    matches=[];seen=set()
    for item in roster['checkpoints']:
        pointer=item['state']
        if pointer['sha256'] in seen:continue
        seen.add(pointer['sha256'])
        record=json.loads(verify_artifact(root,pointer,max_bytes=1<<20))
        if record['ledger']['updates']!=paused['ledger'].updates:continue
        if len(matches)>=8:raise ValueError('pause-boundary candidate cap')
        store=GeneralizationCheckpointStore(root,result['compatibility'],paused['ledger'].cap)
        state=store.load(pointer)
        components={key:differences(state['states'][key],paused['states'][key]) for key in ('weights','optimizer','rng')}
        matches.append(dict(checkpoint=pointer['sha256'],ledger=state['ledger'].snapshot(),components=components,
                            equal=all(v['equal'] for v in components.values())))
    return dict(available=bool(matches),updates=paused['ledger'].updates,candidates=matches,
                interpretation=('No retained state at the pause update count; divergence timing remains unknown.' if not matches else
                'A mismatch here precedes resumption. A match localizes the observed difference to subsequent execution; it does not by itself identify the cause.'),
                compared=['weights','optimizer','rng'],excluded=['data','replay','ledger','scheduler','history','pins','selection'])


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
    parser.add_argument('--matrix',help='Optional completed fresh matrix for CPU-only old/new pause inspection')
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
    matrix=confined(ROOT,args.matrix) if args.matrix else None
    if matrix is not None:runs['matrix-paused']=matrix_pause(matrix,run,ledger['metadata']['images'])
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
    if matrix is not None:
        save(output/'matrix-input.json',dict(diagnostic_sha256=file_hash(matrix/'diagnostic.json'),
             pause_result_sha256=file_hash(matrix/'split-first/result.json')))
    mandatory=['summary.json']+(['inspection.json'] if (output/'inspection.json').is_file() else [])
    archive=pack(output,mandatory=mandatory,cap=4<<20,optional_cap=2<<20);verify(archive)
    print(json.dumps(dict(status=status,archive=str(archive.relative_to(ROOT))),indent=2))
    return 0 if status=='PASS' else 1


if __name__=='__main__':
    if sys.argv[1:]==['--container']:
        inspect_saved({n:Path('/saved')/n for n in ('full','paused','resumed','matrix-paused')
                       if (Path('/saved')/n).exists()},Path('/output'))
    else:
        try:raise SystemExit(main())
        except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
