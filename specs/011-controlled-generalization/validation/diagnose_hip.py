#!/usr/bin/env python3
"""Disposable selective diagnostics; never a qualification or research authority.

Host dependencies are stdlib only. Keep this helper outside qualification sources:
it consumes an unchanged installed build and records its own hash separately.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else None
CASES = ('refresh', 'parameter-targets', 'failure-packaging', 'retained-eval', 'replay-resume')
# Resume covers three 180s learner calls plus proof, startup and cleanup.
PROBES = {'native': 180, 'resume': 720, 'retained': 900}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def install_trace():
    """Observe existing scheduler callbacks; no new deadlines or frame walking."""
    import dataclasses
    import hashlib
    import json
    import sys
    import threading
    import time
    from oeis_learn.fresh import pipeline_scheduler as scheduler
    from oeis_learn.fresh.pipeline_jobs import NativeRequest
    original = scheduler.drive
    lock = threading.Lock()
    count = 0

    def emit(phase, request=None):
        nonlocal count
        with lock:
            if count >= 256:
                return
            count += 1
            event = dict(phase=phase, monotonic=time.monotonic())
            if request is not None:
                event.update(job=hashlib.sha256(str(request.job['job_id']).encode()).hexdigest(),
                             remaining_seconds=request.deadline-time.monotonic(),
                             cancel_available=request.cancel is not None)
            print('native-trace: ' + json.dumps(event), file=sys.stderr, flush=True)

    def wrapped(request):
        def operation():
            emit('native-start', request)
            try:
                return request.operation()
            finally:
                emit('native-end', request)
        def cancel():
            emit('cancel-start', request)
            try:
                return request.cancel()
            finally:
                emit('cancel-end', request)
        emit('native-queued', request)
        return dataclasses.replace(request, operation=operation,
                                   cancel=cancel if request.cancel is not None else None)

    def observed(generator):
        value = None
        try:
            while True:
                try:
                    request = generator.send(value)
                except StopIteration as done:
                    return done.value
                value = yield wrapped(request) if isinstance(request, NativeRequest) else request
        finally:
            generator.close()

    def drive(decoder, generators, **kwargs):
        emit('scheduler-start')
        try:
            return original(decoder, [observed(g) for g in generators], **kwargs)
        finally:
            emit('scheduler-end')
    scheduler.drive = drive


def plan(args, image):
    from oeis_learn.generalization.resources import CASE_SECONDS
    output = Path(args.output)
    stages = []
    def helper(name, operation, seconds, needs):
        target = output / name
        argv = [sys.executable, str(Path(__file__).resolve()), '--internal', operation,
                '--engine', args.engine, '--manifest', args.manifest,
                '--previous', args.previous, '--output', str(target)]
        stages.append(dict(name=name, operation=operation, seconds=seconds,
                           needs=needs, argv=argv, target=str(target)))
    helper('compatibility', 'compatibility', 600, [])
    for probe in dict.fromkeys(args.probes):
        helper('probe-'+probe, probe, PROBES[probe], ['compatibility'])
    selected = set(args.cases)
    if 'replay-resume' in selected:
        selected.add('retained-eval')
    for case in CASES:
        if case not in selected:
            continue
        target = output / case
        argv = [sys.executable, str(ROOT/'scripts/foundation_run.py'), '--engine', args.engine,
                '--image', image, 'generalization', 'acceptance', '--manifest', args.manifest,
                '--qualification', str(Path(args.previous)/'qualify'),
                '--inputs', str(Path(args.previous)/'prepare'), '--case', case, '--output', str(target)]
        stages.append(dict(name=case, operation='acceptance', seconds=CASE_SECONDS[case],
                           needs=['compatibility']+(['retained-eval'] if case=='replay-resume' else []),
                           argv=argv, target=str(target)))
    return stages


def execute(stages, root, *, runner, metadata=None):
    record = dict(version='generalization-diagnostic/v1', status='PARTIAL',
                  feature_qualified=False, scientific_status='UNMEASURED',
                  plan=stages, results=[], metadata=metadata or {})
    finished = {}
    stopped = False
    for stage in stages:
        missing = [n for n in stage['needs'] if finished.get(n) != 'PASS']
        began = time.monotonic()
        if stopped or missing:
            result = dict(status='BLOCKED', reason='CONTAINMENT_OR_INTERRUPT' if stopped else 'DEPENDENCY', dependencies=missing)
        else:
            print('diagnostic: '+stage['name'], flush=True)
            try:
                result = runner(stage, root)
            except KeyboardInterrupt:
                result = dict(status='INTERRUPTED')
            except Exception as exc:
                result = dict(status='FAIL', reason=type(exc).__name__, containment='UNACKNOWLEDGED')
        if result['status']=='INTERRUPTED' or result.get('containment')=='UNACKNOWLEDGED':
            stopped = True
        finished[stage['name']] = result['status']
        record['results'].append(dict(name=stage['name'], elapsed_seconds=time.monotonic()-began, **result))
        record['status'] = ('INTERRUPTED' if any(r['status']=='INTERRUPTED' for r in record['results']) else
                            'FAIL' if any(r['status']!='PASS' for r in record['results']) else 'PASS')
        save(Path(root)/'diagnostic.json', record)
    return record


def uncontained(root):
    """Inspect controls only; bounded walk, no checkpoint/cache/symlink traversal."""
    visited = 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'.runtime','checkpoints','input','inputs','selected','native-cache','cells'}
                   and not (Path(folder)/d).is_symlink()]
        visited += len(dirs)+len(files)
        if visited > 12000:
            return True
        for name in ('cleanup.json', 'process.json'):
            path = Path(folder)/name
            if name not in files:
                continue
            if path.is_symlink() or path.stat().st_size > 256<<10:
                return True
            value = json.loads(path.read_text())
            if name=='cleanup.json' and (value.get('acknowledged') is not True or value.get('within_bound') is not True):
                return True
            if name=='process.json' and (value.get('cleanup_failure') or value.get('outcome')=='INTERRUPTED'):
                return True
    return False


def run(stage, root):
    from generalization_hip_serial import run as canonical_run
    code = canonical_run(stage, Path(root), stage['seconds'])
    target = ROOT/stage['target']
    result_path = target/'result.json'
    value = json.loads(result_path.read_text()) if result_path.is_file() else {}
    status = value.get('status', 'FAIL')
    if code == 130:
        status = 'INTERRUPTED'
    elif code or status != 'PASS':
        status = 'FAIL'
    outer = Path(root)/'private/logs'/stage['name']/'process.json'
    control = json.loads(outer.read_text()) if outer.is_file() else {}
    unsafe = (uncontained(target) or bool(code and not result_path.is_file()) or bool(control.get('cleanup_failure')) or
              bool(control.get('failure','')) and 'Timeout' in str(control.get('failure')))
    measurements={k:value[k] for k in ('device','updates','resume_equal','resume_checkpoint','admitted','restore_equal','model_discoveries','original') if k in value}
    return dict(status=status, exit_code=code, measurements=measurements,
                containment='UNACKNOWLEDGED' if unsafe else 'NO_UNACKNOWLEDGED_CLEANUP',
                reason=value.get('reason') or value.get('result',{}).get('reason'))


def export(root):
    from oeis_learn.pipeline_validation.generalization_diagnostics import collect
    from oeis_learn.pipeline_validation.bundle import pack, verify
    root = Path(root)
    destination = root/'export'
    destination.mkdir()
    record = json.loads((root/'diagnostic.json').read_text())
    save(destination/'diagnostic.json', record)
    save(destination/'summary.json', dict(selected_scope=record['status'],
         feature_qualified=False, scientific_status='UNMEASURED', scope='selective diagnostics only'))
    collect(root, destination/'diagnostics')
    # Traced reproduction uses the normal collector's case layout.
    if (root/'probe-retained').is_dir():
        collect(root/'probe-retained', destination/'retained-probe')
    from oeis_learn.generalization.resources import log_tail
    from oeis_learn.pipeline_validation.generalization_diagnostics import terminal_fields, control_fields
    for stage in record.get('plan', []):
        name=stage['name']
        for channel in ('stdout','stderr'):
            path=root/'private/logs'/name/(channel+'.log')
            if path.is_file() and not path.is_symlink():
                # Supervisor output is capped at 1MiB; only export a 64KiB excerpt.
                with path.open('rb') as stream:
                    opening=stream.read(32768);stream.seek(max(len(opening),path.stat().st_size-32768))
                    tail=stream.read(32768)
                target=destination/'logs'/(name+'-'+channel+'.txt');target.parent.mkdir(exist_ok=True)
                target.write_text(log_tail((opening+tail).decode(errors='replace')))
        for suffix in ('result.json','probe/cleanup.json','probe/logs/process.json'):
            path=root/name/suffix
            if path.is_file() and not path.is_symlink() and path.stat().st_size<=256<<10:
                value=json.loads(path.read_text())
                projected=terminal_fields(value) if suffix=='result.json' else control_fields(value)
                save(destination/'controls'/name/suffix,projected)
    archive=pack(destination, mandatory=['summary.json','diagnostic.json','diagnostics/index.json'],
                 cap=32<<20, optional_cap=24<<20)
    target=root.with_suffix('.zip')
    if target.exists():
        raise ValueError('diagnostic archive exists')
    archive.rename(target)
    verify(target)
    return target


def context(args):
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.generalization_session import load_session
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.pipeline_validation.generalization_runtime import load
    session, children=load_session(ROOT,args.manifest)
    previous=confined(ROOT,args.previous)
    prepared=json.loads((previous/'prepare/prepared.json').read_text())
    from oeis_learn.pipeline_validation.identity import validate
    validate(prepared, {'version','session','cases'}, 'generalization-prepared-session/v1')
    if prepared['session']!=session['identity'] or set(prepared['cases'])!=set(children):
        raise ValueError('prepared session differs from supplied manifest')
    refs=prepared['cases']['retained-eval']
    for ref in refs.values():
        verify_artifact(ROOT,ref,max_bytes=1<<20)
    base,study,runtime=load(ROOT,refs['manifest']['path'],refs['runtime']['path'])
    return previous,session,children,base,study,runtime


def host_for(engine):
    from foundation_local import Host, parser
    return Host(parser().parse_args(['--engine',engine,'status','--run','run-state/diagnostic-unused']))


def container_probe(name):
    """Tiny real installed paths; diagnostic output has no qualification schema."""
    root=Path('/output')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if name=='resume':
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('actual HIP required; CPU fallback forbidden')
        sys.path.insert(0,'/opt/oeis-build')
        from tests.integration.test_generalization_gpu import resume_probe
        from oeis_learn.fresh.bigint_config import resolve
        result=resume_probe(root/'private/resume',resolve(dict(version='bigint-bootstrap/v1')),'cuda')
        return dict(status='PASS',device='hip',**result)
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.pipeline_validation.generalization_replay import search_round,restore_pool
    from oeis_learn.decoder.bigint_codec import encode_body
    source='(program (return (int 0)))'
    task=dict(id='diagnostic-zero',role='training',ancestry=['mechanical','training'],
              source='(program (return (add (int 0) (int 0))))',
              indices=list(map(str,range(100))),values=['0']*100)
    candidate=dict(original_source=source,final_source=source,assignment={},tokens=encode_body(source))
    from oeis_learn.pipeline_validation.identity import digest
    with SynthesisClient(digest('diagnostic-native'),cache_root=root/'private/native-cache') as client:
        pool,report=search_round([task],propose=lambda *_:candidate,client=client,
                                checkpoint='diagnostic',seed=7,seconds=60,diagnostic=True)
        restored=restore_pool(pool.snapshot(),tasks=[task],client=client,deadline=time.monotonic()+60)
        if report['status']!='PASS' or len(pool.entries)!=1 or restored.snapshot()!=pool.snapshot():
            raise ValueError('native admission/restore probe incomplete')
    return dict(status='PASS',admitted=1,restore_equal=True,model_discoveries=0)


def launch_probe(host, image, name, output):
    import uuid
    import shutil
    from oeis_learn.container_runtime import private_runtime_environment
    from oeis_learn.pipeline_validation.supervision import launch,reclaim
    from contextlib import nullcontext
    target=output/'probe';target.mkdir()
    helper=output/'private/helper';helper.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve(),helper/'diagnose_hip.py')
    environment=private_runtime_environment(target,destination='/output')
    environment.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1')
    gpu=name=='resume'
    if gpu:host.select_gpu()
    container='generalization-diagnostic-'+uuid.uuid4().hex
    command=host.engine.command(image=image,name=container,
        mounts=[(helper,'/diagnostic',False),(target,'/output',True)],
        argv=['python','/diagnostic/diagnose_hip.py','--container-probe',name],gpu=gpu,
        render_node=host.policy['render_node'] if gpu else None,
        memory_bytes=host.policy['memory_bytes'],cpus=host.policy['cpus'],network=False,
        tmp_bytes=2<<30,environment=environment)
    cleanup=dict(deadline=time.monotonic()+PROBES[name]-15,allowance_seconds=30)
    try:
        with host.gpu_lock() if gpu else nullcontext():
            launch(command,target/'logs',seconds=PROBES[name]-45,stream_cap=1<<20,
                   cleanup_seconds=30,cleanup_state=cleanup)
    finally:reclaim(host.engine,container,target,cleanup)
    return json.loads((target/'result.json').read_text())


def internal(args):
    previous,session,children,base,study,runtime=context(args)
    from oeis_learn.pipeline_validation.identity import confined
    output=confined(ROOT,args.output,exists=False);output.mkdir(parents=True,exist_ok=False)
    host=host_for(args.engine)
    if args.internal=='compatibility':
        from oeis_learn.pipeline_validation.generalization_commands import _qualification
        from types import SimpleNamespace
        result=_qualification(host,ROOT,output,base,study,runtime,
                              SimpleNamespace(qualification=str(previous/'qualify')))
        return dict(status='PASS',original=result['original'])
    if args.internal in ('native','resume'):
        return launch_probe(host,runtime['images'][1],args.internal,output)
    import inspect
    from oeis_learn.pipeline_validation import generalization_roles as roles
    from oeis_learn.pipeline_validation.generalization_commands import _validate
    from oeis_learn.pipeline_validation.generalization_workflow import evaluate,_checkpoints
    original=roles.ROLE_PROGRAM
    # Instrument only this diagnostic subprocess. Installed sources stay untouched.
    roles.ROLE_PROGRAM=inspect.getsource(install_trace)+'\ninstall_trace()\n'+original
    host.select_gpu()
    case=output/'retained-eval'
    try:
        execution=_validate(host,ROOT,case,base,study,runtime,seconds=60)
        return evaluate(host,ROOT,case/'case/private/evaluation',base,study,runtime,execution,
                        checkpoints=[c['identity'] for c,_ in _checkpoints(base,runtime['evaluation']['checkpoints'])],
                        publish_selection=False)
    finally:roles.ROLE_PROGRAM=original


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=['docker','podman'],required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--previous',required=True,help='original session with qualification and prepared inputs')
    parser.add_argument('--output',required=True)
    parser.add_argument('--probes',nargs='*',choices=list(PROBES),default=list(PROBES))
    parser.add_argument('--cases',nargs='*',choices=list(CASES),default=list(CASES))
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--internal',choices=['compatibility',*PROBES],help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
    from oeis_learn.pipeline_validation.identity import confined,file_hash
    if args.internal:
        output=confined(ROOT,args.output,exists=False)
        try:
            result=internal(args)
        except Exception as exc:
            result=dict(status='FAIL',reason=type(exc).__name__,error=str(exc)[:2048])
        save(output/'result.json',result)
        return 0 if result['status']=='PASS' else 1
    previous,session,children,*_=context(args)
    output=confined(ROOT,args.output,exists=False)
    if output.exists() or output.with_suffix('.zip').exists():
        raise ValueError('fresh diagnostic directory and archive required')
    if not args.probes and not args.cases:
        raise ValueError('select at least one probe or case')
    stages=plan(args,children['refresh'][2]['images'][0])
    if args.dry_run:
        print(json.dumps(dict(stages=stages,ceiling_seconds=sum(s['seconds'] for s in stages),
                             feature_qualified=False),indent=2));return 0
    output.mkdir(parents=True)
    metadata=dict(helper_sha256=file_hash(Path(__file__)),tests_sha256=file_hash(Path(__file__).with_name('test_diagnose_hip.py')),
                  session=session['identity'],previous=str(previous.relative_to(ROOT)),
                  images=children['refresh'][2]['images'])
    from oeis_learn.generalization.resources import CAPS
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope
    with scope(output,CAPS):
        record=execute(stages,output,runner=run,metadata=metadata)
    archive=export(output)
    print(json.dumps(dict(status=record['status'],archive=str(archive.relative_to(ROOT)),feature_qualified=False),indent=2))
    return 0 if record['status']=='PASS' else 1


if __name__=='__main__':
    if sys.argv[1:2]==['--container-probe']:
        try:
            value=container_probe(sys.argv[2])
        except Exception as exc:
            value=dict(status='FAIL',reason=type(exc).__name__,error=str(exc)[:2048])
        save(Path('/output/result.json'),value)
        raise SystemExit(0 if value['status']=='PASS' else 1)
    try:
        raise SystemExit(main())
    except (ValueError,OSError) as exc:
        print(str(exc),file=sys.stderr);raise SystemExit(2)
