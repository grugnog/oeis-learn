"""Portable CPU-only host and container boundary for the temporary throughput gate."""
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2] if len(HERE.parents)>2 else None
HELPERS=('fresh_pilot_throughput.py','fresh_pilot_throughput_host.py','fresh_pilot_inputs.py','fresh_pilot.py')


def host_for(engine):
    from foundation_local import Host,parser
    return Host(parser().parse_args(['--engine',engine,'--memory-gib','64','--cpus','4','status','--run','run-state/throughput-unused']))


def binding(host,image):
    import platform
    from oeis_learn.pipeline_validation.identity import digest,file_hash
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from fresh_pilot_throughput import POLICY,RESOURCES
    return dict(policy=POLICY,image=host.engine.image(image),source=digest(collect_snapshot(ROOT)),
        helpers={name:file_hash(HERE/name) for name in HELPERS},resources=RESOURCES,
        engine_facts=host.engine.facts(),operator_host=digest([platform.node(),platform.machine(),platform.system()]))


def bounded(path):
    from oeis_learn.pipeline_validation.identity import seal
    if path.is_symlink() or not path.is_file() or path.stat().st_size>1<<20:raise ValueError('bounded original record required')
    value=json.loads(path.read_bytes())
    if seal(value)!=value:raise ValueError('record identity mismatch')
    return value


def require_receipt(path,expected,*,now=None):
    from oeis_learn.generalization.contracts import verify_artifact
    from fresh_pilot_throughput import assess
    root=Path(path);record=bounded(root/'result.json')
    if record.get('version')!='fresh-pilot-throughput-receipt/v1' or record.get('status')!='PASS' or record.get('binding')!=expected:
        raise ValueError('compatible original PASS throughput receipt required')
    stamp=record.get('recorded_at')
    if type(stamp) not in (int,float) or not 0<=(time.time() if now is None else now)-stamp<=72*3600:
        raise ValueError('throughput receipt expired or from future')
    report=json.loads(verify_artifact(root,record['metrics'],max_bytes=1<<20))
    import re
    if not isinstance(report.get('protocol'),str) or not re.fullmatch(r'sha256:[a-f0-9]{64}',report['protocol']):
        raise ValueError('frozen protocol identity required')
    verdict=assess(report)
    process=json.loads(verify_artifact(root,record['process'],max_bytes=1<<20))
    cleanup=json.loads(verify_artifact(root,record['cleanup'],max_bytes=1<<20))
    if (process.get('outcome')!='COMPLETE' or process.get('returncode')!=0 or process.get('cleanup_failure') or
        cleanup.get('acknowledged') is not True or cleanup.get('within_bound') is not True):
        raise ValueError('successful native process and acknowledged cleanup required')
    if verdict['status']!='PASS' or verdict!=record['verdict']:raise ValueError('throughput headroom gate failed')
    return record,report


def require_for_run(args):
    from oeis_learn.pipeline_validation.identity import confined
    path=getattr(args,'throughput',None)
    if not path or not getattr(args,'cpu_image',None):
        raise ValueError('run the CPU-only throughput command first; supply --throughput PATH and --cpu-image')
    host=host_for(args.engine)
    return require_receipt(confined(ROOT,path),binding(host,args.cpu_image))


def inside(job_path,output):
    from oeis_learn.pipeline_validation.identity import digest,file_hash
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from fresh_pilot_inputs import research_inputs
    from fresh_pilot_throughput import POLICY,measure,save
    began=time.monotonic();job=bounded(Path(job_path));root=Path(output);phase='source-binding'
    try:
        if digest(source_snapshot())!=job['binding']['source']:raise ValueError('installed source differs from checkout')
        phase='helper-binding'
        if {n:file_hash(HERE/n) for n in HELPERS}!=job['binding']['helpers']:raise ValueError('helper projection changed')
        phase='design';design=research_inputs()
        seconds=job['work_seconds']-(time.monotonic()-began)
        phase='measurement';result=measure(design,root/'measurement',seconds=seconds)
    except (Exception,KeyboardInterrupt) as exc:
        result=dict(version=POLICY,status='INCOMPLETE',phase=phase,reason=type(exc).__name__)
        save(root/'measurement/metrics.json',result)
    return 0  # host maps the explicit measured verdict; exit alone never grants PASS


def export(root):
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.bundle import pack,verify
    from oeis_learn.pipeline_validation.generalization_diagnostics import control_fields
    destination=root/'report';destination.mkdir()
    record=bounded(root/'result.json');write(destination/'result.json',record)
    write(destination/'summary.json',dict(selected_scope=record['status'],feature_qualified=False,
        scientific_status='UNMEASURED',scope='CPU/native throughput diagnostic only'))
    metrics=root/'probe/measurement/metrics.json'
    omissions=[]
    if metrics.exists():
        try:write(destination/'metrics.json',bounded(metrics))
        except (ValueError,OSError,TypeError):omissions.append(dict(artifact='metrics',reason='INVALID_OR_UNREADABLE'))
    for name in ('process.json','cleanup.json'):
        path=root/'probe/logs'/name
        if path.exists():
            try:
                if path.is_symlink() or path.stat().st_size>1<<20:raise ValueError('control cap')
                write(destination/name,control_fields(json.loads(path.read_bytes())))
            except (ValueError,OSError,TypeError):omissions.append(dict(artifact=name,reason='INVALID_OR_UNREADABLE'))
    for channel in ('stdout','stderr'):
        path=root/'probe/logs'/(channel+'.log')
        if path.is_file() and not path.is_symlink():
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-65536))
                (destination/(channel+'.log')).write_bytes(stream.read(65536))
    write(destination/'omissions.json',omissions)
    archive=pack(destination,mandatory=['result.json','summary.json'],cap=8<<20,optional_cap=4<<20)
    verify(archive);return archive


def run(engine,image,output):
    import shutil,uuid
    from oeis_learn.pipeline_validation.identity import confined,write,seal
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.supervision import launch,reclaim
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope,guard
    from oeis_learn.container_runtime import private_runtime_environment
    from fresh_pilot_throughput import SECONDS,RESOURCES,assess,save
    root=confined(ROOT,output,exists=False)
    if root.exists():raise ValueError('fresh throughput output required')
    root.mkdir(parents=True);deadline=time.monotonic()+SECONDS
    record=dict(version='fresh-pilot-throughput-receipt/v1',status='FAIL',recorded_at=time.time(),
        scientific_status='UNMEASURED',feature_qualified=False)
    target=root/'probe';target.mkdir();helper=root/'private/helpers';helper.mkdir(parents=True)
    try:
        host=host_for(engine);record['binding']=binding(host,image)
        for name in HELPERS:shutil.copyfile(HERE/name,helper/name)
        work=deadline-time.monotonic()-90
        if work<=0:raise TimeoutError('throughput setup exhausted allowance')
        write(helper/'job.json',seal(dict(binding=record['binding'],work_seconds=work)))
        environment=private_runtime_environment(target,destination='/output')
        environment.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1')
        name='generalization-throughput-'+uuid.uuid4().hex
        argv=['python','-u','/diagnostic/fresh_pilot_throughput_host.py','--inside']
        command=host.engine.command(image=record['binding']['image'],name=name,
            mounts=[(helper,'/diagnostic',False),(target,'/output',True)],argv=argv,gpu=False,
            memory_bytes=RESOURCES['memory_bytes'],cpus=RESOURCES['cpus'],network=False,tmp_bytes=2<<30,environment=environment)
        cleanup=dict(deadline=deadline,allowance_seconds=30)
        limits=dict(study_bytes=8<<30,checkpoint_bytes=0,shard_bytes=4<<30,raw_bytes=4<<20)
        try:
            with scope(root,limits),guard(target):
                launch(command,target/'logs',seconds=deadline-time.monotonic()-30,stream_cap=1<<20,
                    cleanup_seconds=30,cleanup_state=cleanup)
        finally:reclaim(host.engine,name,target/'logs',cleanup)
        report=bounded(target/'measurement/metrics.json');record['verdict']=assess(report)
        record['metrics']=reference(target/'measurement/metrics.json',root)
        record['process']=reference(target/'logs/process.json',root)
        record['cleanup']=reference(target/'logs/cleanup.json',root)
        record['status']=record['verdict']['status']
        # Re-read all original bytes and containment records before permitting PASS.
        save(root/'result.json',record)
        if record['status']=='PASS':require_receipt(root,record['binding'])
    except (Exception,KeyboardInterrupt) as exc:
        record['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL'
        record['reason']=type(exc).__name__
    save(root/'result.json',record)
    archive=export(root)
    print(json.dumps(dict(status=record['status'],archive=str(archive.relative_to(ROOT)),
        verdict=record.get('verdict'),scientific_status='UNMEASURED'),indent=2),flush=True)
    return 0 if record['status']=='PASS' else 3 if record['status']=='BLOCKED' else 1


if __name__=='__main__':
    if sys.argv[1:]!=['--inside']:raise SystemExit('use fresh_pilot.py throughput on the host')
    raise SystemExit(inside('/diagnostic/job.json','/output'))
