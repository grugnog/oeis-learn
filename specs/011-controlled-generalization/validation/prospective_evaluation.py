#!/usr/bin/env python3
"""Temporary one-command current-source qualification and retained evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2] if len(HERE.parents)>2 else None
if ROOT is not None:sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(HERE)]
from oeis_learn.pipeline_validation.identity import confined,seal,write,digest,file_hash
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact
from prospective_evaluation_policy import design,lineage,prepare,evaluation_identity,summarize

HELPERS=('prospective_evaluation.py','prospective_evaluation_policy.py')
CAMPAIGN_SECONDS=30*3600
REPORT_SECONDS=3600
LIMITS={'prepare':14400,'preflight':1800,'generate-static':23400,'generate-refreshed':23400,
        'score-static':1800,'score-refreshed':1800}


def read(path,cap=16<<20):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>cap:raise ValueError('bounded regular JSON required')
    return json.loads(path.read_bytes())


def save(path,value):
    pending=path.with_suffix('.next');write(pending,seal(value),exclusive=False);pending.replace(path)


def host_for(engine):
    from foundation_local import Host,parser
    return Host(parser().parse_args(['--engine',engine,'--memory-gib','64','--cpus','4','status','--run','run-state/unused']))


def inside(job_path,output):
    """CPU-only custodian, with sixteen explicit immutable exclusion files."""
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.pipeline_validation.generalization_role import native_batch
    from oeis_learn.synthesis.client import SynthesisClient
    job=read(Path(job_path));root=Path(output);started=time.monotonic()
    try:
        if (seal(job)!=job or digest(source_snapshot())!=job['source'] or
            {n:file_hash(HERE/n) for n in HELPERS}!=job['helpers']):raise ValueError('installed source/helper binding changed')
        protocol=job['design']
        if seal(protocol)!=protocol:raise ValueError('frozen design changed')
        deadline=started+job['seconds']
        originals=[dict(item,root=Path('/exclusions')) for item in job['originals']]
        with SynthesisClient(protocol['identity'],cache_root=root/'private/native-cache') as native:
            def execute(*_):raise AssertionError('native batch required')
            execute.batch=lambda rows:native_batch(native,rows,deadline=deadline)
            report=prepare(protocol,originals,root/'cohort',execute=execute,
                seconds=min(14000,deadline-time.monotonic()),workers=3)
        write(root/'result.json',seal(dict(status='PASS',summary=reference(root/'cohort/summary.json',root),design=protocol['identity'])))
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'result.json',seal(dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',
            reason=type(exc).__name__,error=str(exc)[:1024])))
        return 1


def native_prepare(engine,image,context,protocol,target):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.container_runtime import private_runtime_environment
    from oeis_learn.pipeline_validation.supervision import launch,reclaim
    from oeis_learn.pipeline_validation.generalization_lifecycle import guard
    deadline=time.monotonic()+LIMITS['prepare']
    host=host_for(engine);target.mkdir(parents=True);inputs=target.parent/'private/cohort-inputs';helpers=inputs/'helpers';helpers.mkdir(parents=True)
    for name in HELPERS:shutil.copyfile(HERE/name,helpers/name)
    originals=[];projection=inputs/'exclusions';projection.mkdir()
    if sum(item['ref']['bytes'] for item in context['originals'])>16<<30:raise ValueError('exclusion projection storage bound')
    mounts=[(helpers,'/helper',False),(projection,'/exclusions',False),(target,'/output',True)]
    for i,item in enumerate(context['originals']):
        name=f'cell-{i}.json';p=confined(ROOT,item['ref']['path'])
        if time.monotonic()>=deadline-120:raise TimeoutError('exclusion projection deadline')
        shutil.copyfile(p,projection/name)
        originals.append(dict(item,ref=dict(item['ref'],path=name)))
    write(helpers/'job.json',seal(dict(design=protocol,source=digest(collect_snapshot(ROOT)),
        helpers={n:file_hash(helpers/n) for n in HELPERS},originals=originals,seconds=min(14000,deadline-time.monotonic()-120))))
    environment=private_runtime_environment(target,destination='/output')
    environment.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1')
    name='prospective-cohort-'+uuid.uuid4().hex
    command=host.engine.command(image=image,name=name,mounts=mounts,
        argv=['python','-u','/helper/prospective_evaluation.py','inside','--job','/helper/job.json','--output','/output'],
        gpu=False,memory_bytes=64<<30,cpus=4,network=False,tmp_bytes=2<<30,environment=environment)
    cleanup=dict(deadline=deadline,allowance_seconds=30)
    try:
        with guard(target):launch(command,target/'logs',seconds=deadline-time.monotonic()-30,
            stream_cap=1<<20,cleanup_seconds=30,cleanup_state=cleanup)
    finally:reclaim(host.engine,name,target/'logs',cleanup)
    result=read(target/'result.json');process=read(target/'logs/process.json');reclaimed=read(target/'logs/cleanup.json')
    if (seal(result)!=result or result['status']!='PASS' or process.get('outcome')!='COMPLETE' or process.get('returncode')!=0
        or process.get('cleanup_failure') or not reclaimed.get('acknowledged') or not reclaimed.get('within_bound')):
        raise ValueError('native preparation or containment failed')
    summary=json.loads(verify_artifact(target,result['summary']))
    for ref in summary['files']:verify_artifact(target/'cohort',ref,max_bytes=64<<20)
    if seal(summary)!=summary or summary['design']!=protocol['identity'] or summary['status']!='PASS':raise ValueError('native cohort binding differs')
    return summary


def publish_inputs(root,context,protocol,qualified,authority,summary):
    """Fresh E0 authority and role-isolated projections, frozen before inference."""
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from prospective_evaluation_policy import roster
    research=root/'research';research.mkdir();cohort=root/'prepare/cohort'
    for ref in summary['files']:verify_artifact(cohort,ref,max_bytes=64<<20)
    prompts=[];targets=[];selected=[]
    for stratum in ('parameter','structural'):
        cell=cohort/'cells'/stratum
        prompts.extend(read(cell/'prompts.json'));targets.extend(read(cell/'private/targets.json'))
        selected.extend(read(cell/'private/selected.json'))
    mapping=roster(selected,protocol)
    if mapping!=summary['roster'] or {r['id'] for r in prompts}!=set(mapping) or {r['id'] for r in targets}!=set(mapping):
        raise ValueError('native selected roster changed')
    for p,t in zip(prompts,targets):
        if p!={k:t[k][:20] if k in ('indices','values') else t[k] for k in ('id','indices','values')}:
            raise ValueError('visible projection mismatch')
    def put(path,value):
        write(research/path,value);return reference(research/path,research)
    prompt=put('visible/prompts.json',prompts);truth=put('truth/targets.json',targets)
    source=put('source.json',collect_snapshot(ROOT));recipe=put('recipe.json',context['recipe'])
    exclusions=put('exclusions.json',seal(dict(original_preparation=context['preparation'],
        inputs=context['originals'],native_summary=summary['identity'],scope='custodian exact exclusions; protected payloads omitted')))
    generator=put('design.json',protocol);ledger=put('native-cohort.json',summary)
    quotas={'family:'+c['family']:c['count'] for c in protocol['cells']}
    manifest=seal(dict(version='generalization-cohort/v1',role='selection',family_rule='fixed-descriptor/v1',
        sources=[digest(r['source']) for r in selected],variants=sorted({r['variant'] for r in selected}),
        groups=[digest(r['group']) for r in selected],quotas=quotas,
        coverage=dict(Counter(tag for r in selected for tag in r['tags'])),exclusions=exclusions,
        indices=selected[0]['indices'],public_payloads=[prompt],private_payloads=[truth],preparation_ledger=ledger))
    cohort_ref=put('cohort.json',manifest)
    selection=put('selection-roster.json',seal(dict(version='generalization-cohort-roster/v1',role='selection',cells=[cohort_ref])))
    marker=put('sealed-assessment.json',seal(dict(status='SEALED',source_study=context['study']['identity'],scope='no assessment execution or score access')))
    if set(context['checkpoints'])!={'static','refreshed'}:raise ValueError('exact frozen retained pair required')
    checkpoint_refs=[]
    for arm,c in sorted(context['checkpoints'].items()):
        checkpoint=c['checkpoint'];weights=checkpoint['weights']
        if seal(checkpoint)!=checkpoint or weights!=protocol['checkpoints'][arm]:raise ValueError('retained checkpoint choice changed')
        origin=confined(ROOT,c['weights_root']);source_path=confined(origin,weights['path'])
        if source_path.stat().st_size!=weights['bytes'] or file_hash(source_path)!=weights['sha256']:
            raise ValueError('retained weights changed before evaluation publication')
        destination=research/'checkpoints'/arm;destination.mkdir(parents=True)
        copy_path=confined(destination,weights['path'],exists=False);copy_path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_path,copy_path)
        if file_hash(copy_path)!=weights['sha256']:raise ValueError('retained weight projection changed')
        write(destination/'checkpoint.json',checkpoint);checkpoint_refs.append(reference(destination/'checkpoint.json',research))
    frozen=seal(dict(protocol,version='prospective-retained-protocol/v1',design=protocol['identity'],roster=mapping,
        prompts=[prompt],targets=[truth],checkpoint_manifests=checkpoint_refs,checkpoint_lineage=context['checkpoints'],
        original_preparation=context['preparation'],parent_manifest=context['parent_manifest'],
        native_summary=summary['identity'],source=digest(collect_snapshot(ROOT)),images=authority['images'],
        qualification=reference(qualified/'qualification.json',ROOT),
        helpers={n:file_hash(HERE/n) for n in (*HELPERS,'fresh_pilot.py','fresh_pilot_inputs.py','observable_evaluation.py','post_pilot_analysis.py')}))
    write(research/'protocol.json',frozen) # Publication precedes all generation.
    study,runtime=evaluation_identity(context['study'],context['runtime'],source=source,recipe=recipe,
        feature=reference(qualified/'feature.json',qualified),images=authority['images'],protocol=frozen,
        cohort=cohort_ref,selection=selection,marker=marker,descriptors=generator,exclusions=exclusions)
    write(research/'study.json',study);write(research/'runtime.json',runtime)
    return research,study,runtime,frozen


def execute(args,root):
    import fresh_pilot as api
    from fresh_pilot_training_probe import qualified
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope
    from oeis_learn.generalization.resources import CAPS
    began=time.monotonic();record=dict(version='prospective-evaluation-workflow/v1',status='FAIL',stages=[],
        assessment='SEALED',research_tokens=0,qualification=None)
    def persist():save(root/'workflow.json',record)
    persist()
    try:
        free=shutil.disk_usage(root).free
        if free<128<<30:raise ValueError(f'128 GiB free required before acceptance; available {free/(1<<30):.1f} GiB')
        protocol=design(read(HERE/'observable-evaluation-protocol.json'));write(root/'design.json',protocol)
        context=lineage(ROOT,args.pilot,args.prepared,protocol)
        write(root/'lineage.json',seal({k:context[k] for k in ('originals','checkpoints','preparation','parent_manifest')}))
        with scope(root,CAPS):
            qualified_root,authority=qualified(ROOT,args,root)
            record['qualification']=str(qualified_root.relative_to(ROOT));persist()
            summary=native_prepare(args.engine,authority['images'][0],context,protocol,root/'prepare')
            record['stages'].append(dict(stage='prepare',status='PASS'));persist()
            research,study,runtime,frozen=publish_inputs(root,context,protocol,qualified_root,authority,summary)
            rel=lambda p:str(p.relative_to(ROOT));manifest=rel(research/'study.json')
            protocol_ref=reference(research/'protocol.json',ROOT)
            def command(stage,argv,output):
                if time.monotonic()-began+LIMITS[stage]>CAMPAIGN_SECONDS-REPORT_SECONDS:
                    raise TimeoutError('campaign reserve exhausted')
                started=time.monotonic();code=api.invoke(argv,root/'private/logs'/stage,seconds=LIMITS[stage])
                result=read(output/'result.json') if (output/'result.json').is_file() else None
                status=api.terminal_status(code,result) if result is None or seal(result)==result else 'FAIL'
                record['stages'].append(dict(stage=stage,status=status,exit_code=code,seconds=time.monotonic()-started));persist()
                print(f'{stage}: {status}, {(time.monotonic()-started)/60:.1f} minutes',flush=True)
                return result,status
            preflight=root/'preflight'
            _,status=command('preflight',api.canonical(args.engine,authority['images'][0],'preflight',rel(preflight),
                ['--manifest',manifest]),preflight)
            if status!='PASS':raise RuntimeError('evaluation-only preflight failed')
            scores={}
            from observable_evaluation import validate_scores
            for arm in ('static','refreshed'):
                if reference(research/'protocol.json',ROOT)!=protocol_ref:raise ValueError('frozen protocol changed')
                c=context['checkpoints'][arm]
                scored,_=api.evaluate_arm(args.engine,qualified_root,research,manifest,study,protocol_ref,arm,c['checkpoint'],
                    confined(ROOT,c['weights_root']),research/'visible',research/'truth',reference(research/'visible/prompts.json',research/'visible'),
                    reference(research/'truth/targets.json',research/'truth'),context['recipe'],command)
                generated=research/('generate-'+arm)/'execution/role/generation'
                outcomes,recoveries=validate_scores(read(generated/'selections.json',256<<20),scored,
                    reference(generated/'selections.json',generated),read(generated/'proposal-bank.json',32<<20))
                if recoveries:raise ValueError('new bank metadata requires legacy repair')
                scores[arm]=outcomes
                write(root/('scoring-'+arm+'.json'),scored)
            write(root/'measurements.json',summarize(frozen['roster'],scores,frozen['controls']))
            record['status']='PASS' if all(s['status']=='PASS' for s in record['stages']) else 'PARTIAL'
    except (Exception,KeyboardInterrupt) as exc:
        record['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL'
        record['error']=type(exc).__name__+': '+str(exc)[:2048]
        print(record['error'],file=sys.stderr,flush=True)
    record['seconds']=time.monotonic()-began;persist()
    return 0 if record['status']=='PASS' else 1


def export(root):
    from oeis_learn.pipeline_validation.bundle import pack,verify
    from fresh_pilot_report import diagnostics
    from oeis_learn.pipeline_validation.generalization_qualification_io import public_closure
    out=root/'report';out.mkdir();mandatory=[];total=0
    def copy(source,name):
        nonlocal total
        if source.is_symlink() or not source.is_file():raise ValueError('regular export payload required')
        total+=source.stat().st_size
        if total>500<<20:raise ValueError('mandatory evaluation evidence exceeds sharing cap')
        target=out/name
        if target.exists():
            if file_hash(target)!=file_hash(source):raise ValueError('conflicting export path')
            total-=source.stat().st_size;return
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);mandatory.append(name)
    workflow=read(root/'workflow.json')
    scientific='UNMEASURED'
    if (root/'measurements.json').is_file() and workflow['status'] in {'PASS','PARTIAL'}:
        scientific=read(root/'measurements.json')['scientific_status']
    write(out/'summary.json',seal(dict(selected_scope=workflow['status'],scientific_status=scientific,
        feature_qualified=False,scope='prospective retained-pilot selection; qualification is separate original evidence',assessment='SEALED')))
    mandatory.append('summary.json')
    for n in ('workflow.json','design.json','lineage.json','measurements.json','scoring-static.json','scoring-refreshed.json'):
        if (root/n).is_file():copy(root/n,n)
    for n in ('protocol.json','study.json','runtime.json','native-cohort.json'):
        if (root/'research'/n).is_file():copy(root/'research'/n,'evaluation/'+n)
    for arm in ('static','refreshed'):
        generated=root/'research'/('generate-'+arm)/'execution/role/generation'
        for n in ('evaluation.json','proposal-bank.json','selections.json'):
            if (generated/n).is_file():copy(generated/n,'generation-'+arm+'/'+n)
        paths=list((generated/'attempts').rglob('*')) if (generated/'attempts').is_dir() else []
        if len(paths)>10000:raise ValueError('attempt export inventory bound')
        for p in sorted(paths):
            if p.is_file() and p.suffix in {'.json','.jsonl'}:copy(p,'generation-'+arm+'/attempts/'+str(p.relative_to(generated/'attempts')))
    if workflow.get('qualification'):
        q=confined(ROOT,workflow['qualification']);authority=read(q/'qualification.json')
        copy(q/'qualification.json','qualification/qualification.json')
        for ref in public_closure(q,authority):copy(confined(q,ref['path']),'qualification/'+ref['path'])
        from oeis_learn.generalization.feature_acceptance import feature_artifacts
        feature=read(q/'feature.json');producer=json.loads(verify_artifact(q,authority['producer']['receipt']))
        copy(q/'feature.json','qualification/feature.json')
        for ref in feature_artifacts(feature,qualification=producer['identity'],root=q):
            copy(confined(q,ref['path']),'qualification/'+ref['path'])
    for p,n in diagnostics(root,root/'diagnostics'):copy(p,'diagnostics/'+p.name)
    for n in (*HELPERS,'observable_evaluation.py'):copy(HERE/n,'helpers/'+n+'.txt')
    archive=pack(out,mandatory=mandatory,cap=512<<20,optional_cap=0);verify(archive)
    return archive


def run(args):
    import fresh_pilot as api
    root=confined(ROOT,args.output,exists=False)
    for value in (args.pilot,args.prepared,args.imported,args.qualified):
        if value:
            original=confined(ROOT,value)
            if root==original or root in original.parents or original in root.parents:raise ValueError('fresh disjoint output required')
    if root.exists() or root.with_suffix('.zip').exists():raise ValueError('fresh output and archive required')
    root.mkdir(parents=True)
    write(root/'arguments.json',vars(args))
    # One external bound covers all host work, not just inner native/model roles.
    code=api.invoke([sys.executable,str(HERE/'prospective_evaluation.py'),'work','--job',str((root/'arguments.json').relative_to(ROOT))],
        root/'private/controller',seconds=CAMPAIGN_SECONDS-REPORT_SECONDS)
    if not (root/'workflow.json').is_file():save(root/'workflow.json',dict(status='FAIL',stages=[],error='controller ended before workflow publication'))
    workflow=read(root/'workflow.json')
    if code==130:save(root/'workflow.json',dict(workflow,status='INTERRUPTED',error='outer controller interrupted'))
    elif code and workflow['status']=='PASS':save(root/'workflow.json',dict(workflow,status='FAIL',error='controller exit disagrees with PASS'))
    packaging=api.invoke([sys.executable,str(HERE/'prospective_evaluation.py'),'package','--run',args.output],
        root/'private/packaging',seconds=REPORT_SECONDS-120)
    try:
        result=read(root/'archive-result.json')
        if packaging or seal(result)!=result:raise ValueError('packaging exit or binding failed')
        archive=confined(root,result['archive']['path']);verify_artifact(root,result['archive'],max_bytes=512<<20)
    except (Exception,KeyboardInterrupt) as exc:
        # Constant-size last resort if the bounded exporter itself could not finish.
        from oeis_learn.pipeline_validation.bundle import pack,verify
        workflow=read(root/'workflow.json');save(root/'workflow.json',dict(workflow,status='FAIL',error='bounded exporter failed: '+type(exc).__name__))
        emergency=root/'emergency';emergency.mkdir()
        write(emergency/'summary.json',dict(selected_scope='FAIL',scientific_status='UNMEASURED',feature_qualified=False,
            scope='export failure; large/private evidence remains in original run directory'))
        shutil.copyfile(root/'workflow.json',emergency/'workflow.json')
        archive=pack(emergency,mandatory=['summary.json','workflow.json'],cap=8<<20,optional_cap=0);verify(archive)
    workflow=read(root/'workflow.json')
    print(json.dumps(dict(status=workflow['status'],archive=str(archive.relative_to(ROOT)),assessment='SEALED'),indent=2),flush=True)
    return 0 if workflow['status']=='PASS' else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--engine',choices=('docker','podman'),required=True)
    r.add_argument('--pilot',default='run-state/generalization/fresh-p1-hourly-20261104T093555Z')
    r.add_argument('--prepared',default='run-state/generalization/fresh-p1-v2-20261103T035534Z')
    r.add_argument('--imported',default='run-state/generalization/uniform-import')
    r.add_argument('--qualified');r.add_argument('--cpu-image');r.add_argument('--gpu-image');r.add_argument('--output',required=True)
    check=sub.add_parser('check-inputs',help='stdlib-only original input verification; no containers')
    check.add_argument('--pilot',default='run-state/generalization/fresh-p1-hourly-20261104T093555Z')
    check.add_argument('--prepared',default='run-state/generalization/fresh-p1-v2-20261103T035534Z')
    w=sub.add_parser('work');w.add_argument('--job',required=True)
    e=sub.add_parser('package');e.add_argument('--run',required=True)
    i=sub.add_parser('inside');i.add_argument('--job',required=True);i.add_argument('--output',required=True)
    a=p.parse_args(argv)
    if a.command=='check-inputs':
        protocol=design(read(HERE/'observable-evaluation-protocol.json'))
        context=lineage(ROOT,a.pilot,a.prepared,protocol)
        free=shutil.disk_usage(ROOT).free
        if free<128<<30:raise ValueError('128 GiB free required before acceptance')
        print(json.dumps(dict(status='INPUTS_VERIFIED',design=protocol['identity'],exclusion_cells=len(context['originals']),
            checkpoints={arm:c['checkpoint']['identity'] for arm,c in context['checkpoints'].items()},
            free_gib=round(free/(1<<30),1),hardware='NOT_RUN'),indent=2))
        return 0
    if a.command=='inside':return inside(a.job,a.output)
    if a.command=='package':
        import fresh_pilot as api
        root=confined(ROOT,a.run)
        try:archive=export(root)
        except (Exception,KeyboardInterrupt) as exc:
            workflow=read(root/'workflow.json');save(root/'workflow.json',dict(workflow,status='FAIL',error='export: '+type(exc).__name__))
            archive=api.recovery(root,read(root/'workflow.json'))
        write(root/'archive-result.json',seal(dict(archive=reference(archive,root))))
        return 0
    if a.command=='work':
        settings=argparse.Namespace(**read(confined(ROOT,a.job)));return execute(settings,confined(ROOT,settings.output))
    if not a.qualified and (not a.cpu_image or not a.gpu_image):p.error('current CPU/GPU image tags or compatible --qualified required')
    return run(a)


if __name__=='__main__':raise SystemExit(main())
