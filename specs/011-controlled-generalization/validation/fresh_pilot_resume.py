"""Temporary, read-only prepared-run reuse; never resumes or resets a learner."""
import json
from pathlib import Path
import shutil

from oeis_learn.pipeline_validation.identity import confined,seal,file_hash
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact

REQUIRED_FREE_BYTES=160<<30  #64 GiB research +64 GiB smoke +32 GiB margin
PREDECESSOR_CONTROLLER='sha256:ff841f86855822f880933796088890667b8a6ac9856c46f4abaef92044f67c10'


def disk_preflight(root):
    free=shutil.disk_usage(root).free
    if free<REQUIRED_FREE_BYTES:
        raise ValueError(f'continuation needs at least 160 GiB free; available {free/(1<<30):.1f} GiB; preserve the original run and free space before retrying')
    return dict(free_bytes=free,required_free_bytes=REQUIRED_FREE_BYTES)


def document(path,*,sealed=True,cap=16<<20):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>cap:raise ValueError('bounded original document required')
    value=json.loads(path.read_bytes())
    if sealed and seal(value)!=value:raise ValueError('original document identity mismatch')
    return value


def command_receipt(folder,stage):
    """Verify every inventoried original byte, streaming large payloads."""
    value=document(folder/'result.json')
    if (value.get('version')!='generalization-command-result/v2' or value.get('status')!='PASS'
        or value.get('stage')!=stage or value.get('original') is not None):
        raise ValueError('original successful canonical command required')
    refs=value['inventory'];seen=set();total=0
    if not isinstance(refs,list) or not 0<len(refs)<=10000:raise ValueError('original inventory required')
    for ref in refs:
        if set(ref)!={'path','bytes','sha256'} or type(ref['bytes']) is not int or ref['bytes']<0:raise ValueError('inventory reference')
        total+=ref['bytes']
        if total>64<<30 or ref['path'] in seen:raise ValueError('bounded unique inventory required')
        seen.add(ref['path']);path=confined(folder,ref['path'])
        if not path.is_file() or path.stat().st_size!=ref['bytes'] or file_hash(path)!=ref['sha256']:
            raise ValueError('original inventory bytes changed: '+ref['path'])
    return value,seen


def prepared_inputs(workspace,previous,output,engine):
    """Verify lineage and the exact pre-learning restart boundary without writing."""
    from oeis_learn.pipeline_validation.generalization_runtime import load
    from oeis_learn.pipeline_validation.generalization_qualification_io import public_closure
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from fresh_pilot_feasibility import verify_prepared
    root=Path(workspace);old=confined(root,previous);new=confined(root,output,exists=False)
    if new==old or new in old.parents or old in new.parents or new.exists() or new.with_suffix('.zip').exists():
        raise ValueError('new disjoint continuation root required')
    space=disk_preflight(root)
    workflow=document(old/'workflow.json')
    if (workflow.get('version')!='fresh-pilot-workflow/v1' or workflow.get('status') not in {'FAIL','INTERRUPTED'}
        or workflow.get('assessment')!='SEALED' or workflow.get('acceptance',{}).get('exit_code')!=0):
        raise ValueError('original failed pre-learning workflow with successful acceptance required')
    stages=workflow['stages']
    if [s['stage'] for s in stages]!=['feasibility','preflight','prepare','smoke','controller']:
        raise ValueError('only the prepared, pre-learning smoke boundary can continue')
    if any(s['status']!='PASS' or s['exit_code']!=0 for s in stages[:3]) or stages[3]['status'] not in {'FAIL','INTERRUPTED'}:
        raise ValueError('completed preparation and failed smoke required')
    if any((old/'research'/name).exists() for name in ('train-static','train-refreshed','measurement-plan.json')):
        raise ValueError('full learning already started; explicit checkpoint resume required')
    job=document(old/'private/smoke-job.json')
    if job.get('version')!='fresh-pilot-smoke-job/v1' or job.get('engine')!=engine:raise ValueError('original smoke engine/job required')
    qualified=confined(root,job['qualified'])
    if qualified!=old/'acceptance/report/qualified':raise ValueError('original acceptance producer required')
    authority=document(qualified/'qualification.json')
    if authority.get('version')!='generalization-qualification/v2':raise ValueError('original v2 qualification required')
    public_closure(qualified,authority)
    if json.loads(verify_artifact(qualified,authority['source_closure']))!=collect_snapshot(root):
        raise ValueError('qualified production source changed; rerun acceptance')
    if authority['resources']['memory_bytes']!=64<<30 or authority['resources']['cpus']!=4:
        raise ValueError('original 64 GiB/four-CPU qualification required')
    prepared=old/'research/prepare'
    terminal,inventory=command_receipt(prepared,'prepare')
    manifest=confined(root,job['manifest'])
    if manifest!=prepared/'preparation/role/private/prepared/study.json' or terminal['result']['manifest']!=job['manifest']:
        raise ValueError('original prepared manifest required')
    for p in (manifest,manifest.with_name('runtime.json')):
        if str(p.relative_to(prepared)) not in inventory:raise ValueError('prepared authority missing from original inventory')
    base,study,runtime=load(root,job['manifest'])
    if runtime['preparation'] is not None or runtime['purpose']!='research' or runtime['images']!=authority['images']:
        raise ValueError('resolved original research and qualified image pair required')
    if json.loads(verify_artifact(base,study['source_closure']))!=collect_snapshot(root):raise ValueError('prepared source changed')
    from oeis_learn.generalization.feature_acceptance import feature_artifacts
    feature=json.loads(verify_artifact(qualified,runtime['feature'],max_bytes=1<<20))
    producer=json.loads(verify_artifact(qualified,authority['producer']['receipt']))
    feature_artifacts(feature,qualification=producer['identity'],root=qualified)
    for key,name in (('design','design.json'),('blueprint','blueprint.json')):
        if confined(root,job[key]['path'])!=old/'research/design'/name:raise ValueError('original design paths required')
    design=json.loads(verify_artifact(root,job['design']))
    cfg=json.loads(verify_artifact(root,job['blueprint']))
    feasible=document(old/'feasibility/result.json')
    source_check=json.loads(verify_artifact(old/'feasibility',feasible['feasibility']))
    if (feasible['status']!='PASS' or seal(design)!=design or design['identity'] not in study['hypothesis']
        or design['identity'] not in cfg['hypothesis'] or source_check['protocol']!=design['identity']
        or len(source_check['cells'])!=16 or any(c['status']!='PASS' for c in source_check['cells'])):
        raise ValueError('complete original frozen protocol/feasibility required')
    verify_prepared(source_check,base)
    # These original control results are reported as reused, with original timing.
    preflight,_=command_receipt(old/'research/preflight','preflight')
    for stage,path in zip(stages[:3],(old/'feasibility/result.json',old/'research/preflight/result.json',prepared/'result.json')):
        if len(stage['originals'])!=1 or Path(stage['originals'][0])!=path:raise ValueError('original stage path mismatch')
    return dict(old=old,workflow=workflow,job=job,qualified=qualified,images=authority['images'],
        manifest=job['manifest'],design=design,cfg=cfg,source_check=source_check,space=space,
        preparation=terminal['identity'],preflight=preflight['identity'])


def throughput_proof(api,context,engine):
    """Only the known controller revision may differ; all measured facts still match."""
    import fresh_pilot_throughput_host as h
    matches=[]
    parent=context['old'].parent
    paths=list(parent.glob('*/result.json'))
    if len(paths)>1024:raise ValueError('throughput search bound; keep originals in the run parent')
    for path in paths:
        try:value=document(path)
        except (ValueError,OSError):continue
        if value.get('identity')==context['workflow']['throughput'] and (path.parent/'probe/measurement/metrics.json').is_file():matches.append(path.parent)
    if len(matches)!=1:raise ValueError('one original throughput run matching the previous workflow required')
    host=h.host_for(engine);expected=h.binding(host,context['images'][0])
    original=document(matches[0]/'result.json')
    old_hash=original.get('binding',{}).get('helpers',{}).get('fresh_pilot.py')
    if old_hash!=PREDECESSOR_CONTROLLER:raise ValueError('unsupported predecessor controller')
    expected['helpers']['fresh_pilot.py']=PREDECESSOR_CONTROLLER
    receipt,report=h.require_receipt(matches[0],expected)
    if report['protocol']!=context['design']['identity']:raise ValueError('throughput protocol changed')
    if any(host.engine.image(image)!=image for image in context['images']):raise ValueError('qualified images unavailable or changed')
    return matches[0],receipt,report
