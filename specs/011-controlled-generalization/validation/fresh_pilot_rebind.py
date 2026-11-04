"""Temporary, explicit source transition for the reviewed full-data pilot.

Original preparation and failed learners remain immutable. This is a new fresh
study, not a resume, qualification producer or general-purpose migration tool.
"""
import json
from pathlib import Path
from oeis_learn.pipeline_validation.identity import seal,write,confined
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact
from fresh_pilot_resume import document,command_receipt
from fresh_pilot_training_policy import assess

PARENT_PROTOCOL='sha256:fac5ab091426a3ae26cd5b12a733754f75d1e1a5d6570a8d3ae21b1482084ca2'
APPROVED_PROBE='sha256:7a0d0bb036b834658253abefec662184d705a7f2a89514534f2a328898fd249c'
PARENT_PREPARATION='sha256:56955fda4ffba7eda116fbcfc26b174e3b0d381130db1185d1d8149b13073b7f'
MILESTONES=[4_000_000,8_000_000,16_000_000,25_000_000]


def prepared_evidence(root,old,engine):
    """Original native preparation supplies data provenance, not new execution authority."""
    from oeis_learn.pipeline_validation.generalization_runtime import load
    from fresh_pilot_feasibility import verify_prepared
    job=document(old/'private/smoke-job.json')
    if job['version']!='fresh-pilot-smoke-job/v1' or job['engine']!=engine:raise ValueError('original prepared job/engine required')
    prepared=old/'research/prepare';terminal,inventory=command_receipt(prepared,'prepare')
    manifest=confined(root,job['manifest'])
    if (terminal['identity']!=PARENT_PREPARATION or terminal['result']['manifest']!=job['manifest']
        or manifest!=prepared/'preparation/role/private/prepared/study.json'):
        raise ValueError('reviewed original preparation required')
    for path in (manifest,manifest.with_name('runtime.json')):
        if str(path.relative_to(prepared)) not in inventory:raise ValueError('original authority missing from inventory')
    base,study,runtime=load(root,job['manifest'])
    design=json.loads(verify_artifact(root,job['design']))
    cfg=json.loads(verify_artifact(root,job['blueprint']))
    if (seal(design)!=design or design['identity']!=PARENT_PROTOCOL or design['identity'] not in cfg['hypothesis']
        or cfg['initializer'] is not None or cfg['seeds']!=[20261016]):raise ValueError('reviewed original full design required')
    for key,ref in [('recipe',runtime['recipe']),('descriptors',study['data']['descriptors']),('exclusions',study['data']['exclusions'])]:
        original=reference(confined(root,cfg[key]),root)
        if any(original[k]!=ref[k] for k in ('bytes','sha256')):raise ValueError('original '+key+' differs')
        if str(confined(base,ref['path']).relative_to(prepared)) not in inventory:raise ValueError('input missing from original inventory')
    for cell in cfg['cells']:
        selected=confined(base,'cells/'+cell['name']+'/private/selected.json')
        if str(selected.relative_to(prepared)) not in inventory:raise ValueError('selection missing from original inventory')
    feasible=document(old/'feasibility/result.json')
    source_check=json.loads(verify_artifact(old/'feasibility',feasible['feasibility']))
    if (feasible['status']!='PASS' or source_check['protocol']!=design['identity'] or len(source_check['cells'])!=16
        or any(c['status']!='FEASIBLE' for c in source_check['cells'])):raise ValueError('full original source feasibility required')
    verify_prepared(source_check,base)
    return dict(old=old,manifest=job['manifest'],job=job,study=study,runtime=runtime,design=design,cfg=cfg,
        source_check=source_check,preparation=terminal['identity'])


def amend(study,runtime,design,source,images,feature):
    """Create new identities; never reseal old evidence as a new producer."""
    from oeis_learn.generalization.execution import bind,verify_binding,validate_policy
    verify_binding(study,runtime);validate_policy(study,runtime)
    if seal(design)!=design or design['identity']!=PARENT_PROTOCOL:raise ValueError('reviewed original full protocol required')
    if (runtime['preparation'] is not None or runtime['purpose']!='research' or runtime['evaluation'] is not None
        or runtime['replay'] is not None or runtime['assessment'] is not None or study['model']['initialization']!='random'
        or study['progression']['stage']!='P1-screen' or study['seeds']!=[20261016]
        or study['budgets']['target_tokens']!=25_000_000 or study['budgets']['seconds']!=21600
        or study['checkpoints']['milestones']!=MILESTONES or study['hypothesis'].count(design['identity'])!=1):
        raise ValueError('original fresh paired full study required')
    for arm,n in (('static',1),('refreshed',12)):
        if len(runtime['training'][arm]['shards'])!=n or runtime['training'][arm]['batch_size']!=16:
            raise ValueError('frozen full shard layout and batch required')
    protocol=seal(dict(design,checkpoint_policy='hourly/v1'))
    proposed=dict(study,source_closure=source,hypothesis=study['hypothesis'].replace(design['identity'],protocol['identity']))
    new,sidecar=bind(proposed,dict(runtime,images=images,feature=feature))
    verify_binding(new,sidecar);validate_policy(new,sidecar)
    return new,sidecar,protocol


def probe_evidence(root,folder,preparation,parent_protocol,qualified,authority):
    """Read exact reviewed probe and independently check its original producers."""
    import fresh_pilot as api
    from fresh_pilot_training_probe import verify_plan
    from fresh_pilot_report import training_timing
    from oeis_learn.pipeline_validation.generalization_runtime import load
    record=document(folder/'result.json');protocol=record['protocol']
    if (record['identity']!=APPROVED_PROBE or record['status']!='PASS'
        or record['version']!='fresh-pilot-training-probe-result/v1' or record['original_preparation']!=preparation
        or seal(protocol)!=protocol or protocol['parent_protocol']!=parent_protocol
        or protocol['checkpoint_policy']!='hourly/v1' or protocol['research_milestones']!=len(MILESTONES)):
        raise ValueError('reviewed hourly probe bound to original full preparation required')
    if (record['qualification']!=reference(qualified/'qualification.json',root) or record['images']!=authority['images']):
        raise ValueError('probe qualification/image binding changed')
    names=['preflight','prepare','train-static','train-refreshed']
    if [s['stage'] for s in record['stages']]!=names:raise ValueError('complete original probe stages required')
    receipts={}
    for name,stage in zip(names,record['stages']):
        output=folder/'research'/name
        if stage['status']!='PASS' or stage['exit_code']!=0 or stage['receipt']!=reference(output/'result.json',root):
            raise ValueError('probe original receipt changed')
        receipts[name]=command_receipt(output,name.split('-')[0])[0]
    base,study,runtime=load(root,receipts['prepare']['result']['manifest'])
    if runtime['preparation'] is not None or runtime['purpose']!='research' or runtime['images']!=authority['images']:
        raise ValueError('resolved probe runtime differs')
    verify_plan(base,study,runtime,protocol)
    results={};initials=[]
    for arm in ('static','refreshed'):
        output=folder/'research'/('train-'+arm);runs=receipts['train-'+arm]['result']['runs']
        if len(runs)!=1:raise ValueError('one original probe learner per arm required')
        result=runs[0]['result'];learner=confined(output/'training',runs[0]['root'])
        if seal(document(learner/'training.json',sealed=False))!=result:raise ValueError('original learner result differs')
        checkpoint,_=api.latest_checkpoint(learner,result,study=study['identity'],arm=arm)
        initials.append(checkpoint['initialization'])
        summary={k:result[k] for k in ('status','tokens','charged_tokens','discarded_tokens','checkpoint_policy','pending_tokens','updates')}
        summary['timing']=training_timing(result['timing'])
        if record['training'][arm]!=summary or result['checkpoint_policy']!='hourly/v1':raise ValueError('probe training summary differs from original')
        role=folder/('proof-'+arm)/'proof/role'
        proof=document(role/'training-proof.json')
        if proof!=document(role/'result.json') or proof['status']!='PASS' or len(proof['runs'])!=1:raise ValueError('original finite-update proof required')
        expected=dict(checkpoint=checkpoint['identity'],weights=checkpoint['weights']['sha256'],initial=checkpoint['initialization'],
            updates=result['updates'],tokens=result['tokens'],finite=True,changed=True)
        if proof['runs'][0]!=expected or record['proofs'][arm]!={'status':'PASS'}:raise ValueError('finite-update proof binding differs')
        process=document(role/'supervision/process.json',sealed=False);cleanup=document(role/'supervision/cleanup.json',sealed=False)
        if process.get('outcome')!='COMPLETE' or process.get('returncode')!=0 or not cleanup.get('acknowledged') or not cleanup.get('within_bound'):
            raise ValueError('completed proof and acknowledged cleanup required')
        results[arm]=result
    if len(set(initials))!=1 or initials[0]!=record['initialization']:raise ValueError('paired probe initializer differs')
    forecast=assess(results,protocol['first_shard_tokens'],research_milestones=protocol['research_milestones'])
    if forecast!=record['forecast'] or forecast['status']!='PASS':raise ValueError('original measurements do not support full-run forecast')
    return record
