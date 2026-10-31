"""Serial production workflow adapters over frozen studies and isolated roles."""
import json
import time
from pathlib import Path
from .identity import seal,write,digest,confined
from .generalization_preparation import reference
from .generalization_roles import project_and_invoke
from oeis_learn.generalization.contracts import verify_artifact


def _status(study,values,*,incomplete=False):
    from oeis_learn.generalization.recovery import Workflow
    required=tuple(str(i) for i in range(len(values)+int(incomplete)))
    if not required:return 'PARTIAL'
    return Workflow(study,required,tuple((str(i),value) for i,value in enumerate(values))).status


def _merge(base,refs,path):
    rows=[];seen=set()
    for ref in refs:
        for row in json.loads(verify_artifact(base,ref)):
            if row['id'] in seen:raise ValueError('duplicate merged role row')
            seen.add(row['id']);rows.append(row)
    write(path,rows)
    return reference(path,path.parent)


def _checkpoints(base,refs):
    result=[];seen=set()
    for ref in refs:
        value=json.loads(verify_artifact(base,ref,max_bytes=1<<20))
        candidates=value['checkpoints'] if value.get('version')=='generalization-retained/v1' else [value]
        for checkpoint in candidates:
            if checkpoint.get('version')=='generalization-checkpoint/v2':
                from oeis_learn.generalization.legacy import verify_legacy
                verify_legacy(confined(base,ref['path']).parent,checkpoint)
            if checkpoint['identity'] in seen:raise ValueError('duplicate checkpoint in evaluation roster')
            seen.add(checkpoint['identity'])
            result.append((checkpoint,confined(base,ref['path']).parent))
    if not 1<=len(result)<=66:raise ValueError('bounded nonempty checkpoint roster required')
    return result


def evaluate(host,workspace,output,base,study,runtime,plan,*,checkpoints,bank=None):
    from .generalization_commands import _copy
    from oeis_learn.fresh.bigint_state import expose_final,finish_final,final_plan
    view=runtime['evaluation']
    if view is None:raise ValueError('frozen evaluation view required')
    available=_checkpoints(base,view['checkpoints']);by_id={c['identity']:(c,r) for c,r in available}
    for descriptor,_ in available:
        if descriptor.get('kind')=='weights_only' and descriptor['compatibility']!=digest(plan['recipe']):
            raise ValueError('historical checkpoint requires its original resolved recipe')
    if not checkpoints or len(set(checkpoints))!=len(checkpoints) or set(checkpoints)-set(by_id):
        raise ValueError('explicit frozen checkpoint subset required')
    if bank is not None and bank!=digest(view['bank_seeds']):raise ValueError('undeclared proposal-bank seed identity')
    seconds=view['seconds']
    if type(seconds) not in (int,float) or not 0<seconds<=21600:raise ValueError('bounded evaluation scope required')
    output=Path(output)
    if output.exists():raise ValueError('fresh evaluation output required')
    output.mkdir(parents=True)
    prompts=output/'private/prompts';truth=output/'private/truth';inputs=output/'private/candidates'
    prompts.mkdir(parents=True);truth.mkdir(parents=True);inputs.mkdir(parents=True)
    prompt_ref=_merge(base,view['prompts'],prompts/'prompts.json')
    truth_ref=_merge(base,view['targets'],truth/'truth.json')
    allocation=seconds/(2*len(checkpoints));deadline=time.monotonic()+seconds
    candidates=[];directories={};runs=[];missing=False;incomplete=False
    for key in checkpoints:
        if time.monotonic()>=deadline:
            missing=True;runs.append(dict(checkpoint=key,status='PARTIAL',reason='BUDGET_EXHAUSTED'));continue
        descriptor,weights_root=by_id[key]
        weights=descriptor['weights'];source=confined(weights_root,weights['path'],exists=False)
        if not source.is_file():
            missing=True;runs.append(dict(checkpoint=key,status='PARTIAL',reason='MISSING_ARTIFACT'));continue
        verify_artifact(weights_root,weights)
        named=key[7:];folder=output/named;assessment=None
        if view['role']=='assessment':
            if runtime['assessment'] is None:raise ValueError('frozen final assessment plan required')
            plan_record=json.loads(verify_artifact(base,runtime['assessment'],max_bytes=1<<20));frozen=final_plan(plan_record)
            matches=[a for a in frozen.arms if a[2]==key and a[3]==runtime['recipe']['sha256'] and a[4]==digest(study['inference']) and a[5]==digest(runtime['images'])]
            if len(matches)!=1:raise ValueError('checkpoint/recipe/portfolio/runtime absent from frozen assessment')
            arm=matches[0]
            ledger=Path(workspace)/'run-state/generalization/exposure'/frozen.cohort[7:]
            name,exposure=expose_final(ledger,plan_record,cohort=truth_ref['sha256'],checkpoint=key,
                recipe=arm[3],portfolio=arm[4],runtime=arm[5],seed=arm[1],debit=2*allocation)
            assessment=(ledger,plan_record,name,exposure)
        try:
            result=project_and_invoke(host,runtime['images'][1],workspace,'generator',
                {'visible_prompts':[prompt_ref],'frozen_weights':[weights]},
                source_roots={'visible_prompts':prompts,'frozen_weights':weights_root},
                output=str((folder/'generation').relative_to(workspace)),operation='evaluate',
                settings=dict(checkpoint=weights,prompts=prompt_ref,recipe=plan['recipe'],study=study['identity'],source=key,
                    device='cuda',bank_seeds=view['bank_seeds'] if bank is not None else None),seconds=min(allocation,deadline-time.monotonic()),gpu=True)
            generation=folder/'generation/role/generation'
            incomplete=incomplete or result['status']!='PASS'
            if result['status'] not in {'PASS','PARTIAL'} or not (generation/'selections.json').is_file():
                runs.append(dict(checkpoint=key,result=result));missing=True;continue
            selection_ref=reference(generation/'selections.json',generation)
            if view['role']=='assessment':
                if time.monotonic()>=deadline:missing=True;continue
                authorization_root=folder/'private/authorization';authorization_root.mkdir(parents=True)
                ledger,plan_record,name,exposure=assessment
                authorization=seal(dict(version='generalization-assessment-authorization/v1',plan=plan_record,exposure=exposure,
                    arm=name,targets=truth_ref,candidates=selection_ref,checkpoint=key))
                write(authorization_root/'authorization.json',authorization)
                scored=project_and_invoke(host,runtime['images'][0],workspace,'assessment',
                    {'assessment_truth':[truth_ref],'frozen_assessment_authorization':[reference(authorization_root/'authorization.json',authorization_root)],'frozen_assessment_candidates':[selection_ref]},
                    source_roots={'assessment_truth':truth,'frozen_assessment_authorization':authorization_root,'frozen_assessment_candidates':generation},
                    output=str((folder/'assessment').relative_to(workspace)),operation='score',
                    settings=dict(study=study['identity'],authorization=reference(authorization_root/'authorization.json',authorization_root)),seconds=min(allocation,deadline-time.monotonic()))
                runs.append(dict(checkpoint=key,generation=result,scoring=scored))
                if scored['status']!='PASS':missing=True
                finish_final(ledger,plan_record,name,'completed' if scored['status']=='PASS' else 'failed');assessment=None
            else:
                # Directory projections never change checkpoint identity or the
                # stable-identity tie breaker used by the selection policy.
                destination=inputs/named;_copy(destination,[(weights_root,weights),(generation,selection_ref)])
                candidates.append(dict(checkpoint=descriptor,selections=selection_ref))
                directories[key]=named
                runs.append(dict(checkpoint=key,generation=result))
        finally:
            if assessment is not None:finish_final(assessment[0],assessment[1],assessment[2],'interrupted')
    decision=None
    if view['role']=='selection' and not missing:
        remaining=deadline-time.monotonic()
        if remaining<=0:missing=True
        else:
            refs=[dict(item,path=directories[c['checkpoint']['identity']]+'/'+item['path']) for c in candidates for item in (c['checkpoint']['weights'],c['selections'])]
            decision=project_and_invoke(host,runtime['images'][0],workspace,'selection',
                {'selection_supervision':[truth_ref],'selection_candidates':refs},
                source_roots={'selection_supervision':truth,'selection_candidates':inputs},
                output=str((output/'selection').relative_to(workspace)),operation='select-checkpoints',
                settings=dict(targets=truth_ref,candidates=candidates,directories=directories,study=study['identity'],recipe=plan['recipe']),seconds=remaining/2)
            missing=decision['status']!='PASS'
    publication=None;pins=[]
    if decision is not None and time.monotonic()<deadline:
        from .generalization_selection_io import publish_selected,persist_selected
        publication=publish_selected(host,workspace,output/'initializer',base,study,runtime,decision,available,seconds=min(60,(deadline-time.monotonic())/2))
        if publication['status']=='FAIL':missing=True
        if time.monotonic()<deadline:
            pins=persist_selected(host,workspace,output/'pins',runtime,decision,available,seconds=deadline-time.monotonic())
            if any(p['status'] not in {'PASS','UNAVAILABLE'} for p in pins):missing=True
    statuses=[]
    for run in runs:
        if 'status' in run:statuses.append(run['status'])
        statuses.extend(run[k]['status'] for k in ('result','generation','scoring') if k in run)
    statuses.extend(value['status'] for value in [decision,publication,*pins] if value is not None and value.get('status')!='UNAVAILABLE')
    result=seal(dict(status=_status(study['identity'],statuses,incomplete=missing or incomplete),runs=runs,selection=decision,initializer=publication,pins=pins,checkpoint_directories=directories,
        scientific_status='UNMEASURED',role=view['role']))
    write(output/'evaluation.json',result)
    return result


def replay(host,workspace,output,base,study,runtime,plan,*,acceptance_resume=False):
    from .generalization_commands import train
    from .generalization_replay import search_isolated
    if study['progression']['stage']!='P4' or plan['initializer'] is None:raise ValueError('frozen literal P4 initializer required')
    settings=runtime['replay']
    if not isinstance(settings,dict) or set(settings)!={'tasks','seconds'}:raise ValueError('closed one-round replay settings required')
    tasks=json.loads(verify_artifact(base,settings['tasks'],max_bytes=8<<30))
    if not 1<=len(tasks)<=256:raise ValueError('bounded nonempty replay task roster required')
    output=Path(output)
    if output.exists():raise ValueError('fresh replay output required')
    output.mkdir(parents=True)
    runs=[]
    for seed in study['seeds']:
        folder=output/str(seed)
        pool,search=search_isolated(host,workspace,folder/'search',images=runtime['images'],weights_root=base,
            checkpoint=plan['initializer']['weights'],tasks=tasks,recipe=plan['recipe'],study=study['identity'],seed=seed,seconds=settings['seconds'])
        pool_root=folder/'search/private';pool_ref=reference(pool_root/'pool.json',pool_root)
        paired=train(host,workspace,folder/'training',base,study,runtime,plan,arms=['continued_sft','verified_replay'],seeds=[seed],
            replay=(pool_root,pool_ref,settings['tasks']))
        # Each current policy is evaluated from its own new weights; the discovery
        # union is never substituted for either model's evaluation outcome.
        evaluations=[]
        if paired['status']=='PASS' and runtime['evaluation'] is not None:
            for run in paired['runs']:
                learner=folder/'training'/run['root']
                evaluation_runtime=dict(runtime,evaluation=dict(runtime['evaluation'],checkpoints=[reference(learner/'checkpoints/retained.json',workspace)]))
                # Data references retain their original root; copy only declared
                # evaluation inputs into a new merged workspace-relative view.
                evaluation_base=folder/run['arm']/'private/input'
                from .generalization_commands import _copy
                data_refs=[*runtime['evaluation']['prompts'],*runtime['evaluation']['targets']]
                _copy(evaluation_base,[(base,r) for r in data_refs])
                roster=json.loads((learner/'checkpoints/retained.json').read_text())
                latest=next(c for c in roster['checkpoints'] if c['identity']==roster['latest'])
                _copy(evaluation_base,[(learner/'checkpoints',latest['weights'])])
                write(evaluation_base/'checkpoint.json',latest)
                evaluation_runtime['evaluation']['checkpoints']=[reference(evaluation_base/'checkpoint.json',evaluation_base)]
                measured=evaluate(host,workspace,folder/run['arm']/'evaluation',evaluation_base,study,evaluation_runtime,plan,checkpoints=[latest['identity']])
                evaluations.append(dict(arm=run['arm'],result=measured))
        resume_proof=None;paired_proof=None
        if acceptance_resume and paired['status']=='PASS':
            from .generalization_acceptance import training_proof
            paired_proof=training_proof(host,workspace,folder/'paired-proof',runtime,study,
                [(folder/'training'/run['root'],run['result']) for run in paired['runs']])
            if paired_proof['status']!='PASS':raise ValueError('finite update proof required for both paired arms')
            extra=dict(arms=['verified_replay'],seeds=[seed],replay=(pool_root,pool_ref,settings['tasks']))
            paused=train(host,workspace,folder/'paused',base,study,runtime,plan,stop_after_updates=1,**extra)
            if len(paused['runs'])!=1 or paused['runs'][0]['result'].get('updates')!=1:raise ValueError('acceptance interruption boundary not reached')
            prior=folder/'paused'/paused['runs'][0]['root']
            pointer=prior/'checkpoints'/paused['runs'][0]['result']['checkpoint']['path']
            continued=train(host,workspace,folder/'resumed',base,study,runtime,plan,resume=str(pointer.relative_to(workspace)),**extra)
            if continued['status']!='PASS':raise ValueError('acceptance resume incomplete')
            original=paired['runs'][1];restored=continued['runs'][0]
            resume_proof=training_proof(host,workspace,folder/'resume-proof',runtime,study,
                [(folder/'training'/original['root'],original['result']),(folder/'resumed'/restored['root'],restored['result'])],compare=True)
        runs.append(dict(seed=seed,search=search,paired=paired,paired_proof=paired_proof,current_policies=evaluations,cumulative_discoveries=len(pool.entries),resume=resume_proof))
        if paired['status']!='PASS':break
    statuses=[value for r in runs for value in [r['paired']['status'],r['search']['status'],*[e['result']['status'] for e in r['current_policies']],*([r['resume']['status']] if r['resume'] else [])]]
    status=_status(study['identity'],statuses,incomplete=len(runs)!=len(study['seeds']))
    result=seal(dict(status=status,rounds=1,runs=runs,scientific_status='UNMEASURED'))
    write(output/'replay.json',result)
    return result
