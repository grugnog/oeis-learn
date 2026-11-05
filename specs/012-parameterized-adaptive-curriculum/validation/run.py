#!/usr/bin/env python3
"""Temporary PAC serial composition; scientific authority stays in installed roles."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from oeis_learn.pipeline_validation.identity import seal,digest,write,confined,file_hash


def stage_order(*,research=False,arms=('B','C')):
    if len(set(arms))!=len(arms) or not {'B','C'}<=set(arms)<=set('ABCD'):raise ValueError('predeclared complete arms required')
    stages=['admission','qualification','calibration','acceptance']
    if research:
        stages+=['baseline']
        for arm in arms:stages+=['train-'+arm,'evaluate-'+arm]
        stages+=['analysis']
    return stages


def canonical_command(engine,operation,manifest,output,*,qualification=None,extra=()):
    if engine not in {'docker','podman'} or operation not in {'preflight','prepare','qualify','acceptance','evaluate','train'}:
        raise ValueError('registered explicit canonical command required')
    argv=[sys.executable,'scripts/foundation_run.py','--engine',engine,'generalization',operation,
          '--manifest',manifest,'--output',output]
    if operation not in {'qualify','preflight'}:
        if qualification is None:raise ValueError('original qualification required')
        argv+=['--qualification',qualification]
    return argv+list(extra)


def invoke(engine,operation,manifest,output,*,workspace=ROOT,qualification=None,extra=(),seconds,runner=None):
    from oeis_learn.pipeline_validation.supervision import launch
    argv=canonical_command(engine,operation,manifest,output,qualification=qualification,extra=extra)
    if runner is not None:return runner(argv,seconds=seconds)
    # Existing supervisor owns timeout, process-group cancellation and bounded logs.
    target=confined(workspace,output,exists=False)
    launch(argv,target.parent/(target.name+'-supervision'),seconds=seconds,cleanup_seconds=30,stream_cap=1<<20)
    path=target/'result.json'
    if not path.is_file() or path.stat().st_size>16<<20:raise ValueError('canonical terminal receipt missing')
    result=json.loads(path.read_text())
    if seal(result)!=result or result.get('stage')!=operation:raise ValueError('canonical producer binding changed')
    return result


def prepare_plan(workspace,*,engine,stages,inputs,source,images,research=False,arms=('B','C')):
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.pipeline_validation.identity import number
    if engine not in {'docker','podman'}:raise ValueError('explicit engine required')
    expected=stage_order(research=research,arms=arms)
    if [s['name'] for s in stages]!=expected:raise ValueError('complete ordered campaign required')
    seen=set()
    for stage in stages:
        if set(stage)!={'name','operation','manifest','seconds','extra','requires'}:raise ValueError('closed canonical stage required')
        number(stage['seconds'],positive=True)
        if not set(stage['requires'])<=seen:raise ValueError('acyclic earlier dependencies required')
        if seen and expected[len(seen)-1] not in stage['requires']:raise ValueError('serial admission/qualification/acceptance dependency required')
        seen.add(stage['name'])
    for ref in inputs:verify_artifact(workspace,ref,max_bytes=32<<30)
    if len(images)!=2 or any(not isinstance(v,str) or len(v)!=71 or not v.startswith('sha256:') for v in images):raise ValueError('immutable image pair required')
    from oeis_learn.pipeline_validation.generalization_pac_export import helper_identity
    return seal(dict(version='pac-serial-plan/v1',engine=engine,research=research,arms=list(arms),source=source,images=images,
        inputs=inputs,stages=stages,seconds=sum(s['seconds'] for s in stages)+3600,finalization_seconds=3600,
        helper=helper_identity(ROOT),scientific=dict(target_tokens=25000000,parameter='DEFERRED',assessment='NOT_EXECUTED',automatic_winner=False)))


def _read(path,cap=16<<20):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>cap:raise ValueError('bounded regular record required')
    return json.loads(path.read_text())


def _persist(path,record):
    pending=path.with_suffix('.next');write(pending,seal(record),exclusive=False);pending.replace(path)


def _reusable(previous,entry,binding,*,workspace=None):
    from oeis_learn.generalization.contracts import verify_artifact
    if entry.get('binding')!=binding or entry.get('status')!='PASS':return False
    if entry.get('process',{}).get('outcome')!='COMPLETE' or entry['process'].get('returncode')!=0:return False
    if not all(entry.get('cleanup',{}).get(k) is True for k in ('within_bound','acknowledged')):return False
    try:
        root=confined(previous,entry['directory'])
        for ref in entry['artifacts']:verify_artifact(root,ref,max_bytes=32<<30)
        for ref in entry.get('resolved_refs',[]):verify_artifact(workspace,ref,max_bytes=32<<30)
    except (OSError,ValueError):return False
    return True


def recover(original,output):
    """Read-only canonical evidence collection and independent archive verification."""
    from oeis_learn.pipeline_validation.generalization_pac_export import export
    return export(original,output)


def execute_plan(plan,output,*,workspace=ROOT,runner=None,previous=None):
    import time
    from oeis_learn.generalization.contracts import verify_artifact
    if seal(plan)!=plan or plan.get('version')!='pac-serial-plan/v1':raise ValueError('intact frozen plan required')
    checked=prepare_plan(workspace,engine=plan['engine'],stages=plan['stages'],inputs=plan['inputs'],source=plan['source'],images=plan['images'],research=plan['research'],arms=plan['arms'])
    if any(checked[k]!=plan[k] for k in checked if k not in {'identity','helper'}):raise ValueError('plan fields changed')
    if runner is None:
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        if digest(collect_snapshot(workspace))!=plan['source']:raise ValueError('source changed; prepare a new frozen plan')
    output=Path(output)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh campaign output required')
    prior=_read(Path(previous)/'workflow.json') if previous else None
    if prior and seal(prior)!=prior:raise ValueError('old workflow seal changed')
    output.mkdir(parents=True);write(output/'plan.json',plan)
    record=dict(version='pac-workflow/v1',plan=plan['identity'],status='PARTIAL',receipts=[],errors=[])
    start=time.monotonic();receipts={};old={e['stage']:e for e in prior['receipts']} if prior else {}
    _persist(output/'workflow.json',record)
    effective_stages=json.loads(json.dumps(plan['stages']))
    effective_seconds=plan['seconds']
    for stage in effective_stages:
        name=stage['name'];began=time.monotonic()
        binding=digest(dict(stage=stage,engine=plan['engine'],source=plan['source'],images=plan['images'],inputs=plan['inputs'],
                            parents=[receipts[n]['producer'] for n in stage['requires']]))
        existing=old.get(name)
        if existing and _reusable(Path(previous),existing,binding,workspace=workspace):
            # Copy only verified metadata/artifacts into a new owned directory;
            # the original successful receipt remains immutable.
            import shutil
            source=confined(previous,existing['directory']);target=output/name
            target.mkdir()
            for ref in existing['artifacts']:
                payload=verify_artifact(source,ref,max_bytes=32<<30)
                destination=confined(target,ref['path'],exists=False);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(payload)
            entry=dict(existing,reused=True,directory=name)
            resolved=[]
            for ref in existing.get('resolved_refs',[]):
                source=confined(workspace,ref['path'])
                if source.is_relative_to(Path(previous)):
                    destination=output/source.relative_to(Path(previous));destination.parent.mkdir(parents=True,exist_ok=True)
                    payload=verify_artifact(workspace,ref,max_bytes=32<<30)
                    if destination.exists() and destination.read_bytes()!=payload:raise ValueError('reused resolved artifact conflict')
                    if not destination.exists():destination.write_bytes(payload)
                    old_path=ref['path'];new_path=str(destination.relative_to(workspace))
                    for key in ('resolved_manifest','resolved_runtime'):
                        if entry.get(key)==old_path:entry[key]=new_path
                    resolved.append(dict(ref,path=new_path))
                else:resolved.append(ref)
            entry['resolved_refs']=resolved
        else:
            target=output/name;target.mkdir()
            print(f'{name}: starting; measured progress unavailable until canonical role output',flush=True)
            try:
                if time.monotonic()-start+stage['seconds']>effective_seconds-plan['finalization_seconds']:raise TimeoutError('protected finalization reserve')
                if runner is not None:value=runner(stage,target)
                else:
                    resume_path=None
                    if existing and existing.get('binding')==binding and name.startswith('train-'):
                        resume_path=_model_resume(plan,stage,Path(previous),workspace,receipts)
                    value=_canonical_stage(plan,stage,target,workspace,receipts,resume_path=resume_path)
                resolved=[]
                for key in ('resolved_manifest','resolved_runtime'):
                    if value.get(key):
                        from oeis_learn.pipeline_validation.generalization_preparation import reference
                        path=confined(workspace,value[key]);resolved.append(reference(path,workspace))
                        if key=='resolved_runtime':
                            runtime_value=_read(path);feature=runtime_value.get('feature')
                            if isinstance(feature,dict) and feature.get('pac'):
                                feature_root=path.parent/Path(feature['pac']['path']).parent
                                for file in feature_root.rglob('*'):
                                    if file.is_symlink():raise ValueError('symlink in acceptance closure')
                                    if file.is_file():resolved.append(reference(file,workspace))
                value['resolved_refs']=resolved
                status=value.get('status','FAIL')
                if status=='PASS' and (value.get('process',{}).get('outcome')!='COMPLETE' or value['process'].get('returncode')!=0 or not all(value.get('cleanup',{}).get(k) is True for k in ('within_bound','acknowledged'))):status='FAIL'
                for ref in value.get('artifacts',[]):verify_artifact(target,ref,max_bytes=32<<30)
                entry=dict(value,stage=name,status=status,binding=binding,directory=name,reused=False,elapsed_seconds=time.monotonic()-began)
            except BaseException as exc:
                if not isinstance(exc,(Exception,KeyboardInterrupt)):raise
                entry=dict(stage=name,status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',binding=binding,directory=name,reused=False,
                    producer=digest([name,type(exc).__name__]),artifacts=[],elapsed_seconds=time.monotonic()-began)
                record['errors'].append(dict(stage=name,kind=type(exc).__name__,status=entry['status']))
                # Capture bounded failed-stage evidence without qualifying it for reuse.
                try:
                    from oeis_learn.pipeline_validation.generalization_lifecycle import inventory
                    entry['artifacts']=inventory(target)
                    for supervision in ('supervision','execution-supervision'):
                        path=target/supervision/'process.json'
                        if path.is_file():entry['process']=_read(path,cap=64<<10);break
                    entry['cleanup']=dict(acknowledged=False,within_bound=False)
                except Exception as capture:
                    record['errors'].append(dict(stage=name,kind=type(capture).__name__,status='FAIL'))
        if name=='calibration' and entry['status']=='PASS' and entry.get('operational_envelope'):
            e=entry['operational_envelope']
            for future in effective_stages:
                if future['name'].startswith('train-'):future['seconds']=max(future['seconds'],e['learner']['outer']+e['validation']['outer']+4*1800+7200)
                elif future['name']=='baseline' or future['name'].startswith('evaluate-'):future['seconds']=max(future['seconds'],e['generation']['outer']+e['proof']['outer']+2*e['validation']['outer']+4*1800+7200)
            effective_seconds=max(effective_seconds,sum(s['seconds'] for s in effective_stages)+3600)
            revision=seal(dict(version='pac-operational-freeze/v1',original_plan=plan['identity'],calibration=entry['producer'],stages=effective_stages,seconds=effective_seconds,finalization_seconds=3600))
            write(output/'operational-freeze.json',revision)
            record['operational_freeze']=revision['identity']
            print('Frozen calibrated operational ceiling: '+str(effective_seconds)+' seconds; scientific scope unchanged',flush=True)
        receipts[name]=entry;record['receipts'].append(entry)
        if entry['status']!='PASS':
            record['status']=entry['status']
            if not record['errors']:record['errors'].append(dict(stage=name,kind='CanonicalStage',status=entry['status']))
            _persist(output/'workflow.json',record);break
        _persist(output/'workflow.json',record)
    else:record['status']='PASS'
    _persist(output/'workflow.json',record)
    try:
        archive=recover(output,output/'public-report')
        import shutil
        shutil.copyfile(archive,output/'report.zip')
        from oeis_learn.pipeline_validation.generalization_archive import verify_archive
        exported=verify_archive(output/'report.zip')
        record['export_status']=exported['software_status']
        if record['status']=='PASS' and exported['software_status']!='PASS':record['status']=exported['software_status']
        _persist(output/'workflow.json',record)
    except Exception as exc:
        record['errors'].append(dict(stage='export',kind=type(exc).__name__,status='FAIL'));record['status']='FAIL'
        _persist(output/'workflow.json',record)
        print('Last-resort evidence: '+str(output/'workflow.json'),flush=True)
    return seal(record)


def _canonical_stage(plan,stage,target,workspace,receipts):
    import time
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    relative=str((target/'execution').relative_to(workspace))
    # Dynamic resolved paths come only from sealed canonical prior-stage outputs.
    manifest=stage['manifest']
    if manifest.startswith('@'):
        parent=receipts[manifest[1:]]
        result=_read(Path(workspace)/parent['resolved_manifest'])
        manifest=parent['resolved_manifest']
    value=invoke(plan['engine'],stage['operation'],manifest,relative,workspace=workspace,
        qualification=stage.get('qualification') or str((target.parent/'qualification/execution').relative_to(workspace)),
        extra=stage['extra'],seconds=stage['seconds'])
    supervision=target/(Path(relative).name+'-supervision')
    process=_read(supervision/'process.json');cleanup=process.get('cleanup_failure') is None
    refs=[reference(target/'execution/result.json',target)]
    for ref in value['inventory']:
        refs.append(dict(ref,path='execution/'+ref['path']))
    result=dict(status=value['status'],producer=value['identity'],process=process,
        cleanup=dict(acknowledged=cleanup,within_bound=cleanup),artifacts=refs)
    if stage['operation']=='prepare' and value['status']=='PASS':
        result['resolved_manifest']=str((target/'execution/preparation/role/private/prepared/study.json').relative_to(workspace))
    return result


def _role_job(plan,operation,settings,artifacts,source_roots,target,workspace,qualification=None,native_admission=None):
    from oeis_learn.pipeline_validation.generalization_pac_setup import envelope
    from oeis_learn.pipeline_validation.supervision import launch
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    kind={'pac-assemble':'preparation','pac-campaign-acceptance':'acceptance','pac-calibrate':'calibration'}[operation]
    limits=envelope(purpose='research',arms=plan['arms'])[kind]
    job=seal(dict(version='pac-role-job/v1',operation=operation,engine=plan['engine'],images=plan['images'],source=plan['source'],
        artifacts=artifacts,source_roots={k:str(Path(v).relative_to(workspace)) for k,v in source_roots.items()},settings=settings,envelope=limits,qualification=qualification,native_admission=native_admission))
    write(target/'job.json',job)
    argv=[sys.executable,'scripts/foundation_run.py','--engine',plan['engine'],'generalization','pac-stage','--job',str((target/'job.json').relative_to(workspace)),
        '--output',str((target/'execution').relative_to(workspace))]
    launch(argv,target/'supervision',seconds=limits['outer']+9000,cleanup_seconds=30,stream_cap=1<<20)
    value=_read(target/'execution/result.json');process=_read(target/'supervision/process.json')
    if value['job']!=job['identity'] or value['source']!=plan['source'] or seal(value)!=value:raise ValueError('canonical job producer mismatch')
    result=dict(status=value['status'],producer=value['identity'],process=process,
        cleanup=dict(acknowledged=process.get('cleanup_failure') is None,within_bound=process.get('cleanup_failure') is None),
        artifacts=[reference(target/'execution/result.json',target),*[dict(r,path='execution/'+r['path']) for r in value['inventory']]])
    result['result']=value['result'];return result


def build_plan(args,workspace=ROOT):
    """Read-only original audit, explicit builds, then freeze one serial campaign."""
    import importlib.util
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.structural import design,prior_protocol
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.engines import Images
    from oeis_learn.pipeline_validation.generalization_pac_setup import envelope
    from inputs import original_chain
    workspace=Path(workspace)
    context=original_chain(workspace,prepared=args.prepared,pilot=args.pilot,protocol=design(prior_protocol()))
    path=confined(workspace,args.output,exists=False)
    if path.exists():raise ValueError('fresh plan file required')
    folder=path.parent
    folder.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(workspace/'scripts'))
    from foundation_local import Host,parser
    host=Host(parser().parse_args(['--engine',args.engine,'status','--run','run-state/unused']))
    images=Images(cap=86400)
    import shutil
    required=(128+64*(int(args.fresh_control)+int(args.bandit_only)))<<30
    if shutil.disk_usage(workspace).free<required:raise ValueError('insufficient free disk for complete selected-arm live graph: '+str(required)+' bytes required')
    pair=[images.ensure(host,target,folder/'build'/target,seconds=43200) for target in ('qualification','gpu')]
    base=confined(workspace,args.pilot)/'execution/private/prepared'
    context['recipe_ref']=reference(confined(base,context['runtime']['recipe']['path']),workspace)
    context['descriptors']=reference(confined(base,context['study']['data']['descriptors']['path']),workspace)
    refs=[context['preparation'],context['recipe_ref'],context['descriptors'],*[r['ref'] for r in context['originals']]]
    write(folder/'original-context.json',context)
    arms=['A','B','C'] if args.fresh_control else ['B','C']
    if args.bandit_only:arms.append('D')
    limits=envelope(purpose='research',arms=arms)
    stages=[]
    for i,name in enumerate(stage_order(research=args.research,arms=arms)):
        operation={'admission':'assemble','qualification':'qualify','calibration':'calibrate','acceptance':'acceptance','baseline':'evaluate','analysis':'analysis'}.get(name,'train' if name.startswith('train-') else 'evaluate')
        category={'admission':'preparation','qualification':'validation','calibration':'calibration','acceptance':'acceptance','baseline':'generation','analysis':'export'}.get(name,'learner' if name.startswith('train-') else 'generation')
        overhead=limits['validation']['outer']+4*1800+7200
        seconds=limits[category]['outer']+overhead+(limits['proof']['outer']+limits['validation']['outer'] if operation=='evaluate' else 0)+(86400 if name=='qualification' else 0)
        stages.append(dict(name=name,operation=operation,manifest='@admission',seconds=seconds,extra=[],requires=[] if not i else [stages[-1]['name']]))
    plan=prepare_plan(workspace,engine=args.engine,stages=stages,inputs=refs+[reference(folder/'original-context.json',workspace)],
        source=digest(collect_snapshot(workspace)),images=pair,research=args.research,arms=arms)
    # Context and requested seed are immutable stage configuration, not outcomes.
    plan['stages'][0]['extra']=[str((folder/'original-context.json').relative_to(workspace)),str(args.seed)]
    plan=seal(plan);write(path,plan)
    print('Frozen plan: '+str(path)+' identity='+plan['identity'],flush=True)
    print(json.dumps(dict(scope='research' if plan['research'] else 'software acceptance only',arms=arms,
        target_tokens_per_research_arm=25000000,parameter='DEFERRED',assessment='NOT_EXECUTED',
        source=plan['source'],images=pair,engine=args.engine,initial_ceiling_seconds=plan['seconds'],
        measured_ETA='unavailable before representative calibration'),sort_keys=True),flush=True)
    return path,plan


def _resolve_manifest(stage,workspace,receipts):
    if stage['manifest'].startswith('@'):
        key=stage['manifest'][1:]
        # Acceptance may publish a compatible gate-bound authority; scientific
        # choices/data/model stay frozen, and native admission bytes stay bound.
        if stage['name'] not in {'qualification','calibration','acceptance'} and 'acceptance' in receipts:key='acceptance'
        if stage['name']=='acceptance' and 'calibration' in receipts:key='calibration'
        parent=receipts[key]
        return parent['resolved_manifest'],parent.get('resolved_runtime')
    return stage['manifest'],None


_canonical_command_stage=_canonical_stage

def _canonical_stage(plan,stage,target,workspace,receipts,*,resume_path=None):
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.generalization.contracts import artifact_refs,verify_artifact
    workspace=Path(workspace)
    if stage['name']=='admission':
        context=_read(confined(workspace,stage['extra'][0]));seed=int(stage['extra'][1])
        settings=dict(context=context,images=plan['images'],engine=plan['engine'],research=plan['research'],arms=plan['arms'],seed=seed,source=plan['source'])
        result=_role_job(plan,'pac-assemble',settings,{'mechanical_sources':plan['inputs'],'isolation_truth':[]},
            {'mechanical_sources':workspace},target,workspace)
        if result['status']!='PASS':return result
        role=target/'execution/execution/role'
        result['resolved_manifest']=str((role/result['result']['manifest']).relative_to(workspace))
        result['resolved_runtime']=str((role/result['result']['runtime']).relative_to(workspace))
        return result
    manifest,runtime_path=_resolve_manifest(stage,workspace,receipts)
    study=_read(confined(workspace,manifest));base=confined(workspace,manifest).parent
    runtime=_read(confined(workspace,runtime_path)) if runtime_path else _read(base/'runtime.json')
    qualification=str((target.parent/'qualification/execution/qualification').relative_to(workspace))
    if stage['name'] in {'calibration','acceptance'}:
        settings=dict(recipe=json.loads(verify_artifact(base,runtime['recipe'])),source=plan['source'],
            environment=digest(dict(images=plan['images'],engine=plan['engine'],backend=study['backend'])))
        artifacts={'training_targets':[],'verified_training_replay':[]};roots={}
        operation='pac-campaign-acceptance' if stage['name']=='acceptance' else 'pac-calibrate'
        if operation=='pac-calibrate':
            bundle=json.loads(verify_artifact(base,runtime['prepared']['training']))
            from oeis_learn.generalization.role_contracts import pac_training_refs
            settings.update(bundle=bundle,study=study)
            artifacts['training_targets']=[*pac_training_refs(bundle,'C'),*pac_training_refs(bundle,'A')];roots['training_targets']=base
        original=receipts['admission']
        native_root=target.parent/'admission/execution'
        native_admission=dict(root=str(native_root.relative_to(workspace)),producer=reference(native_root/'result.json',native_root),admission=reference(base/'admission.json',native_root))
        result=_role_job(plan,operation,settings,artifacts,roots,target,workspace,qualification=qualification,native_admission=native_admission)
        result.update(resolved_manifest=manifest,resolved_runtime=runtime_path)
        if stage['name']=='calibration' and result['status']=='PASS':
            from oeis_learn.pipeline_validation.generalization_pac_calibration import apply_calibration
            study,runtime=apply_calibration(study,runtime,result['result'])
            write(base/'calibrated-study.json',study);write(base/'calibrated-runtime.json',runtime)
            result.update(resolved_manifest=str((base/'calibrated-study.json').relative_to(workspace)),resolved_runtime=str((base/'calibrated-runtime.json').relative_to(workspace)),operational_envelope=runtime['envelope'])
        if stage['name']=='acceptance' and result['status']=='PASS':
            from oeis_learn.pipeline_validation.generalization_execution import bind_pac
            # Preserve the acceptance root: its nested proof references resolve there.
            acceptance_root=target/'execution/execution/role/acceptance'
            ref=reference(acceptance_root/'pac-acceptance.json',workspace)
            runtime['feature']=dict(qualification=None,pac=dict(ref,path=str((acceptance_root/'pac-acceptance.json').relative_to(base)) if acceptance_root.is_relative_to(base) else 'pac-feature/pac-acceptance.json'))
            import shutil
            destination=base/'pac-feature'
            if not destination.exists():
                destination.mkdir()
                # Only public proof artifacts, no learner state, enter this closure.
                record=_read(acceptance_root/'pac-acceptance.json')
                for r in [reference(acceptance_root/'pac-acceptance.json',acceptance_root),*artifact_refs(record)]:
                    src=confined(acceptance_root,r['path']);dst=confined(destination,r['path'],exists=False);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            runtime['feature']['pac']=reference(destination/'pac-acceptance.json',base)
            study,runtime=bind_pac(study,runtime)
            write(base/'qualified-study.json',study);write(base/'qualified-runtime.json',runtime)
            result.update(resolved_manifest=str((base/'qualified-study.json').relative_to(workspace)),resolved_runtime=str((base/'qualified-runtime.json').relative_to(workspace)))
        return result
    if stage['name']=='analysis':
        from analysis import analyze_campaign
        value=analyze_campaign(target.parent,output=target)
        return dict(status='PASS',producer=digest(value),process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True),
            artifacts=[reference(target/'analysis.json',target),reference(target/'cases.md',target)])
    extra=['--runtime',runtime_path] if runtime_path else []
    if stage['operation']=='train':
        extra+=['--arms',stage['name'].split('-')[1],'--seeds',str(study['seed'])]
        if resume_path:extra+=['--resume',resume_path]
    if stage['operation']=='evaluate':
        if stage['name']=='baseline':
            context=_read(confined(workspace,plan['stages'][0]['extra'][0]))
            import shutil
            c=context['checkpoints']['static'];root=confined(workspace,c['weights_root'])
            owned=target/'private/retained';weights=c['checkpoint']['weights']
            destination=confined(owned,weights['path'],exists=False);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(confined(root,weights['path']),destination)
            write(owned/'checkpoint.json',c['checkpoint'])
            extra+=['--checkpoints',('A-reference=' if study['control_mode']=='fresh-control' else 'A=')+str((owned/'checkpoint.json').relative_to(workspace))]
        else:
            arm=stage['name'].split('-')[1]
            root=target.parent/('train-'+arm)/'execution'/('training-'+arm)/'role/private/learner/checkpoints'
            roster=_read(root/'retained.json');descriptor=next(c for c in roster['checkpoints'] if c['identity']==roster['latest'])
            write(root/'pac-final.json',descriptor,exclusive=False)
            extra+=['--checkpoints',arm+'='+str((root/'pac-final.json').relative_to(workspace))]
        extra+=['--proposal-bank',digest(list(range(20261101,20261109)))]
    effective=dict(stage,manifest=manifest,extra=extra,qualification=qualification)
    result=_canonical_command_stage(plan,effective,target,workspace,receipts)
    result.update(resolved_manifest=manifest,resolved_runtime=runtime_path)
    return result


def _model_resume(plan,stage,previous,workspace,receipts):
    """Use only verified compatible full state; never restore a policy journal alone."""
    from oeis_learn.pipeline_validation.generalization_commands import _resume_inputs
    from oeis_learn.pipeline_validation.generalization_runtime import learner_config
    from oeis_learn.pipeline_validation.generalization_execution import bind_compatibility
    from oeis_learn.generalization.contracts import verify_artifact
    arm=stage['name'].split('-')[1]
    root=previous/stage['name']/'execution'/('training-'+arm)/'role/private/learner/checkpoints'
    pointer=root/'generalization-latest.json'
    if not pointer.exists():return None
    latest=_read(pointer);path=confined(root,latest['path'])
    parent,ref,refs=_resume_inputs(workspace,str(path.relative_to(workspace)))
    if ref!=latest:raise ValueError('resume latest pointer changed')
    manifest,runtime_path=_resolve_manifest(stage,workspace,receipts)
    base=confined(workspace,manifest).parent;study=_read(confined(workspace,manifest));runtime=_read(confined(workspace,runtime_path))
    prepared=json.loads(verify_artifact(base,runtime['prepared']['training']))
    recipe=json.loads(verify_artifact(base,runtime['recipe']))
    config=learner_config(study,runtime,recipe,arm,study['seed'],prepared=prepared)
    data=digest(dict(prepared=prepared['identity'],arm=arm,batch_size=config['batch_size']))
    expected=digest(dict(config=config,data=data,**bind_compatibility('cuda',{})))
    state=json.loads(verify_artifact(parent,ref))
    if state['contract']!=expected:raise ValueError('failed learner full state is incompatible; no automatic fresh restart')
    return str(path.relative_to(workspace))


def bootstrap_failure(args,exc,*,workspace=ROOT):
    """Even preflight/build failure gets an independently verified negative archive."""
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    output=confined(workspace,str(Path(args.output)/'bootstrap-failure'),exists=False)
    if output.exists():raise ValueError('preserve previous bootstrap failure; choose a fresh output')
    output.mkdir(parents=True)
    arms=['A','B','C'] if args.fresh_control else ['B','C']
    if args.bandit_only:arms.append('D')
    stages=[dict(name=n,operation='prepare',manifest='UNAVAILABLE',seconds=3600,extra=[],requires=[] if not i else [order[i-1]])
            for order in [stage_order(research=args.research,arms=arms)] for i,n in enumerate(order)]
    plan=prepare_plan(workspace,engine=args.engine,stages=stages,inputs=[],source=digest(collect_snapshot(workspace)),
        images=[digest('UNAVAILABLE qualification image'),digest('UNAVAILABLE GPU image')],research=args.research,arms=arms)
    plan['stages'][0]['extra']=['UNAVAILABLE',str(args.seed)];plan=seal(plan)
    write(output/'plan.json',plan)
    workflow=dict(version='pac-workflow/v1',plan=plan['identity'],status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'BLOCKED',
        receipts=[],errors=[dict(stage='bootstrap',kind=type(exc).__name__,status='FAIL')])
    _persist(output/'workflow.json',workflow)
    print('Preflight/build blocked: '+str(exc),file=sys.stderr,flush=True)
    try:
        import shutil
        archive=recover(output,output/'public-report');shutil.copyfile(archive,output/'report.zip')
        print('Verified failure report: '+str(output/'report.zip'),flush=True)
    except Exception as secondary:
        workflow['errors'].append(dict(stage='export',kind=type(secondary).__name__,status='FAIL'))
        _persist(output/'workflow.json',workflow)
        print('Last-resort evidence: '+str(output/'workflow.json'),flush=True)
    return 130 if isinstance(exc,KeyboardInterrupt) else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('prepare','run'):
        command=sub.add_parser(name)
        command.add_argument('--engine',choices=('docker','podman'),required=True)
        command.add_argument('--prepared',default='run-state/generalization/fresh-p1-v2-20261103T035534Z')
        command.add_argument('--pilot',default='run-state/generalization/fresh-p1-hourly-20261104T093555Z')
        command.add_argument('--output',required=True)
        command.add_argument('--plan')
        command.add_argument('--research',action='store_true')
        command.add_argument('--fresh-control',action='store_true');command.add_argument('--bandit-only',action='store_true')
        command.add_argument('--seed',type=int,default=20261016)
    resume=sub.add_parser('resume');resume.add_argument('--plan',required=True);resume.add_argument('--previous',required=True);resume.add_argument('--output',required=True)
    recovery=sub.add_parser('recover');recovery.add_argument('--previous',required=True);recovery.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.action=='recover':
        archive=recover(confined(ROOT,args.previous),confined(ROOT,args.output,exists=False))
        print('Verified recovered report: '+str(archive),flush=True);return 0
    if args.action=='resume':
        plan=_read(confined(ROOT,args.plan));result=execute_plan(plan,confined(ROOT,args.output,exists=False),previous=confined(ROOT,args.previous))
    elif args.action=='prepare':build_plan(args);return 0
    else:
        if args.plan:
            plan=_read(confined(ROOT,args.plan))
            if args.engine!=plan['engine']:raise ValueError('requested engine differs from frozen plan')
        else:
            import copy
            inputs=copy.copy(args);inputs.output=str(Path(args.output)/'inputs/plan.json')
            try:_,plan=build_plan(inputs)
            except (Exception,KeyboardInterrupt) as exc:return bootstrap_failure(args,exc)
        result=execute_plan(plan,confined(ROOT,str(Path(args.output)/'campaign'),exists=False))
    return 0 if result['status']=='PASS' else 130 if result['status']=='INTERRUPTED' else 1


if __name__=='__main__':raise SystemExit(main())
