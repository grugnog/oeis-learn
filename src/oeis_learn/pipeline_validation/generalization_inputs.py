"""Standard-library assembly of explicit inputs; no model work or qualification."""
import argparse,copy,json,shutil,sys,zipfile
from pathlib import Path
from .identity import seal,write,confined,file_hash,digest
from .generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact,artifact_refs
from oeis_learn.generalization.execution import bind,validate_policy,RECIPES,FACTORS,PAIRS


def _json(path,cap=4<<20):
    if not path.is_file() or path.stat().st_size>cap:raise ValueError('bounded regular JSON file required: '+str(path))
    return json.loads(path.read_text())


def _fresh(workspace,output):
    target=confined(workspace,output,exists=False)
    if target.exists():raise ValueError('fresh input destination required')
    return target


def _copy_file(source,destination):
    if source.is_symlink() or not source.is_file():raise ValueError('regular input required: '+str(source))
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if file_hash(destination)!=file_hash(source):raise ValueError('conflicting input destination')
    else:shutil.copyfile(source,destination)


def import_uniform(workspace,*,bundle,metadata,weights,output):
    """Import only the verified uniform best/latest bytes and original metadata."""
    from .bundle import verify
    from oeis_learn.generalization.legacy import audit_uniform,verify_legacy
    workspace=Path(workspace);target=_fresh(workspace,output)
    archive=confined(workspace,bundle);verify(archive)
    metadata=confined(workspace,metadata);weights=confined(workspace,weights)
    def member(z,name):
        if z.getinfo(name).file_size>4<<20:raise ValueError('metadata member size cap')
        return z.read(name)
    with zipfile.ZipFile(archive) as z:
        originals={key:member(z,name) for key,name in dict(report='stages/study.report/study.json',
            producer='stages/followup.uniform/receipt.json',history='stages/followup.uniform/train/validation-history.json').items()}
    with zipfile.ZipFile(metadata) as z:
        if sorted(z.namelist())!=['prepared.json','recipe.json']:raise ValueError('metadata ZIP must contain exactly prepared.json and recipe.json')
        originals.update({key:member(z,key+'.json') for key in ('prepared','recipe')})
    docs={key:json.loads(value) for key,value in originals.items()}
    report=docs['report']
    if seal(report)!=report or report['version']!='pipeline-followup-study-result/v1':raise ValueError('original follow-up study report required')
    matches=[r for r in report['arms'] if r['profile']['arm']=='uniform']
    if len(matches)!=1:raise ValueError('exactly one original uniform record required')
    training=matches[0];checked=audit_uniform(training,docs['prepared'],docs['recipe'],docs['history'],docs['producer'])
    import hashlib
    def hashed(data):return 'sha256:'+hashlib.sha256(data).hexdigest()
    if docs['prepared']['files']['recipe.json']['sha256']!=hashed(originals['recipe']):raise ValueError('original recipe bytes mismatch')
    if docs['producer']['files'].get('train/validation-history.json')!=hashed(originals['history']):raise ValueError('original history inventory mismatch')
    paths={label:confined(weights,'private/'+label+'-weights.pt') for label in ('best','latest')}
    for label,path in paths.items():
        if not path.is_file() or path.stat().st_size>32<<30 or file_hash(path)!=training[label+'_weights']:
            raise ValueError('uniform '+label+' weights mismatch')
    target.mkdir(parents=True)
    refs={}
    for key,data in originals.items():
        p=target/'provenance'/(key+'.json');p.parent.mkdir(exist_ok=True);p.write_bytes(data);refs[key]=reference(p,target)
    write(target/'provenance/training.json',training);refs['training']=reference(target/'provenance/training.json',target)
    descriptors=[]
    for label,path in paths.items():
        destination=target/'private'/(label+'-weights.pt');_copy_file(path,destination)
        if file_hash(destination)!=training[label+'_weights']:raise ValueError('weights changed during copy')
        proof=seal(dict(version='generalization-legacy-import/v1',label=label,**refs))
        write(target/(label+'-provenance.json'),proof)
        descriptor=seal(dict(version='generalization-checkpoint/v2',kind='weights_only',weights=reference(destination,target),
            state=None,initialization=None,parent=None,**checked[label],pins=['historical_'+label],compatibility=digest(docs['recipe']),provenance=reference(target/(label+'-provenance.json'),target)))
        verify_legacy(target,descriptor);write(target/(label+'-checkpoint.json'),descriptor);descriptors.append(descriptor)
    result=seal(dict(version='generalization-legacy-roster/v1',checkpoints=descriptors,selection=checked['selection'],
        resume_available=False,initializer_available=False,source_bundle=file_hash(archive)))
    write(target/'import.json',result)
    return dict(status='IMPORTED',output=str(target.relative_to(workspace)),checkpoints=[c['identity'] for c in descriptors],resume_available=False,initializer_available=False)


def _checkpoint(workspace,path,target,namespace):
    source=confined(workspace,path);value=_json(source)
    candidates=value['checkpoints'] if value.get('version')=='generalization-retained/v1' else [value]
    parent=source.parent;destination=target/namespace
    _copy_file(source,destination/source.name)
    for candidate in candidates:
        if seal(candidate)!=candidate:raise ValueError('checkpoint descriptor seal mismatch')
        refs=[candidate['weights']]
        if candidate.get('kind')=='weights_only':
            from oeis_learn.generalization.legacy import verify_legacy
            verify_legacy(parent,candidate)
            proof=json.loads(verify_artifact(parent,candidate['provenance']))
            refs.extend([candidate['provenance'],*artifact_refs(proof)])
        for ref in refs:
            original=confined(parent,ref['path']);_copy_file(original,destination/ref['path'])
            if reference(destination/ref['path'],destination)!=ref:raise ValueError('checkpoint payload changed')
    return reference(destination/source.name,target)


BLUEPRINT_FIELDS={'version','stage','purpose','hypothesis','recipe','source_closure','images','initializer','descriptors','exclusions',
    'cells','training','evaluation','replay','seeds','budgets','milestones','storage','sharing_policy'}


def assemble_study(workspace,blueprint,output):
    """Paths in the blueprint are workspace-relative; output references are local."""
    workspace=Path(workspace);target=_fresh(workspace,output)
    cfg=_json(confined(workspace,blueprint)) if isinstance(blueprint,str) else copy.deepcopy(blueprint)
    if set(cfg) not in (BLUEPRINT_FIELDS,BLUEPRINT_FIELDS|{'feature'}) or cfg['version']!='generalization-input-blueprint/v1':raise ValueError('closed versioned input blueprint required')
    stage=cfg['stage']
    if stage not in RECIPES or stage=='P0':raise ValueError('use historical pilot-followup for P0')
    # Verify all direct input paths before creating output. Deep checks remain in
    # the canonical custodian; assembly never issues admission/qualification PASS.
    for key in ('recipe','source_closure','descriptors','exclusions'):_json(confined(workspace,cfg[key]))
    for cell in cfg['cells']:
        if 'candidates' in cell:_json(confined(workspace,cell['candidates']),8<<30)
    if cfg['initializer'] is not None:confined(workspace,cfg['initializer'])
    feature=None
    if cfg.get('feature') is not None:
        feature_path=confined(workspace,cfg['feature']);_json(feature_path)
        feature=reference(feature_path,feature_path.parent)
    if cfg['purpose']=='research' and feature is None:raise ValueError('research assembly requires original completed feature.json from the qualification root')
    if cfg['evaluation'] is not None:
        for path in cfg['evaluation']['checkpoints']:confined(workspace,path)
    if stage=='P4' and cfg['initializer'] is None:raise ValueError('P4 requires a verified selection-designated initializer; legacy best-loss weights are insufficient')
    target.mkdir(parents=True)
    def copy_input(key):
        destination=target/'inputs'/(key+'.json');_copy_file(confined(workspace,cfg[key]),destination);return reference(destination,target)
    recipe_ref=copy_input('recipe');recipe=_json(target/recipe_ref['path'])
    model={k:v for k,v in recipe['model'].items() if k!='seed'}
    for key in ('model_profile','inputs'):
        if key in recipe:model[key]=recipe[key]
    refs={key:copy_input(key) for key in ('source_closure','descriptors','exclusions')}
    initializer='random'
    if cfg['initializer'] is not None:
        from .generalization_runtime import initializer_refs
        p=confined(workspace,cfg['initializer']);local=dict(kind='retained',artifact=reference(p,p.parent))
        for ref in initializer_refs(p.parent,local):_copy_file(confined(p.parent,ref['path']),target/'initializer'/ref['path'])
        initializer=dict(kind='retained',artifact=reference(target/'initializer'/p.name,target))
    cells=copy.deepcopy(cfg['cells'])
    from collections import Counter
    quotas=Counter()
    for ordinal,cell in enumerate(cells):
        quotas.update(cell['quotas'])
        if 'candidates' in cell:
            path=target/'inputs'/('candidates-'+str(ordinal)+'.json');_copy_file(confined(workspace,cell['candidates']),path);cell['candidates']=reference(path,target)
    write(target/'inputs/materialization.json',dict(version='generalization-materialization-plan/v2',cells=cells))
    generator=reference(target/'inputs/materialization.json',target)
    write(target/'pending.json',dict(status='PREPARATION_REQUIRED'));pending=reference(target/'pending.json',target)
    arms=['retained','controls','shared_absent_bank'] if stage=='E0' else list(PAIRS[stage])
    study=dict(version='generalization-study/v3',recipe=RECIPES[stage],track='generic_only',hypothesis=cfg['hypothesis'],changing_factors=[FACTORS[stage]],arms=arms,seeds=cfg['seeds'],
        source_closure=refs['source_closure'],model=dict(profile=recipe['version'],initialization=initializer,architecture=digest(model)),precision='fp32',
        data=dict(generator=generator,descriptors=refs['descriptors'],manifests=[pending],target_format='literal',sharing_policy=cfg['sharing_policy'],exclusions=refs['exclusions']),
        selection=dict(cohort=pending,completion_floor=.95,objective='structural_family_macro_confirmed',tie_breaks=['parameter_family_macro_confirmed','earlier_tokens','identity']),
        assessment=dict(cohort=pending,authorization=pending,exposure_ledger=pending,feedback_to_selection=False),
        inference=dict(visible_terms=20,total_terms=100,attempts=8,seconds=60,policy='portfolio-clock60-matched-drain/v2'),
        budgets=cfg['budgets'],checkpoints=dict(milestones=cfg['milestones'],retain=['best','latest','selected']),
        progression=dict(stage=stage,advance_rule='selection-only',stop_rule='budget',assessment_feedback=False))
    evaluation=copy.deepcopy(cfg['evaluation'])
    if evaluation is not None:
        evaluation['checkpoints']=[_checkpoint(workspace,path,target,'retained/'+str(i)) for i,path in enumerate(evaluation['checkpoints'])]
    runtime=dict(version='generalization-runtime/v2',purpose=cfg['purpose'],recipe=recipe_ref,images=cfg['images'],preparation=None,training=cfg['training'],evaluation=evaluation,replay=cfg['replay'],assessment=None,feature=feature,storage=cfg['storage'])
    study,runtime=bind(study,runtime);validate_policy(study,runtime)
    prep=seal(dict(version='generalization-preparation/v3',proposed_study=study['identity'],generator=generator,numeric_law='independent-uniform-integers[-16,16]/v1',descriptors=refs['descriptors'],
        roles=['training','selection','assessment'],quotas=dict(quotas),exclusions=refs['exclusions'],seeds=cfg['seeds'],budgets=cfg['budgets'],
        study_settings={k:study[k] for k in ('recipe','arms','model','precision','checkpoints')}))
    write(target/'preparation.json',prep);runtime=seal(dict(runtime,preparation=reference(target/'preparation.json',target)))
    from .generalization_runtime import preparation_views,load
    preparation_views(target,study,runtime,prep)
    write(target/'study.json',study);write(target/'runtime.json',runtime)
    load(workspace,str((target/'study.json').relative_to(workspace)))
    write(target/'assembly.json',seal(dict(status='ASSEMBLED',study=study['identity'],runtime=runtime['identity'],qualification='NOT_RUN',native_admission='NOT_RUN')))
    return dict(status='ASSEMBLED',manifest=str((target/'study.json').relative_to(workspace)),runtime=str((target/'runtime.json').relative_to(workspace)))


def assemble_session(workspace,cases,output):
    from .generalization_session import load_session
    workspace=Path(workspace);target=confined(workspace,output,exists=False)
    if target.exists():raise ValueError('fresh session file required')
    if set(cases)!={'retained-eval','refresh','parameter-targets','replay-resume'}:raise ValueError('four case paths required')
    mapping={}
    for name,path in cases.items():
        if isinstance(path,dict):
            if set(path)!={'manifest','runtime'}:raise ValueError('closed case path pair required')
            manifest=confined(workspace,path['manifest']);runtime=confined(workspace,path['runtime'])
        else:
            manifest=confined(workspace,path);runtime=manifest.parent/'runtime.json'
        try:mapping[name]=dict(manifest=reference(manifest,target.parent),runtime=reference(runtime,target.parent))
        except ValueError as exc:raise ValueError('session file must be an ancestor of all four child studies') from exc
    record=seal(dict(version='generalization-acceptance-session/v1',cases=mapping))
    write(target,record)
    try:load_session(workspace,str(target.relative_to(workspace)))
    except BaseException:target.unlink();raise
    return dict(status='ASSEMBLED',manifest=str(target.relative_to(workspace)),identity=record['identity'])


def main(argv=None,*,workspace=None):
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    legacy=commands.add_parser('import-uniform')
    for flag in ('bundle','metadata','weights','output'):legacy.add_argument('--'+flag,required=True)
    study=commands.add_parser('study');study.add_argument('--blueprint',required=True);study.add_argument('--output',required=True)
    session=commands.add_parser('session');session.add_argument('--cases',required=True);session.add_argument('--output',required=True)
    snapshot=commands.add_parser('source-closure');snapshot.add_argument('--output',required=True)
    args=p.parse_args(argv);root=Path(workspace or Path.cwd()).resolve()
    try:
        if args.command=='import-uniform':result=import_uniform(root,**{k:getattr(args,k) for k in ('bundle','metadata','weights','output')})
        elif args.command=='study':result=assemble_study(root,args.blueprint,args.output)
        elif args.command=='session':result=assemble_session(root,_json(confined(root,args.cases)),args.output)
        else:
            from oeis_learn.experiments.source_snapshot import collect_snapshot
            destination=_fresh(root,args.output);write(destination,collect_snapshot(root))
            result=dict(status='RECORDED',source_closure=reference(destination,root),qualification='NOT_RUN')
        print(json.dumps(result,indent=2));return 0
    except (ValueError,OSError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        print(str(exc),file=sys.stderr);return 2
