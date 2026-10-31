"""Frozen multi-case acceptance session; no implicit research or model choices."""
import copy,json,shutil,time
from pathlib import Path
from .identity import seal,validate,write,confined
from .generalization_preparation import reference
from .generalization_runtime import load
from .generalization_commands import _copy,_validate,_qualification,_finish
from oeis_learn.generalization.contracts import verify_artifact
from oeis_learn.generalization.feature_acceptance import OBLIGATIONS
from oeis_learn.generalization.interfaces import EXIT_CODES


def load_session(workspace,manifest):
    path=confined(workspace,manifest)
    if path.stat().st_size>1<<20:raise ValueError('acceptance session size cap')
    session=json.loads(path.read_text())
    if session.get('version') not in {'generalization-acceptance-session/v1','generalization-acceptance-session/v2'}:raise ValueError('registered session version required')
    validate(session,{'version','cases'},session['version'])
    names=set(OBLIGATIONS)-{'failure-packaging'}
    if not isinstance(session['cases'],dict) or set(session['cases'])!=names:raise ValueError('four frozen production-path acceptance inputs required')
    children={};images=None;source=None
    for name,refs in session['cases'].items():
        if set(refs)!={'manifest','runtime'}:raise ValueError('closed acceptance case inputs required')
        for ref in refs.values():verify_artifact(path.parent,ref,max_bytes=1<<20)
        manifest_path=confined(path.parent,refs['manifest']['path']);runtime_path=confined(path.parent,refs['runtime']['path'])
        base,study,runtime=load(workspace,str(manifest_path.relative_to(workspace)),str(runtime_path.relative_to(workspace)))
        from oeis_learn.generalization.execution import validate_policy
        validate_policy(study,runtime,case=name)
        if runtime['purpose']!='acceptance':raise ValueError('session cannot execute research work')
        if images is None:images=runtime['images'];source=study['source_closure']['sha256']
        if runtime['images']!=images or study['source_closure']['sha256']!=source:raise ValueError('one exact qualification scope required')
        learning=study['budgets']['seconds'];evaluation=(runtime['evaluation'] or {}).get('seconds',0)
        work=evaluation if name=='retained-eval' else 2*learning+120 if name in {'refresh','parameter-targets'} else (runtime['replay'] or {}).get('seconds',0)+4*learning+2*evaluation+120
        if work>360:raise ValueError('acceptance case cannot fit qualification inspection, validation and bounded work in 900 seconds')
        children[name]=(base,study,runtime,manifest_path,runtime_path)
    from oeis_learn.generalization.bootstrap import is_pending
    for name,(_,study,runtime,_,_) in children.items():
        if not is_pending(study['model']['initialization']):continue
        source=children['retained-eval']
        if session['version']!='generalization-acceptance-session/v2' or name!='replay-resume' or study['model']['initialization']['study']!=source[1]['identity']:
            raise ValueError('frozen retained-eval dependency required')
        if any(study['model'][k]!=source[1]['model'][k] for k in ('architecture','profile')) or study['precision']!=source[1]['precision'] or runtime['recipe']['sha256']!=source[2]['recipe']['sha256']:
            raise ValueError('bootstrap model/recipe differs from retained evaluation')
    return session,children


def _execute(args,workspace,host_factory):
    workspace=Path(workspace);session,children=load_session(workspace,args.manifest)
    anchor=children['refresh'];output=confined(workspace,args.output,exists=False)
    if getattr(args,'runtime',None) is not None:raise ValueError('acceptance session binds its own runtime sidecars')
    if args.operation not in {'preflight','qualify','prepare','acceptance','report'}:raise ValueError('unregistered acceptance session operation')
    if args.image is not None and args.image!=anchor[2]['images'][0]:raise ValueError('global image differs from frozen acceptance image')
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh acceptance output required')
    if args.operation=='qualify':
        from .generalization_commands import main as command
        effective=copy.copy(args);effective.manifest=str(anchor[3].relative_to(workspace));effective.runtime=str(anchor[4].relative_to(workspace))
        return command(effective,workspace,host_factory)
    if args.operation=='report':return report(args,workspace,host_factory,session,children)
    host=host_factory(args);output.mkdir(parents=True);began=time.monotonic()
    def terminal(result):
        record=seal(dict(version='generalization-session-result/v1',session=session['identity'],operation=args.operation,status=result['status'],result=result))
        write(output/'result.json',record);write(output/'summary.json',dict(selected_scope=result['status'],scientific_status='UNMEASURED'))
        return EXIT_CODES[result['status']]
    qualified=None
    try:
        if args.operation=='preflight':
            from oeis_learn.generalization.resources import preflight,CAPS
            storage={k:sum(child[2]['storage'][k] for child in children.values()) for k in CAPS}
            reservations=preflight(**storage,available_bytes=shutil.disk_usage(workspace).free)
            plans={name:_validate(host,workspace,output/name,base,study,runtime,seconds=120) for name,(base,study,runtime,_,_) in children.items()}
            return terminal(dict(status='PASS',plans=plans,reservations=reservations,qualification_required=True,repeat_reason='fresh current input and resource inspection'))
        qualified=_qualification(host,workspace,output,anchor[0],anchor[1],anchor[2],args)
        if args.operation=='prepare':
            from .generalization_runtime import execution_refs,publish_runtime
            from .generalization_roles import project_and_invoke
            prepared={}
            for ordinal,(name,(base,study,runtime,manifest_path,runtime_path)) in enumerate(children.items()):
                allocation=min(120,(590-(time.monotonic()-began))/(2*(len(children)-ordinal)))
                if allocation<=0:raise ValueError('preparation session reservation exhausted')
                _validate(host,workspace,output/name,base,study,runtime,seconds=allocation)
                if runtime['preparation'] is not None:
                    refs=execution_refs(base,study,runtime,preparation=True)
                    authority=json.loads(verify_artifact(base,runtime['preparation']))
                    result=project_and_invoke(host,runtime['images'][0],workspace,'preparation',{'mechanical_sources':refs,'isolation_truth':[]},
                        source_roots={'mechanical_sources':base,'isolation_truth':base},output=str((output/name/'prepare').relative_to(workspace)),
                        operation='prepare',settings=dict(preparation=authority,proposed=study),seconds=min(allocation,authority['budgets']['seconds']))
                    if result['status']!='PASS':return terminal(dict(status=result['status'],case=name,result=result))
                    root=output/name/'prepare/role/private/prepared';publish_runtime(base,root,study,runtime)
                    manifest_path=root/'study.json';runtime_path=root/'runtime.json'
                prepared[name]=dict(manifest=reference(manifest_path,workspace),runtime=reference(runtime_path,workspace))
            write(output/'prepared.json',seal(dict(version='generalization-prepared-session/v1',session=session['identity'],cases=prepared)))
            return terminal(dict(status='PASS',prepared=reference(output/'prepared.json',workspace),repeat_reason='revalidate frozen inputs and publish fresh role projections' if getattr(args,'reuse_run',None) else None))
        if args.operation!='acceptance':raise ValueError('unregistered session operation')
        inputs=confined(workspace,args.inputs)
        prepared=json.loads((inputs/'prepared.json').read_text());validate(prepared,{'version','session','cases'},'generalization-prepared-session/v1')
        if prepared['session']!=session['identity'] or set(prepared['cases'])!=set(children):raise ValueError('prepared session differs from frozen inputs')
        name=args.case;key='refresh' if name=='failure-packaging' else name
        refs=prepared['cases'][key]
        for ref in refs.values():verify_artifact(workspace,ref,max_bytes=1<<20)
        base,study,runtime=load(workspace,refs['manifest']['path'],refs['runtime']['path'])
        # Preparation can resolve artifacts, not model/budget/arm choices.
        original=children[key][1]
        if any(study[k]!=original[k] for k in ('model','budgets','arms','seeds','precision','inference','checkpoints','progression','recipe','track','hypothesis','changing_factors','source_closure')):raise ValueError('prepared acceptance authority changed')
        from oeis_learn.generalization.execution import verify_resolution
        verify_resolution(children[key][2],runtime)
        if any(study['data'][k]!=original['data'][k] for k in ('generator','descriptors','exclusions','target_format','sharing_policy')):raise ValueError('prepared data policy changed')
        from oeis_learn.generalization.bootstrap import is_pending
        if is_pending(study['model']['initialization']):
            from .generalization_bootstrap import resolve
            base,study,runtime=resolve(workspace,output,base,study,runtime,session=session,children=children,prepared=prepared,qualification=qualified['original'])
        dependency_source=session['version']=='generalization-acceptance-session/v2' and name=='retained-eval'
        if getattr(args,'reuse_run',None) and not dependency_source:
            previous=confined(workspace,args.reuse_run)/name
            from oeis_learn.generalization.feature_acceptance import verify_cases
            prior=json.loads((previous/'session-binding.json').read_text())
            if prior!=dict(session=session['identity'],study=study['identity'],runtime=runtime['identity']):raise ValueError('acceptance reuse binding changed')
            ref=reference(previous/'case/receipt.json',previous.parent)
            refs=verify_cases({name:ref},qualification=qualified['original'],root=previous.parent)
            # Exact original receipts remain original; no REUSED qualification.
            for ref in refs:
                relative=str(Path(ref['path']).relative_to(name))
                _copy(output,[(previous,dict(ref,path=relative))])
            write(output/'session-binding.json',prior)
            write(output/'reuse.json',dict(original=ref,source=str(previous.relative_to(workspace))))
            return terminal(dict(status='PASS',case=name,reused=True))
        plan=_validate(host,workspace,output,base,study,runtime,seconds=60)
        # The case owns the stage receipt directory; validation observations are
        # retained beside it and only its explicit public inventory is exported.
        from .generalization_acceptance import run_case
        case_root=output/'case'
        result=run_case(name,host,workspace,case_root,base,study,runtime,plan,qualification=qualified['original'],evidence_root=output.parent)
        # Preserve relative artifact refs while publishing the original case tree
        # at the session's stable case name. The parent is never mounted wholesale.
        if result['status']=='PASS':
            result['case_root']=str(case_root.relative_to(workspace))
        if dependency_source and getattr(args,'reuse_run',None):result['repeat_reason']='revalidate original selection and publish initializer dependency'
        write(output/'session-binding.json',dict(session=session['identity'],study=study['identity'],runtime=runtime['identity']))
        return terminal(result)
    except (Exception,KeyboardInterrupt) as exc:
        if args.operation=='acceptance' and qualified is not None and not (output/'case/receipt.json').exists():
            from .receipts import Ledger
            ledger=Ledger(output/'case','acceptance.'+args.case,dict(case=args.case,qualification=qualified['original']))
            ledger.transition('READY');ledger.transition('RUNNING');ledger.transition('PARTIAL' if isinstance(exc,KeyboardInterrupt) else 'FAIL',reason=type(exc).__name__)
        return terminal(dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',reason=type(exc).__name__,error=str(exc)))


def report(args,workspace,host_factory,session,children):
    from .generalization_qualification_io import public_closure
    from .generalization_roles import project_and_invoke
    from oeis_learn.generalization.feature_acceptance import assemble_feature,verify_cases
    from oeis_learn.container_validation import inspect_current_image
    root=confined(workspace,args.inputs);output=confined(workspace,args.output,exists=False)
    image=children['refresh'][2]['images'][0]
    if args.image is not None and args.image!=image:raise ValueError('acceptance report image mismatch')
    host=host_factory(args)
    if host.engine.image(image)!=image:raise ValueError('immutable acceptance report image changed')
    output.mkdir(parents=True);inspect_current_image(host,image,output/'inspection',seconds=120)
    staging=output/'private/public-input';staging.mkdir(parents=True)
    qualification=root/'qualify';q=json.loads((qualification/'qualification.json').read_text())
    refs=[reference(qualification/'qualification.json',qualification),*public_closure(qualification,q)]
    _copy(staging,[(qualification,r) for r in refs])
    original=json.loads(verify_artifact(qualification,q['producer']['receipt']))['identity']
    cases={};statuses={}
    for name in OBLIGATIONS:
        folder=root/name/'case'
        if not (folder/'receipt.json').is_file():continue
        stage=json.loads((folder/'receipt.json').read_text());statuses[name]=stage['state']
        # Case artifact refs are relative to their parent's directory and cannot
        # be rewritten without changing original identities. Keep one namespace
        # per case in the archive and rebase only the outer receipt reference.
        relative=name+'/case/receipt.json';target=staging/name
        child_refs=[reference(folder/'receipt.json',folder.parent)]
        child_refs.extend(reference(folder/file,folder.parent) for file in stage['files'])
        _copy(target,[(folder.parent,r) for r in child_refs])
        cases[name]=reference(staging/relative,staging)
    # The original case documents' ArtifactRefs must use the archive-wide case
    # namespace. Producers below write those refs at their canonical session root.
    missing=sorted(set(OBLIGATIONS)-set(cases))
    status=next((s for s in ('FAIL','BLOCKED','INCONCLUSIVE') if s in statuses.values()),'PARTIAL' if missing or 'PARTIAL' in statuses.values() else 'PASS')
    workflow=None
    if (root/'workflow.json').is_file():
        _copy(staging,[(root,reference(root/'workflow.json',root))]);workflow=reference(staging/'workflow.json',staging)
        ledger=json.loads(verify_artifact(staging,workflow));codes={r['exit_code'] for r in ledger['receipts']}
        if codes & {1,2}:status='FAIL'
        elif 130 in codes:status='INTERRUPTED'
        elif 3 in codes and status=='PASS':status='BLOCKED'
        elif {r['stage'] for r in ledger['receipts']}!={'preflight','qualify','prepare',*OBLIGATIONS} and status=='PASS':status='PARTIAL'
    feature=None
    if status=='PASS':
        record=assemble_feature(cases,qualification=original,root=staging);write(staging/'feature.json',record);feature=reference(staging/'feature.json',staging)
    inventory=[reference(p,staging) for p in sorted(staging.rglob('*')) if p.is_file()]
    record=seal(dict(version='generalization-acceptance-archive/v1',session=session['identity'],qualification=reference(staging/'qualification.json',staging),
        feature=feature,cases=cases,inventory=inventory,software_status=status,scientific_status='UNMEASURED',omissions=missing,workflow=workflow))
    write(staging/'evidence.json',record)
    result=project_and_invoke(host,image,workspace,'report',{'public_evidence':[reference(staging/'evidence.json',staging),*inventory]},
        source_roots={'public_evidence':staging},output=str((output/'packaging').relative_to(workspace)),operation='package',settings=dict(evidence=reference(staging/'evidence.json',staging)),seconds=480)
    if 'archive' in result:
        role=output/'packaging/role';verify_artifact(role,result['archive'],max_bytes=512<<20)
        shutil.copyfile(confined(role,result['archive']['path']),output.with_suffix('.zip'))
        if result['status']=='PASS':_copy(output/'qualified',[(staging,r) for r in inventory])
    return EXIT_CODES[result['status']]


def main(args,workspace,host_factory):
    from .generalization_lifecycle import scope
    from oeis_learn.generalization.resources import CAPS
    _,children=load_session(Path(workspace),args.manifest)
    limits={key:min(CAPS[key],sum(c[2]['storage'][key] for c in children.values())) for key in CAPS}
    output=confined(workspace,args.output,exists=False)
    with scope(output,limits,extra_roots=[c[0] for c in children.values()]):return _execute(args,workspace,host_factory)
