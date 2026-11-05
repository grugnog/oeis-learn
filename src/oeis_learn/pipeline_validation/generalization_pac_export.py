"""Bounded public collection; original private producer records stay immutable."""
import json
from pathlib import Path
from .identity import seal,write,confined,digest,file_hash
from .generalization_preparation import reference
from .generalization_diagnostics import control_fields
from oeis_learn.generalization.contracts import verify_artifact,artifact_refs


def helper_identity(root):
    """Bind temporary orchestration/analysis separately from installed learner code."""
    root=Path(root);files=[]
    for directory in ('specs/011-controlled-generalization/validation','specs/012-parameterized-adaptive-curriculum/validation'):
        files.extend(sorted((root/directory).glob('*.py')))
    if not files:raise ValueError('helper source closure unavailable')
    return digest({str(p.relative_to(root)):file_hash(p) for p in files})


def collect(original,staging,*,workspace=None):
    from .bundle import public
    from .generalization_qualification_io import public_closure
    workspace=Path(workspace) if workspace is not None else Path(__file__).resolve().parents[3]
    original=Path(original);staging=Path(staging);staging.mkdir(parents=True,exist_ok=False)
    workflow=json.loads((original/'workflow.json').read_text())
    plan=json.loads((original/'plan.json').read_text())
    if seal(workflow)!=workflow or seal(plan)!=plan or workflow['plan']!=plan['identity']:raise ValueError('original workflow/plan binding mismatch')
    refs=[];used=0;omissions=[]
    entries={row['stage']:row for row in workflow['receipts']}
    def original_file(path):
        relative=path.relative_to(original);stage=relative.parts[0]
        entry=entries.get(stage,{})
        root=original/entry.get('directory',stage)
        key=str(path.relative_to(root))
        ref=next((r for r in entry.get('artifacts',[]) if r['path']==key),None)
        if ref is None:raise ValueError('public proof absent from original stage inventory')
        return verify_artifact(root,ref,max_bytes=480<<20)
    def put(name,value):
        write(staging/name,value);ref=reference(staging/name,staging);refs.append(ref);return ref
    def copy(root,ref,prefix):
        nonlocal used
        path=str(Path(prefix)/ref['path'])
        if not public(Path(path)):raise ValueError('private producer payload cannot be exported')
        if used+ref['bytes']>480<<20 or len(refs)>9000:raise ValueError('PAC public proof cap')
        payload=verify_artifact(root,ref,max_bytes=480<<20)
        if Path(root).is_relative_to(original):original_file(Path(root)/ref['path'])
        used+=len(payload)
        destination=confined(staging,path,exists=False);destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():destination.write_bytes(payload);refs.append(dict(ref,path=path))
        return dict(ref,path=path)
    qualification=acceptance=native=calibration=None
    qroot=original/'qualification/execution/qualification'
    if (qroot/'qualification.json').is_file():
        q=json.loads((qroot/'qualification.json').read_text())
        for ref in [reference(qroot/'qualification.json',qroot),*public_closure(qroot,q)]:copy(qroot,ref,'qualification')
        qualification=reference(staging/'qualification/qualification.json',staging)
    aroot=original/'acceptance/execution/execution/role/acceptance'
    if (aroot/'pac-acceptance.json').is_file():
        record=json.loads((aroot/'pac-acceptance.json').read_text())
        for ref in [reference(aroot/'pac-acceptance.json',aroot),*artifact_refs(record)]:copy(aroot,ref,'acceptance')
        acceptance=reference(staging/'acceptance/pac-acceptance.json',staging)
    croot=original/'calibration/execution/execution/role'
    if (croot/'calibration.json').is_file():calibration=copy(croot,reference(croot/'calibration.json',croot),'calibration')
    nroot=original/'admission/execution/execution/role/private/assembly/resolved'
    if (nroot/'admission.json').is_file():
        a=json.loads(original_file(nroot/'admission.json'))
        if seal(a)!=a or a['status']!='PASS':raise ValueError('native admission failed/changed')
        if plan.get('scientific',{}).get('structural_scope')!=a.get('structural_scope') and plan.get('scientific',{}).get('structural_scope') is not None:raise ValueError('native scope differs from frozen plan')
        producer=next((r['producer'] for r in workflow['receipts'] if r['stage']=='admission'),None)
        native=put('native-admission.json',seal(dict(**({'structural_scope':a['structural_scope']} if a.get('structural_scope') else {}),version='pac-native-public/v2' if a.get('structural_scope') else 'pac-native-public/v1',status=a['status'],source=a['source'],images=a['images'],
            originals=[{k:v for k,v in x.items() if k!='ref'} for x in a['originals']],roster=a['roster'],
            admission_identity=a['identity'],producer=producer,protected_artifacts='OMITTED')))
    # These builder-produced summaries contain hashes/counts, never protected terms.
    for name in ('availability.json','training/representation.json'):
        path=nroot/name
        if path.is_file():copy(nroot,reference(path,nroot),'admission')
    # Public source program cases are produced only by the scoped analysis adapter.
    analysis=original/'analysis/analysis.json'
    scientific='INCOMPLETE' if plan['research'] else 'UNMEASURED'
    if analysis.is_file():
        copy(analysis.parent,reference(analysis,analysis.parent),'analysis')
        review=analysis.with_name('cases.md')
        if review.is_file():copy(analysis.parent,reference(review,analysis.parent),'analysis')
        report=json.loads(original_file(analysis))
        if report.get('version')=='pac-analysis/v2':
            from oeis_learn.generalization.pac_analysis import verify_analysis
            input_path=confined(analysis.parent,report['inputs']['path'])
            inputs=json.loads(verify_artifact(analysis.parent,report['inputs']))
            verify_analysis(report,inputs)
            copy(analysis.parent,report['inputs'],'analysis')
            accepted=entries['acceptance'];authority=[]
            for key in ('resolved_manifest','resolved_runtime'):
                path=confined(workspace,accepted[key])
                ref=next((r for r in accepted['resolved_refs'] if confined(workspace,r['path'])==path),None)
                if ref is None:raise ValueError('frozen research authority absent from accepted inventory')
                authority.append((json.loads(verify_artifact(workspace,ref,max_bytes=16<<20)),ref))
            (study,study_ref),(runtime,runtime_ref)=authority
            from .generalization_execution import verify_pac_runtime
            verify_pac_runtime(study,runtime)
            if runtime['purpose']!='research' or inputs['experiment']!=study or inputs['control']!=runtime['control'] or inputs['profiles']!=runtime['profiles']:raise ValueError('research measurements differ from frozen accepted authority')
            if native is None:raise ValueError('research native population authority unavailable')
            native_record=json.loads(verify_artifact(staging,native))
            put('research-authority.json',seal(dict(version='pac-research-authority/v1',experiment=study,control=runtime['control'],profiles=runtime['profiles'],
                study_sha256=study_ref['sha256'],runtime_sha256=runtime_ref['sha256'],admission_identity=native_record['admission_identity'],
                producer=accepted['producer'],binding=accepted['binding'],inputs=inputs['identity'],images=runtime['images'],plan=plan['identity'])))
        if plan['research'] and report.get('version')=='pac-analysis/v2':
            from oeis_learn.generalization.pac_analysis import scientific_status
            scientific=scientific_status(report,inputs)
    stages=[]
    for row in workflow['receipts']:
        stages.append(dict(stage=row['stage'],status=row['status'],producer=row['producer'],binding=row['binding'],
            process=control_fields(row.get('process',{})),cleanup=control_fields(row.get('cleanup',{})),
            elapsed_seconds=row.get('elapsed_seconds',0),reused=row.get('reused',False)))
    operational=original/'operational-freeze.json'
    limits=json.loads(operational.read_text()) if operational.is_file() else None
    put('campaign-controls.json',seal(dict(version='pac-campaign-controls/v1',structural_scope=plan.get('scientific',{}).get('structural_scope'),
        requested=[dict(stage=s['name'],seconds=s.get('seconds')) for s in plan['stages']],
        effective=[dict(stage=s['name'],seconds=s['seconds']) for s in limits['stages']] if limits else None,
        campaign_seconds=limits['seconds'] if limits else plan.get('seconds'),finalization_seconds=plan.get('finalization_seconds'),
        progress='stage elapsed/terminal status; see bounded projected logs; heartbeat is not completed work')))
    for row in workflow['receipts']:
        for folder in ('supervision','execution-supervision'):
            for stream in ('stdout.log','stderr.log'):
                path=original/row['stage']/folder/stream
                if path.is_file() and not path.is_symlink():
                    # Keep only public control/progress lines. Tracebacks, paths,
                    # source/term payloads and arbitrary role errors stay private.
                    size=path.stat().st_size
                    with path.open('rb') as handle:
                        handle.seek(max(0,size-(64<<10)));tail=handle.read(64<<10).decode('utf8',errors='replace')
                    lines=[line[:2048] for line in tail.splitlines() if line.startswith(('learner: ','calibration: ')) and '/' not in line and '\\' not in line]
                    put('logs/'+row['stage']+'/'+folder+'-'+stream+'.json',dict(original_bytes=size,original_sha256=file_hash(path) if size<=1<<20 else None,
                        projection='recognized phase/time lines only; raw logs remain private',lines=lines[-64:]))
    from .generalization_failures import stage_failures
    nested_errors=[]
    for row in workflow['receipts']:
        if row['status']=='PASS':continue
        details=stage_failures(original/row.get('directory',row['stage']),refs=row.get('artifacts',[]))
        put('failures/'+row['stage']+'.json',details)
        nested_errors.extend(dict(stage=row['stage'],kind=f['detail']['kind'],status='FAIL') for f in details['failures'])
    errors=list({(e['stage'],e['kind'],e['status']):e for e in [*nested_errors,*workflow['errors']]}.values())
    status=workflow['status']
    if status=='PASS' and not all((qualification,acceptance,native,calibration)):
        status='PARTIAL';omissions.append('complete original qualification/native/acceptance/calibration closure unavailable')
    seed=int(plan['stages'][0]['extra'][1]) if len(plan['stages'][0]['extra'])==2 else 0
    scope=dict(research=plan['research'],arms=plan['arms'],seed=seed,target_tokens=25000000,parameter='DEFERRED',assessment='NOT_EXECUTED')
    return seal(dict(version='pac-public-evidence/v1',plan=plan['identity'],source=plan['source'],images=plan['images'],
        backend=__import__('oeis_learn.pipeline_validation.generalization_execution',fromlist=['POLICY_ID']).POLICY_ID,
        helper=dict(requested=plan['helper'],observed=helper_identity(Path(__file__).resolve().parents[3])),
        scope=scope,required=[s['name'] for s in plan['stages']],stages=stages,software_status=status,scientific_status=scientific,errors=errors,
        omissions=omissions+['private tensors, protected truth, raw commands and original host paths omitted'],inventory=refs,
        qualification=qualification,acceptance=acceptance,native_admission=native,calibration=calibration))


def export(original,output):
    from .generalization_finalization import publish_pac
    original=Path(original);output=Path(output)
    staging=output.parent/(output.name+'-payloads')
    try:
        record=collect(original,staging)
        return publish_pac(record,output,payload_root=staging)
    except (OSError,ValueError,KeyError) as exc:
        # Broken rich closure cannot suppress a truthful, independently verified
        # minimal export. Preserve the original run and failed export directory.
        plan=json.loads((original/'plan.json').read_text());workflow=json.loads((original/'workflow.json').read_text())
        if seal(plan)!=plan or seal(workflow)!=workflow:raise ValueError('original plan/workflow integrity unavailable') from exc
        seed=int(plan['stages'][0]['extra'][1]) if len(plan['stages'][0]['extra'])==2 else 0
        from .generalization_failures import stage_failures
        minimal_payload=output.with_name(output.name+'-minimal-payloads');minimal_payload.mkdir(parents=True,exist_ok=False)
        minimal_refs=[];nested_errors=[]
        for row in workflow['receipts']:
            if row['status']=='PASS':continue
            details=stage_failures(confined(original,row.get('directory',row['stage'])),refs=row.get('artifacts',[]))
            path=minimal_payload/'failures'/(row['stage']+'.json');write(path,details);minimal_refs.append(reference(path,minimal_payload))
            nested_errors.extend(dict(stage=row['stage'],kind=f['detail']['kind'],status='FAIL') for f in details['failures'])
        errors=list({(e['stage'],e['kind'],e['status']):e for e in [*nested_errors,*workflow['errors'],dict(stage='export',kind=type(exc).__name__,status='FAIL')]}.values())
        record=seal(dict(version='pac-public-evidence/v1',plan=plan['identity'],source=plan['source'],images=plan['images'],
            backend=__import__('oeis_learn.pipeline_validation.generalization_execution',fromlist=['POLICY_ID']).POLICY_ID,
            helper=dict(requested=plan['helper'],observed=helper_identity(Path(__file__).resolve().parents[3])),
            scope=dict(research=plan['research'],arms=plan['arms'],seed=seed,target_tokens=25000000,parameter='DEFERRED',assessment='NOT_EXECUTED'),
            required=[s['name'] for s in plan['stages']],stages=[],software_status='INTERRUPTED' if workflow['status']=='INTERRUPTED' else 'FAIL',
            scientific_status='INCOMPLETE' if plan['research'] else 'UNMEASURED',errors=errors,
            omissions=['rich proof closure unavailable; original failure and export directories retained'],inventory=minimal_refs,qualification=None,acceptance=None,native_admission=None,calibration=None))
        return publish_pac(record,output.with_name(output.name+'-minimal'),payload_root=minimal_payload)
