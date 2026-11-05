"""Bounded public collection; original private producer records stay immutable."""
import json
from pathlib import Path
from .identity import seal,write,confined
from .generalization_preparation import reference
from .generalization_diagnostics import control_fields
from oeis_learn.generalization.contracts import verify_artifact,artifact_refs


def collect(original,staging):
    from .bundle import public
    from .generalization_qualification_io import public_closure
    original=Path(original);staging=Path(staging);staging.mkdir(parents=True,exist_ok=False)
    workflow=json.loads((original/'workflow.json').read_text())
    plan=json.loads((original/'plan.json').read_text())
    if seal(workflow)!=workflow or seal(plan)!=plan or workflow['plan']!=plan['identity']:raise ValueError('original workflow/plan binding mismatch')
    refs=[];used=0;omissions=[]
    def put(name,value):
        write(staging/name,value);ref=reference(staging/name,staging);refs.append(ref);return ref
    def copy(root,ref,prefix):
        nonlocal used
        path=str(Path(prefix)/ref['path'])
        if not public(Path(path)):raise ValueError('private producer payload cannot be exported')
        if used+ref['bytes']>480<<20 or len(refs)>9000:raise ValueError('PAC public proof cap')
        payload=verify_artifact(root,ref,max_bytes=480<<20);used+=len(payload)
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
        a=json.loads((nroot/'admission.json').read_text())
        if seal(a)!=a or a['status']!='PASS':raise ValueError('native admission failed/changed')
        producer=next((r['producer'] for r in workflow['receipts'] if r['stage']=='admission'),None)
        native=put('native-admission.json',seal(dict(version='pac-native-public/v1',status=a['status'],source=a['source'],images=a['images'],
            originals=[{k:v for k,v in x.items() if k!='ref'} for x in a['originals']],roster=a['roster'],
            admission_identity=a['identity'],producer=producer,protected_artifacts='OMITTED')))
    # Public source program cases are produced only by the scoped analysis adapter.
    analysis=original/'analysis/analysis.json'
    if analysis.is_file():copy(analysis.parent,reference(analysis,analysis.parent),'analysis')
    stages=[]
    for row in workflow['receipts']:
        stages.append(dict(stage=row['stage'],status=row['status'],producer=row['producer'],binding=row['binding'],
            process=control_fields(row.get('process',{})),cleanup=control_fields(row.get('cleanup',{})),
            elapsed_seconds=row.get('elapsed_seconds',0),reused=row.get('reused',False)))
    status=workflow['status']
    if status=='PASS' and not all((qualification,acceptance,native,calibration)):
        status='PARTIAL';omissions.append('complete original qualification/native/acceptance/calibration closure unavailable')
    seed=int(plan['stages'][0]['extra'][1]) if len(plan['stages'][0]['extra'])==2 else 0
    scope=dict(research=plan['research'],arms=plan['arms'],seed=seed,target_tokens=25000000,parameter='DEFERRED',assessment='NOT_EXECUTED')
    return seal(dict(version='pac-public-evidence/v1',plan=plan['identity'],source=plan['source'],images=plan['images'],
        backend=__import__('oeis_learn.pipeline_validation.generalization_execution',fromlist=['POLICY_ID']).POLICY_ID,
        helper=dict(requested=plan['helper'],observed=__import__('oeis_learn.pipeline_validation.identity',fromlist=['file_hash']).file_hash(Path(__file__).resolve().parents[3]/'specs/012-parameterized-adaptive-curriculum/validation/run.py')),
        scope=scope,required=[s['name'] for s in plan['stages']],stages=stages,software_status=status,scientific_status='UNMEASURED',errors=workflow['errors'],
        omissions=omissions+['private tensors, protected truth, raw commands and original host paths omitted'],inventory=refs,
        qualification=qualification,acceptance=acceptance,native_admission=native,calibration=calibration))


def export(original,output):
    from .generalization_finalization import publish_pac
    original=Path(original);output=Path(output)
    staging=output.parent/(output.name+'-payloads')
    record=collect(original,staging)
    return publish_pac(record,output,payload_root=staging)
