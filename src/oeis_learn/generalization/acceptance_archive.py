"""Offline reconciliation of software acceptance; no scientific success claims."""
import json
from pathlib import Path,PurePosixPath
from .contracts import artifact_refs,verify_artifact
from .feature_acceptance import OBLIGATIONS,verify_feature,feature_artifacts,verify_cases
from .qualification import consume,FACTS
from oeis_learn.pipeline_validation.identity import validate,seal,confined
from oeis_learn.pipeline_validation.bundle import public


def verify_acceptance(record,*,root):
    validate(record,{'version','session','qualification','feature','cases','inventory','software_status','scientific_status','omissions','workflow'},'generalization-acceptance-archive/v1')
    inventory=record['inventory'];indexed={r['path']:r for r in inventory}
    if len(indexed)!=len(inventory) or len(inventory)>9997 or sum(r['bytes'] for r in inventory)>512<<20:raise ValueError('acceptance archive inventory cap')
    for ref in inventory:
        if not public(Path(ref['path'])):raise ValueError('private acceptance payload')
        verify_artifact(root,ref,max_bytes=512<<20)
    def read(ref):
        if indexed.get(ref['path'])!=ref:raise ValueError('acceptance artifact absent from inventory')
        return json.loads(verify_artifact(root,ref,max_bytes=1<<20))
    q=read(record['qualification']);original=consume(q,{k:q[k] for k in FACTS},root=root,required_suites=q['suite_selection'],required_capabilities=q['covered_capabilities'])
    pending=list(artifact_refs(q));seen=set()
    while pending:
        ref=pending.pop()
        if ref['path'] in seen:continue
        seen.add(ref['path'])
        if indexed.get(ref['path'])!=ref:raise ValueError('missing qualification closure member')
        try:value=json.loads(verify_artifact(root,ref,max_bytes=512<<20))
        except (ValueError,UnicodeError):continue
        pending.extend(artifact_refs(value))
    if not isinstance(record['cases'],dict) or set(record['cases'])-set(OBLIGATIONS):raise ValueError('unknown acceptance case')
    statuses={}
    for name,ref in record['cases'].items():
        receipt=read(ref)
        if receipt.get('version')!='pipeline-stage/v1' or seal(receipt)!=receipt or receipt.get('stage')!='acceptance.'+name or receipt.get('binding')!={'case':name,'qualification':original} or 'reused_from' in receipt:
            raise ValueError('original acceptance stage required')
        status=receipt['state']
        if status not in {'PASS','FAIL','PARTIAL','BLOCKED','INCONCLUSIVE'}:raise ValueError('terminal acceptance stage required')
        statuses[name]=status
        for relative,sha in receipt['files'].items():
            if PurePosixPath(relative).is_absolute() or any(p in ('','.','..') for p in relative.split('/')):raise ValueError('invalid receipt file path')
            path=str(PurePosixPath(ref['path']).parent/relative);child=indexed.get(path)
            if child is None or child['sha256']!=sha:raise ValueError('missing case file')
            verify_artifact(root,child)
    missing=sorted(set(OBLIGATIONS)-set(statuses))
    for child in verify_cases({name:record['cases'][name] for name,status in statuses.items() if status=='PASS'},qualification=original,root=root):
        if indexed.get(child['path'])!=child:raise ValueError('missing verified case closure member')
    actual=next((s for s in ('FAIL','BLOCKED','INCONCLUSIVE') if s in statuses.values()),'PARTIAL' if missing or 'PARTIAL' in statuses.values() else 'PASS')
    if record['workflow'] is not None:
        workflow=read(record['workflow'])
        if seal(workflow)!=workflow or workflow.get('version')!='generalization-serial-workflow/v1' or workflow.get('research_authorized') is not False:raise ValueError('sealed acceptance workflow required')
        receipts=workflow['receipts'];seen_stages=set()
        required={'preflight','qualify','prepare',*OBLIGATIONS}
        for row in receipts:
            if set(row)!={'stage','exit_code','elapsed_seconds'} or row['stage'] in seen_stages or row['stage'] not in required or row['exit_code'] not in {0,1,2,3,130}:raise ValueError('invalid pre-packaging workflow receipt')
            seen_stages.add(row['stage'])
        codes={row['exit_code'] for row in receipts}
        if codes & {1,2}:actual='FAIL'
        elif 130 in codes:actual='INTERRUPTED'
        elif 3 in codes and actual=='PASS':actual='BLOCKED'
        elif seen_stages!=required and actual=='PASS':actual='PARTIAL'
    if record['software_status']!=actual or record['scientific_status']!='UNMEASURED' or record['omissions']!=missing:
        raise ValueError('stale acceptance status or omissions')
    if actual=='PASS':
        if record['feature'] is None:raise ValueError('complete original feature acceptance required')
        feature=read(record['feature'])
        if feature['cases']!=record['cases']:raise ValueError('feature case substitution')
        verify_feature(feature,qualification=original,root=root)
        for child in feature_artifacts(feature,qualification=original,root=root):
            if indexed.get(child['path'])!=child:raise ValueError('missing feature closure member')
    elif record['feature'] is not None:raise ValueError('partial acceptance cannot qualify research')
    return dict(integrity='PASS',software_status=actual,scientific_status='UNMEASURED',scope='software acceptance only',
        cases=statuses,missing=missing,feature_qualified=actual=='PASS')


def verify_pac_evidence(record,*,root,finalized=True):
    """Reconcile original public proof closure and all terminal boundaries."""
    from oeis_learn.generalization.metrics import _pac_roster
    from oeis_learn.generalization.structural import original_cells
    from oeis_learn.generalization.feature_acceptance import verify_pac_acceptance
    from oeis_learn.generalization.qualification import consume_public,FACTS
    from oeis_learn.pipeline_validation.identity import digest
    fields={'version','plan','source','images','backend','helper','scope','required','stages','software_status','scientific_status',
            'errors','omissions','inventory','qualification','acceptance','native_admission','calibration'}
    validate(record,fields,'pac-public-evidence/v1')
    if record['scientific_status'] not in {'UNMEASURED','INCOMPLETE','EXPLORATORY_SINGLE_SEED'}:raise ValueError('scoped scientific status')
    import re
    identities=[record[k] for k in ('plan','source','backend')]+record['images']+[record['helper'][k] for k in ('requested','observed')]
    if len(record['images'])!=2 or set(record['helper'])!={'requested','observed'} or any(not isinstance(v,str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',v) for v in identities):raise ValueError('complete source/image/helper identities required')
    for error in record['errors']:
        if set(error)!={'stage','kind','status'} or not isinstance(error['kind'],str) or not re.fullmatch(r'[A-Za-z_]{1,64}',error['kind']):raise ValueError('bounded ordered public error types required')
    scope=record['scope']
    if set(scope)!={'research','arms','seed','target_tokens','parameter','assessment'} or type(scope['research']) is not bool or scope['target_tokens']!=25000000 or scope['parameter']!='DEFERRED' or scope['assessment']!='NOT_EXECUTED':raise ValueError('frozen PAC scientific scope required')
    if type(scope['seed']) is not int or not 0<=scope['seed']<2**32:raise ValueError('frozen seed required')
    if len(set(scope['arms']))!=len(scope['arms']) or not {'B','C'}<=set(scope['arms'])<=set('ABCD'):raise ValueError('predeclared complete arms')
    required=['admission','qualification','calibration','acceptance']
    if scope['research']:
        required+=['baseline']
        for arm in scope['arms']:required+=['train-'+arm,'evaluate-'+arm]
        required+=['analysis']
    if record['required']!=required:raise ValueError('complete frozen campaign stages required')
    inventory=record['inventory'];indexed={r['path']:r for r in inventory}
    if len(indexed)!=len(inventory) or len(inventory)>9990 or sum(r['bytes'] for r in inventory)>512<<20:raise ValueError('PAC evidence inventory bounds')
    for ref in inventory:
        if not public(Path(ref['path'])):raise ValueError('private PAC payload')
        verify_artifact(root,ref,max_bytes=512<<20)
    def read(ref):
        if ref is None or indexed.get(ref['path'])!=ref:raise ValueError('mandatory original proof missing from inventory')
        return json.loads(verify_artifact(root,ref,max_bytes=16<<20))
    analysis_ref=indexed.get('analysis/analysis.json')
    if analysis_ref is not None:
        report=read(analysis_ref)
        if report.get('version')=='pac-analysis/v2':
            from .pac_analysis import verify_analysis
            if report['inputs']['path']!='research-inputs.json':raise ValueError('fixed public research input path required')
            input_ref=dict(report['inputs'],path='analysis/research-inputs.json')
            verify_analysis(report,read(input_ref))
    seen=[]
    for stage in record['stages']:
        if set(stage)!={'stage','status','producer','binding','process','cleanup','elapsed_seconds','reused'}:raise ValueError('closed public stage receipt')
        if stage['stage'] not in required or stage['stage'] in seen:raise ValueError('stage order/identity')
        seen.append(stage['stage'])
        if stage['status']=='PASS' and (stage['process'].get('outcome')!='COMPLETE' or stage['process'].get('returncode')!=0 or stage['process'].get('cleanup_failure_type') or not all(stage['cleanup'].get(k) is True for k in ('within_bound','acknowledged'))):raise ValueError('application PASS has failed process/cleanup')
    if seen!=required[:len(seen)]:raise ValueError('dependent stages ran out of order')
    status=record['software_status'];statuses={s['status'] for s in record['stages']}
    allowed={'PASS','FAIL','BLOCKED','PARTIAL','INCONCLUSIVE','INTERRUPTED'}
    if status not in allowed or not statuses<=allowed:raise ValueError('known terminal outcomes')
    qualified=False
    if status=='PASS':
        if seen!=required or statuses!={'PASS'} or record['errors']:raise ValueError('incomplete campaign cannot PASS')
        native=read(record['native_admission'])
        if seal(native)!=native or native.get('status')!='PASS' or native.get('source')!=record['source'] or native.get('images')!=record['images']:raise ValueError('native admission source/image mismatch')
        if sorted(native['originals'],key=lambda r:r['name'])!=sorted(original_cells(),key=lambda r:r['name']):raise ValueError('all original exclusion cells required')
        if native.get('version') not in {'pac-native-public/v1','pac-native-public/v2'}:raise ValueError('known native population version required')
        if (native['version']=='pac-native-public/v2')!=('structural_scope' in native):raise ValueError('native structural scope version mismatch')
        _pac_roster(native['roster'],scope=native.get('structural_scope'))
        q=read(record['qualification']);qroot=confined(root,record['qualification']['path']).parent
        consume_public(q,{k:q[k] for k in FACTS},root=qroot,required_suites=['bigint-cpu','bigint-gpu'],required_capabilities=['weight-load','finite-update','native-exact'])
        if q['images']!=record['images'] or q['runtime'].get('execution_policy')!=record['backend'] or digest(json.loads(verify_artifact(qroot,q['source_closure'])))!=record['source']:raise ValueError('qualification closure differs')
        feature=read(record['acceptance']);froot=confined(root,record['acceptance']['path']).parent
        environment=digest(dict(images=record['images'],engine=q['engine_security']['engine'],backend=record['backend']))
        verify_pac_acceptance(feature,root=froot,source=record['source'],environment=environment,device='hip')
        calibration=read(record['calibration'])
        from oeis_learn.pipeline_validation.generalization_pac_calibration import verify_calibration
        verify_calibration(calibration,source=record['source'],environment=environment,arms=scope['arms'])
        qualified=True
    elif status not in {'FAIL','INTERRUPTED','BLOCKED','INCONCLUSIVE','PARTIAL'}:raise ValueError('negative scope required')
    result=seal(dict(integrity='PASS',software_status=status,scientific_status=record['scientific_status'],feature_qualified=qualified,evidence=record['identity']))
    if finalized:
        terminal=json.loads(confined(root,'terminal.json').read_text())
        verification=json.loads(confined(root,'verification.json').read_text())
        from oeis_learn.pipeline_validation.generalization_preparation import reference
        if seal(terminal)!=terminal or terminal.get('status')!='COMPLETE' or terminal.get('evidence')!=reference(confined(root,'evidence.json'),Path(root)) or terminal.get('verification')!=reference(confined(root,'verification.json'),Path(root)) or verification!=result:
            raise ValueError('missing/stale terminal COMPLETE or verifier binding')
    return result
