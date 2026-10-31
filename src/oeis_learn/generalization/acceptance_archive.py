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
