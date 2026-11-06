"""Bounded public failure causes; private exception text remains at its producer."""
import json
import os
from pathlib import Path
import re
from .identity import seal,digest,file_hash,confined
from oeis_learn.generalization.contracts import verify_artifact

_MESSAGES={
    'insufficient PAC reserved disk storage':('insufficient_disk_storage','Available free disk is below the frozen PAC reservation.'),
    "'source'":('missing_source_field','A required source field is missing.'),
    'missing artifact':('missing_artifact','A required verified artifact is missing.'),
    'invalid frozen execution: missing artifact':('missing_artifact','A required verified artifact is missing.'),
    'completed post-exclusion quota deficit; no timeout/reseed retry':('structural_quota_deficit','Frozen post-exclusion structural quotas are incomplete.'),
    'PAC structural native execution incomplete':('structural_native_incomplete','Structural native execution did not complete.'),
    'native/reference disagreement':('native_reference_disagreement','Native and independent reference outputs disagree.'),
    'protected finalization reserve':('finalization_reserve','The protected finalization reserve was reached.'),
}
_KIND_CODES={
    'StorageReservation':('insufficient_disk_storage','Available free disk is below the frozen PAC reservation.'),
    'QUALIFICATION_FAILED':('qualification_failed','Qualification failed; inspect the verified substage and test diagnostics.'),
    'QualificationStageFailure':('qualification_stage_failed','A qualification substage failed; original private logs remain at the producer.'),
    'QualificationBlocked':('qualification_blocked','Qualification was blocked by an unsuccessful prerequisite.'),
    'QualificationTestFailure':('qualification_tests_failed','Qualification tests failed; see the bounded test diagnostics.'),
    'QuotaDeficit':('structural_quota_deficit','Frozen post-exclusion structural quotas are incomplete.'),
    'CleanupFailure':('cleanup_failure','Cleanup was not acknowledged within its bound.'),
    'ProcessFailure':('process_failure','The supervised process failed.'),
}
_FIELDS={'version','identity','kind','code','message','deficits','private_detail_sha256','private_detail_truncated'}


def public_failure(exc=None,*,kind=None,message=None,deficits=None):
    if exc is not None:
        kind=type(exc).__name__;message=str(exc)
        if kind=='QuotaDeficit':deficits=getattr(exc,'deficits',None)
    if not isinstance(kind,str) or not re.fullmatch(r'[A-Za-z_]{1,64}',kind):kind='ApplicationFailure'
    raw=message if isinstance(message,str) else ''
    code,safe=_KIND_CODES.get(kind,_MESSAGES.get(raw,('private_exception_detail','Private exception detail omitted; inspect the original producer result.')))
    deficits={} if deficits is None else deficits
    if (not isinstance(deficits,dict) or len(deficits)>128 or any(not isinstance(k,str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',k)
        or type(v) is not int or not 0<v<=1000000 for k,v in deficits.items())):raise ValueError('bounded opaque quota deficits required')
    return seal(dict(version='pac-public-failure/v1',kind=kind,code=code,message=safe,deficits=dict(sorted(deficits.items())),
        private_detail_sha256=digest(raw[:65536]),private_detail_truncated=len(raw)>65536))


def _application(value):
    prior=value.get('failure')
    if isinstance(prior,dict):
        if set(prior)!=_FIELDS or seal(prior)!=prior or prior['version']!='pac-public-failure/v1':raise ValueError('invalid public failure record')
        public_failure(kind=prior['kind'],message=prior['message'],deficits=prior['deficits'])
        # Known messages are fixed public vocabulary; unknown raw text never crosses.
        allowed=set(_MESSAGES.values())|set(_KIND_CODES.values())|{('private_exception_detail','Private exception detail omitted; inspect the original producer result.')}
        if (prior['code'],prior['message']) not in allowed or type(prior['private_detail_truncated']) is not bool or not re.fullmatch(r'sha256:[0-9a-f]{64}',prior['private_detail_sha256']):raise ValueError('unsafe public failure fields')
        if prior['code']=='private_exception_detail' and not prior['private_detail_truncated']:
            # Exact hashes of fixed source literals can safely upgrade old
            # reports. Never infer or publish arbitrary private exception text.
            known={digest(raw):detail for raw,detail in _MESSAGES.items()}
            detail=known.get(prior['private_detail_sha256'])
            if detail is not None:return seal(dict(prior,code=detail[0],message=detail[1]))
        return prior
    return public_failure(kind=value.get('reason','ApplicationFailure'),message=value.get('error'))


def stage_failures(root,*,refs=None,max_entries=8192,max_bytes=4<<20):
    """Read only bounded result/control metadata, deepest application cause first.

    When recovering an old run, every source must match its original inventory.
    This does not rewrite or qualify failed artifacts and never reads tensor/truth
    directories. A cap or missing proof becomes an explicit omission.
    """
    root=Path(root);index={r['path']:r for r in refs} if refs is not None else None
    from .generalization_qualification_failures import RECEIPT,TESTS,test_summary
    import xml.etree.ElementTree as ET
    pending=[Path('.')];failures=[];omissions=[];entries=used=0;seen=set()
    qualification=dict(stages=[],tests=[]);storage=[]
    skip={'private','.runtime','input','inputs','checkpoints','native','native-cache','cache','caches','build'}
    while pending:
        folder=pending.pop()
        if len(folder.parts)>32 or root.is_symlink():omissions.append(dict(kind='DISCOVERY_BOUND'));continue
        try:
            with os.scandir(root/folder) as stream:
                children=[]
                for child in stream:
                    entries+=1
                    if entries>max_entries:break
                    children.append(child)
            if entries>max_entries:omissions.append(dict(kind='ENTRY_BOUND'));break
        except OSError:omissions.append(dict(kind='UNREADABLE_DIRECTORY'));continue
        for child in sorted(children,key=lambda c:c.name,reverse=True):
            path=Path(child.path);relative=path.relative_to(root);name=str(relative)
            if child.is_symlink():continue
            if child.is_dir(follow_symlinks=False):
                if child.name not in skip:pending.append(relative)
                continue
            application=path.name=='result.json'
            control=path.name in {'process.json','cleanup.json'} and path.parent.name.endswith('supervision')
            receipt=RECEIPT.fullmatch(name);tests=TESTS.fullmatch(name)
            stderr=name in {'supervision/stderr.log','execution-supervision/stderr.log'}
            storage_check=name=='execution/storage-check.json'
            if not (application or control or receipt or tests or stderr or storage_check):continue
            if any(not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',p) for p in relative.parts):omissions.append(dict(kind='UNSAFE_PATH'));continue
            try:
                path=confined(root,name);size=path.stat().st_size
                cap=4<<20 if tests else 1<<20
                if size>cap or used+size>max_bytes:raise ValueError('metadata size cap')
                if index is not None:
                    ref=index.get(name)
                    if ref is None:raise ValueError('original metadata unbound')
                    payload=verify_artifact(root,ref,max_bytes=cap);sha=ref['sha256']
                else:payload=path.read_bytes();sha=file_hash(path)
                used+=len(payload)
                if stderr:
                    # Only an exact fixed public CLI diagnostic is recognized.
                    # Arbitrary stderr and private exception text never cross.
                    if payload.rstrip(b'\r\n')==b'ERROR: insufficient PAC reserved disk storage':
                        failures.append((-2,-len(relative.parts),dict(origin=name,producer_sha256=sha,
                            detail=public_failure(kind='StorageReservation',message='insufficient PAC reserved disk storage'))))
                    continue
                if tests:
                    summary=test_summary(payload,origin=name,sha256=sha)
                    qualification['tests'].append(summary)
                    if summary['counts']['failures'] or summary['counts']['errors']:
                        failures.append((-1,-len(relative.parts),dict(origin=name,producer_sha256=sha,
                            detail=public_failure(kind='QualificationTestFailure'))))
                    continue
                value=json.loads(payload)
                if not isinstance(value,dict):raise ValueError('metadata object required')
                if storage_check:
                    from oeis_learn.generalization.resources import pac_free_storage
                    expected=pac_free_storage(required_bytes=value['required_bytes'],available_bytes=value['available_bytes'])
                    if seal(value)!=value or value!=expected:raise ValueError('original closed storage check required')
                    storage.append(dict(origin=name,producer_sha256=sha,check=value))
                    if value['status']=='BLOCKED':
                        failures.append((-2,-len(relative.parts),dict(origin=name,producer_sha256=sha,
                            detail=public_failure(kind='StorageReservation',message='insufficient PAC reserved disk storage'))))
                    continue
                if receipt:
                    if (seal(value)!=value or value.get('version')!='pipeline-stage/v1' or value.get('stage')!=receipt[1]
                        or value.get('state') not in {'PASS','FAIL','PARTIAL','BLOCKED','UNAVAILABLE'}):raise ValueError('original qualification receipt required')
                    qualification['stages'].append(dict(stage=receipt[1],state=value['state'],origin=name,producer_sha256=sha))
                    if value['state']!='PASS':
                        kind='QualificationBlocked' if value['state']=='BLOCKED' else 'QualificationStageFailure'
                        failures.append((0,-len(relative.parts),dict(origin=name,producer_sha256=sha,detail=public_failure(kind=kind))))
                    continue
                detail=None;priority=0 if application else 2 if path.name=='cleanup.json' else 1
                if application and value.get('status') in {'FAIL','BLOCKED','INTERRUPTED'}:
                    if path.parent.name=='role' or value.get('version') in {'pac-job-result/v1','generalization-command-result/v2'}:
                        if seal(value)!=value:raise ValueError('original role seal changed')
                        projected=value.get('result') if path.parent.name!='role' else value
                        if isinstance(projected,dict):
                            # Packaging can wrap an earlier work error; retain both.
                            prior=projected.get('work_result')
                            if isinstance(prior,dict) and prior.get('status') in {'FAIL','BLOCKED','INTERRUPTED'}:
                                failures.append((priority,-len(relative.parts)-1,dict(origin=name,producer_sha256=sha,detail=_application(prior))))
                            detail=_application(projected)
                elif path.name=='process.json' and (value.get('outcome') in {'FAIL','INTERRUPTED'} or value.get('returncode') not in (None,0)):
                    failure=value.get('failure','');kind=failure.split(':',1)[0] if isinstance(failure,str) and failure else 'ProcessFailure'
                    detail=public_failure(kind=kind,message=failure)
                elif path.name=='cleanup.json' and (value.get('acknowledged') is False or value.get('within_bound') is False or value.get('failure')):
                    detail=public_failure(kind='CleanupFailure')
                if detail is not None:
                    failures.append((priority,-len(relative.parts),dict(origin=name,producer_sha256=sha,detail=detail)))
            except (ValueError,OSError,TypeError,KeyError,UnicodeError,ET.ParseError):omissions.append(dict(origin=name,kind='UNVERIFIED_OR_UNAVAILABLE'))
    failures.sort(key=lambda x:(x[0],x[1],x[2]['origin']))
    unique=[]
    for _,_,record in failures:
        if record['detail']['identity'] not in seen:
            seen.add(record['detail']['identity']);unique.append(record)
    qualification['stages'].sort(key=lambda r:r['stage']);qualification['tests'].sort(key=lambda r:r['origin'])
    return seal(dict(version='pac-stage-failures/v3',failures=unique,omissions=omissions,qualification=qualification,storage=storage,
        examined_entries=entries,metadata_bytes=used,private_payloads='OMITTED'))
