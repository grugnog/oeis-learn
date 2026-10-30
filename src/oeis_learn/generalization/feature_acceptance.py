"""Original per-case acceptance evidence required before research admission.

Hashes establish integrity and binding, not independent attestation of execution.
Only trusted canonical producers may issue the original stage/result records.
"""
import json
from pathlib import PurePosixPath,Path
from types import MappingProxyType
from .contracts import verify_artifact,artifact_refs
from oeis_learn.pipeline_validation.identity import seal,confined,validate
from oeis_learn.pipeline_validation.bundle import public

OBLIGATIONS=MappingProxyType({
    'retained-eval':(('weights_loaded','visible_only','matched_budgets'),('checkpoint_identity','evaluation')),
    'refresh':(('fixed_support','fresh_shards','finite_update'),('preparation','training')),
    'parameter-targets':(('literal_update','parameterized_update','grounding_verified'),('literal_training','parameterized_training','target_audit')),
    'replay-resume':(('training_only_admission','paired_control','full_state_resume','finite_update'),('admission','continued_sft','verified_replay','resume')),
    'failure-packaging':(('failure_preserved','archive_verified','private_payloads_excluded'),('failure_ledger','archive_verification')),
})
CAP=512<<20


def _verify_feature(feature,*,qualification,root):
    """Verify the closed feature authority, original stages and complete public closure."""
    validate(feature,{'version','status','qualification','cases'},'generalization-feature-acceptance/v2')
    if feature['status']!='PASS' or feature['qualification']!=qualification:
        raise ValueError('feature acceptance qualification/status mismatch')
    cases=feature['cases']
    if not isinstance(cases,dict) or set(cases)!=set(OBLIGATIONS):raise ValueError('complete feature case references required')
    seen={};cache={};used=0
    def read(ref):
        nonlocal used
        if not isinstance(ref,dict) or set(ref)!={'path','bytes','sha256'}:raise ValueError('case artifact reference required')
        if not isinstance(ref['path'],str) or not public(Path(ref['path'])):raise ValueError('public case evidence required')
        key=ref['path']
        if key in seen:
            if seen[key]!=ref:raise ValueError('conflicting case artifact reference')
            return cache[key]
        if type(ref['bytes']) is not int or ref['bytes']<0 or used+ref['bytes']>CAP or len(seen)>=10000:
            raise ValueError('feature acceptance evidence cap')
        payload=verify_artifact(root,ref,max_bytes=CAP)
        used+=len(payload);seen[key]=ref;cache[key]=payload
        return payload
    def document(ref):
        if not isinstance(ref,dict) or type(ref.get('bytes')) is not int or not 0<=ref['bytes']<=1<<20:
            raise ValueError('case document cap')
        value=json.loads(read(ref))
        if not isinstance(value,dict):raise ValueError('case document required')
        return value
    identities=set()
    for name,(checks,labels) in OBLIGATIONS.items():
        ref=cases[name];stage=document(ref)
        required={'version','identity','stage','attempt','binding','state','reason','files'}
        optional={'observations','prerequisites','command','counts','cleanup','prerequisite_equivalence'}
        if not required<=set(stage) or set(stage)-required-optional or stage['version']!='pipeline-stage/v1' or seal(stage)!=stage:
            raise ValueError('intact original case receipt required')
        if stage['state']!='PASS' or stage['stage']!='acceptance.'+name or stage['binding']!={'case':name,'qualification':qualification}:
            raise ValueError('original PASS case binding required')
        if not isinstance(stage['attempt'],str) or not stage['attempt'] or stage['identity'] in identities:
            raise ValueError('unique original case attempt required')
        identities.add(stage['identity'])
        files=stage['files']
        if not isinstance(files,dict) or not 1<=len(files)<=10000:raise ValueError('case artifact inventory required')
        parent=PurePosixPath(ref['path']).parent
        indexed={}
        for relative,sha in files.items():
            if not isinstance(relative,str) or not relative or PurePosixPath(relative).is_absolute() or any(p in ('','..','.') for p in relative.split('/')):
                raise ValueError('noncanonical case artifact path')
            path=str(parent/relative)
            try:size=confined(root,path).stat().st_size
            except OSError as exc:raise ValueError('missing case artifact') from exc
            child=dict(path=path,bytes=size,sha256=sha)
            read(child);indexed[relative]=child
        if 'case-result.json' not in indexed:raise ValueError('mandatory case result missing')
        result=document(indexed['case-result.json'])
        validate(result,{'version','case','qualification','status','checks','artifacts'},'generalization-acceptance-result/v1')
        if result['status']!='PASS' or result['case']!=name or result['qualification']!=qualification:
            raise ValueError('case result binding/status mismatch')
        if not isinstance(result['checks'],dict) or set(result['checks'])!=set(checks) or any(v is not True for v in result['checks'].values()):
            raise ValueError('incomplete case acceptance checks')
        artifacts=result['artifacts']
        if not isinstance(artifacts,dict) or set(artifacts)!=set(labels):raise ValueError('mandatory case artifacts missing')
        for label in labels:
            if artifacts[label]!=indexed.get(label+'.json'):raise ValueError('case artifact substituted or absent from original receipt')
        # Include nested public artifacts; missing/private descendants fail closed.
        pending=list(artifacts.values());expanded=set()
        while pending:
            child=pending.pop();payload=read(child)
            if child['path'] in expanded:continue
            expanded.add(child['path'])
            try:value=json.loads(payload)
            except (ValueError,UnicodeError):continue
            pending.extend(artifact_refs(value))
    return feature['identity'],list(seen.values())


def verify_feature(feature,*,qualification,root):
    return _verify_feature(feature,qualification=qualification,root=root)[0]


def feature_artifacts(feature,*,qualification,root):
    """Exact verified closure, including original stage files, for role projection."""
    return _verify_feature(feature,qualification=qualification,root=root)[1]


def assemble_feature(cases,*,qualification,root):
    """Producer helper issues an aggregate PASS only after all originals verify."""
    feature=seal(dict(version='generalization-feature-acceptance/v2',status='PASS',qualification=qualification,cases=cases))
    verify_feature(feature,qualification=qualification,root=root)
    return feature
