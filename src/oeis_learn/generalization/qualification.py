"""Exact qualification matching with original producer payloads, not REUSED chains."""
import json
from .contracts import validate, verify_artifact, artifact_refs
from oeis_learn.pipeline_validation.identity import seal

FACTS=frozenset({'source_closure','build_dependencies','images','runtime','hardware','precision','resources','engine_security'})
PRODUCER_FIELDS=FACTS  # the facts object, independent of full study authority


def qualify_reuse(receipt,current,*,required_suites,required_capabilities,root,research=False,feature_receipt=None):
    expected={'version','identity','status','facts','suites','capabilities','artifacts','dependencies','feature_acceptance'}
    if set(receipt)!=expected or receipt['version']!='generalization-producer/v1' or seal(receipt)!=receipt or receipt['status']!='PASS':
        raise ValueError('intact original PASS producer required')
    if set(current)!=FACTS or receipt['facts']!=current:
        raise ValueError('qualification facts changed (including unmapped source closure)')
    if not set(required_suites)<=set(receipt['suites']) or not set(required_capabilities)<=set(receipt['capabilities']):
        raise ValueError('qualification coverage missing')
    if not current['images'] or not all(str(x).startswith('sha256:') for x in current['images']):
        raise ValueError('immutable inspected images required')
    if not isinstance(current['hardware'],dict) or not isinstance(current['runtime'],dict) or not (current['hardware'].get('hip_available') is True or current['runtime'].get('hip_available') is True):
        raise ValueError('actual HIP required; CPU substitute forbidden')
    for ref in artifact_refs(current): verify_artifact(root,ref)
    for ref in receipt['dependencies']:
        dependency=json.loads(verify_artifact(root,ref))
        if dependency.get('status')!='PASS' or seal(dependency)!=dependency:
            raise ValueError('original dependency PASS required')
    if not receipt['artifacts']: raise ValueError('missing original device evidence')
    device_evidence=[]
    for ref in receipt['artifacts']:
        payload=verify_artifact(root,ref)
        try: value=json.loads(payload)
        except (ValueError,UnicodeError): continue
        if value.get('version')=='generalization-device-evidence/v1': device_evidence.append(value)
    if not any(seal(e)==e and e.get('device')=='hip' and e.get('loaded_weights') and all(e.get(k) is True for k in ('finite_forward','finite_backward','parameters_changed')) for e in device_evidence):
        raise ValueError('real load/finite-update evidence missing')
    if research:
        if feature_receipt is None: raise ValueError('completed feature acceptance required before research')
        feature=json.loads(verify_artifact(root,feature_receipt))
        if seal(feature)!=feature or feature.get('status')!='PASS' or feature.get('qualification')!=receipt['identity'] or set(feature.get('cases',()))!={'retained-eval','refresh','parameter-targets','replay-resume','failure-packaging'}:
            raise ValueError('incomplete feature acceptance')
    return receipt['identity']


def consume(authority,current,*,root,required_suites,required_capabilities,research=False,feature_receipt=None):
    """Schema authority adapter; consumers retain a separate full study identity."""
    q=validate('qualification',authority).record()
    facts={k:q[k] for k in FACTS}
    if facts!=current: raise ValueError('current qualification authority differs')
    for ref in artifact_refs(q): verify_artifact(root,ref)
    producer=json.loads(verify_artifact(root,q['producer']['receipt']))
    if set(q['suite_selection'])!=set(producer['suites']) or set(q['covered_capabilities'])!=set(producer['capabilities']) or q['producer']['artifacts']!=producer['artifacts']:
        raise ValueError('producer/authority binding mismatch')
    return qualify_reuse(producer,current,root=root,required_suites=required_suites,required_capabilities=required_capabilities,research=research,feature_receipt=feature_receipt)
