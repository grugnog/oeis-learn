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
    policy=current['runtime'].get('execution_policy')
    if policy is not None and not any(seal(e)==e and e.get('device')=='hip' and e.get('loaded_weights') and
        all(e.get(k) is True for k in ('finite_forward','finite_backward','parameters_changed','resume_equal')) and
        e.get('execution_policy')==policy for e in device_evidence):
        raise ValueError('device/resume numerical policy evidence differs')
    if research:
        if feature_receipt is None: raise ValueError('completed feature acceptance required before research')
        from .feature_acceptance import verify_feature
        feature=json.loads(verify_artifact(root,feature_receipt,max_bytes=1<<20))
        verify_feature(feature,qualification=receipt['identity'],root=root)
    return receipt['identity']


def consume(authority,current,*,root,required_suites,required_capabilities,research=False,feature_receipt=None):
    """Schema authority adapter; consumers retain a separate full study identity."""
    q=validate('qualification',authority).record()
    return _consume_bound(q,current,root=root,required_suites=required_suites,required_capabilities=required_capabilities,research=research,feature_receipt=feature_receipt)


def _consume_bound(q,current,*,root,required_suites,required_capabilities,research=False,feature_receipt=None):
    facts={k:q[k] for k in FACTS}
    if facts!=current: raise ValueError('current qualification authority differs')
    for ref in artifact_refs(q): verify_artifact(root,ref)
    producer=json.loads(verify_artifact(root,q['producer']['receipt']))
    if set(q['suite_selection'])!=set(producer['suites']) or set(q['covered_capabilities'])!=set(producer['capabilities']) or q['producer']['artifacts']!=producer['artifacts']:
        raise ValueError('producer/authority binding mismatch')
    return qualify_reuse(producer,current,root=root,required_suites=required_suites,required_capabilities=required_capabilities,research=research,feature_receipt=feature_receipt)


def consume_public(q,current,*,root,required_suites,required_capabilities):
    """Strict PAC public-archive subset of v2, usable on a stdlib-only host.

    Full schema validation still runs in the installed qualification consumer.
    This narrower public boundary checks every exported field without importing
    the container's schema dependency or admitting legacy research authority.
    """
    import re
    def closed(value,fields):
        if not isinstance(value,dict) or set(value)!=set(fields):raise ValueError('closed public qualification fields required')
    def text(value):
        if not isinstance(value,str) or not 1<=len(value)<=1024:raise ValueError('bounded qualification text required')
    def hash_value(value):
        if not isinstance(value,str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',value):raise ValueError('qualification hash required')
    def positive(value):
        if type(value) is not int or value<1:raise ValueError('positive qualification integer required')
    def refs(values,minimum,maximum):
        if not isinstance(values,list) or not minimum<=len(values)<=maximum:raise ValueError('bounded qualification references required')
        for ref in values:
            closed(ref,{'path','bytes','sha256'});text(ref['path']);hash_value(ref['sha256'])
            if not re.fullmatch(r'(?:[A-Za-z0-9_-][A-Za-z0-9_.-]*/)*[A-Za-z0-9_-][A-Za-z0-9_.-]*',ref['path']):raise ValueError('qualification path required')
            verify_artifact(root,ref,max_bytes=512<<20)
    closed(q,FACTS|{'version','identity','suite_selection','covered_capabilities','producer','artifact_inventory','status'})
    if q['version']!='generalization-qualification/v2' or seal(q)!=q or q['status']!='PASS':raise ValueError('intact original public qualification required')
    for key in ('suite_selection','covered_capabilities'):
        values=q[key]
        if not isinstance(values,list) or not 1<=len(values)<=4096 or any(not isinstance(v,str) for v in values) or len(set(values))!=len(values):raise ValueError('unique bounded qualification coverage required')
        for value in values:text(value)
    if not isinstance(q['images'],list) or len(q['images'])!=2:raise ValueError('PAC image pair required')
    for image in q['images']:hash_value(image)
    refs([q['source_closure']],1,1);refs(q['build_dependencies'],1,64);refs(q['artifact_inventory'],0,10000)
    closed(q['runtime'],{'hip','driver','python','torch','execution_policy'})
    for key,value in q['runtime'].items():hash_value(value) if key=='execution_policy' else text(value)
    closed(q['hardware'],{'hip_available','device','memory_bytes'})
    if q['hardware']['hip_available'] is not True:raise ValueError('actual HIP fact required')
    text(q['hardware']['device']);positive(q['hardware']['memory_bytes']);text(q['precision'])
    resources=q['resources'];closed(resources,{'memory_bytes','cpus','torch_intraop','torch_interop','native_threads','learners'})
    positive(resources['memory_bytes']);positive(resources['cpus'])
    if resources['cpus']>16 or any(type(resources[k]) is not int or resources[k]!=v for k,v in dict(torch_intraop=2,torch_interop=1,native_threads=1,learners=1).items()):raise ValueError('qualified resource policy required')
    security=q['engine_security'];closed(security,{'engine','runtime_socket','engine_version','runtime','rootless','selinux','network','read_only_root','no_new_privileges','capabilities'})
    if security['engine'] not in {'docker','podman'} or any(type(security[k]) is not bool for k in ('rootless','selinux')):raise ValueError('explicit engine/security facts required')
    for key in ('runtime_socket','network'):
        if security[key] is not False:raise ValueError('isolated qualification required')
    for key in ('read_only_root','no_new_privileges'):
        if security[key] is not True:raise ValueError('isolated qualification required')
    if security['capabilities']!='none':raise ValueError('no capabilities required')
    text(security['engine_version']);text(security['runtime'])
    closed(q['producer'],{'receipt','status','artifacts'})
    if q['producer']['status']!='PASS':raise ValueError('original producer PASS required')
    refs([q['producer']['receipt']],1,1);refs(q['producer']['artifacts'],1,10000)
    return _consume_bound(q,current,root=root,required_suites=required_suites,required_capabilities=required_capabilities)
