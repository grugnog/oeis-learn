"""Finite installed HIP feature rehearsal; Q capability reuse remains separate."""
import os
from pathlib import Path
import time
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, reference, seal, verify_refs

from .progressive_acceptance import inventory

NODES=tuple(inventory()['integrated_hip']['nodes'])


def device(purpose):
    if purpose=='research' or os.environ.get('OEIS_PROGRESSIVE_ACCEPTANCE_DEVICE')=='actual_HIP':
        from .progressive_qualification import require_hip
        require_hip()
        return 'cuda'
    return 'cpu'


def selected_profile(default):
    """Only the separately owned device rehearsal selects its exact Q profile."""
    path=os.environ.get('OEIS_PROGRESSIVE_ACCEPTANCE_REQUEST')
    if os.environ.get('OEIS_PROGRESSIVE_ACCEPTANCE_DEVICE')!='actual_HIP':
        return default
    if not path:
        raise ValueError('frozen device acceptance request required')
    from oeis_learn.generalization.progressive.models import resolve_profile
    request=load(path,maximum=4<<20)
    p=request['profile']
    from oeis_learn.generalization.progressive.models import validate_profile
    validate_profile(p)
    # Training seeds remain paired fixture inputs; architecture/dropout are the
    # exact selected capability. A changed profile requires matching Q originals.
    return resolve_profile(p['base_recipe'],shape=p['shape'],seed=default['model']['seed'],dropout=p['model']['dropout'])


def run_inside(request, output):
    import sys
    import xml.etree.ElementTree as ET
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from .progressive_acceptance import _bounded_process
    from .progressive_qualification import hardware_facts, require_hip
    fields={'version','identity','campaign','runtime','source','image','engine','hardware','profile','execution_policy','seconds'}
    if not isinstance(request,dict) or set(request)!=fields or request['version']!='progressive-device-acceptance-request/v1':
        raise ValueError('closed frozen device request required')
    began=time.monotonic(); require_hip()
    from .generalization_execution import verify_execution
    if request['execution_policy']!=verify_execution():
        raise ValueError('frozen device numerical policy differs')
    if (seal(request)!=request or request['source']!=digest(source_snapshot()) or
        request['image']!=os.environ.get('OEIS_QUALIFICATION_IMAGE') or
        request['engine']!=os.environ.get('OEIS_QUALIFICATION_ENGINE') or
        request['hardware']!=hardware_facts()):
        raise ValueError('current exact device campaign scope required')
    if type(request['seconds']) is not int or not 1<=request['seconds']<=1800:
        raise ValueError('finite integrated device allowance required')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    atomic_write(output/'request.json',request)
    import zipfile
    installed=output/'installed'
    with zipfile.ZipFile(output.parent/'package.whl') as archive:
        archive.extractall(installed)
    env=dict(os.environ,PYTHONPATH=str(installed)+os.pathsep+'/opt/oeis-build',OEIS_PROGRESSIVE_ACCEPTANCE_DEVICE='actual_HIP',
             OEIS_PROGRESSIVE_ACCEPTANCE_REQUEST=str(output/'request.json'),
             OEIS_REQUIRE_NATIVE='1',OEIS_REQUIRE_BIGINT_BUILD='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    code=_bounded_process([sys.executable,'-m','pytest','-q','-o','addopts=',
                           '--basetemp',str(output/'private'), '--junitxml',str(output/'cases.xml'),*NODES],
                          cwd='/opt/oeis-build',env=env,log=output/'process.log',
                          seconds=request['seconds'],maximum=4<<20)
    path=output/'cases.xml'
    if path.is_symlink() or path.stat().st_size>4<<20:
        raise ValueError('bounded original device cases required')
    entries=ET.parse(path).findall('.//testcase')
    names=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in entries]
    if code or sorted(names)!=sorted(NODES) or any(c.find(t) is not None for c in entries for t in ('error','failure','skipped')):
        raise ValueError('every registered actual device scenario must pass')
    if request['source']!=digest(source_snapshot()):
        raise ValueError('source changed during device acceptance')
    result=seal(dict(version='progressive-device-acceptance/v1',request=digest(request),
                     campaign=request['campaign'],runtime=request['runtime'],source=request['source'],
                     image=request['image'],engine=request['engine'],hardware=request['hardware'],
                     profile=request['profile'],status='PASS',device='actual_HIP',cases=list(NODES),
                     junit=reference(output,'cases.xml'),actual_elapsed_us=int((time.monotonic()-began)*1e6)))
    atomic_write(output/'producer.json',result)
    return result


def verify_producer(root, producer, request):
    import xml.etree.ElementTree as ET
    fields={'version','identity','request','campaign','runtime','source','image','engine','hardware','profile','status','device','cases','junit','actual_elapsed_us'}
    if (not isinstance(producer,dict) or set(producer)!=fields or seal(producer)!=producer or producer.get('version')!='progressive-device-acceptance/v1' or
        producer.get('status')!='PASS' or producer.get('device')!='actual_HIP' or producer.get('cases')!=list(NODES) or
        producer.get('request')!=digest(request) or any(producer.get(k)!=request[k] for k in
            ('campaign','runtime','source','image','engine','hardware','profile')) or
        type(producer.get('actual_elapsed_us')) is not int or producer['actual_elapsed_us']<=0):
        raise ValueError('original exact feature device producer required')
    [raw]=verify_refs(root,[producer['junit']],max_bytes=4<<20)
    if b'<!DOCTYPE' in raw or b'<!ENTITY' in raw:
        raise ValueError('device XML declarations forbidden')
    cases=ET.fromstring(raw).findall('.//testcase')
    actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
    if sorted(actual)!=sorted(NODES) or any(c.find(t) is not None for c in cases for t in ('error','failure','skipped')):
        raise ValueError('missing or unsuccessful original device cases')
    return producer


def run_campaign(args, root, host, campaign, q, plan, output, software, *, began):
    from .progressive_handoff import stage, validate_device_proofs
    from .progressive_roles import current_runtime
    from .progressive_evidence import publish
    from .identity import confined
    from .generalization_qualification_io import current_facts
    runtime=load(confined(root,args.campaign).parent/'runtime.json')
    if (runtime['identity']!=campaign['payload']['runtime'] or runtime['purpose']!=campaign['purpose'] or runtime['track']!=campaign['track']):
        raise ValueError('frozen campaign/runtime binding differs')
    qroot=confined(root,plan['qualification']['path']).parent
    if plan['qualification'] not in runtime['payload']['qualification']:
        raise ValueError('acceptance qualification differs from frozen runtime')
    checked=current_runtime(host,root,output/'runtime-inspection',runtime,producer_root=qroot)
    source=digest(__import__('oeis_learn.experiments.source_snapshot',fromlist=['collect_snapshot']).collect_snapshot(root))
    matrix,refs=validate_device_proofs(qroot,q,source=source,image=q['images'][1],engine=host.engine.name)
    profiles={c['profile']['identity']:c['profile'] for c in matrix['cases'] if
              digest({k:v for k,v in c['profile']['model'].items() if k!='seed'})==checked.payload['scope']['model']}
    if not profiles:
        raise ValueError('selected campaign model absent from original Q cases')
    profile=next(iter(profiles.values()))
    if {c['input_length'] for c in matrix['cases'] if c['profile']==profile and c['microbatch']==7}!={4,20}:
        raise ValueError('four-term and twenty-term accumulation scope required')
    if software['source']!=source:
        raise ValueError('software acceptance source differs')
    remaining=int(plan['seconds']-(time.monotonic()-began))-30
    if remaining<=0:
        raise TimeoutError('acceptance allowance exhausted before device batch')
    request=seal(dict(version='progressive-device-acceptance-request/v1',campaign=campaign['identity'],
                      runtime=runtime['identity'],source=source,image=q['images'][1],engine=host.engine.name,
                      hardware=matrix['hardware'],execution_policy=q['runtime']['execution_policy'],profile=profile,seconds=min(1800,remaining)))
    target=output/'device';target.mkdir()
    atomic_write(target/'input.json',request)
    from .progressive_acceptance import hardware_plan
    atomic_write(target/'reuse.json',hardware_plan(qroot,matrix,refs))
    import shutil
    [wheel]=(output/'software/suite/dist').glob('*.whl')
    shutil.copyfile(wheel,target/'package.whl')
    code="from oeis_learn.generalization.progressive.contracts import load;from oeis_learn.pipeline_validation.progressive_device_acceptance import run_inside;run_inside(load('/work/output/input.json'),'/work/output/integrated')"
    stage(host,root,q['images'][1],target,['python','-c',code],gpu=True,seconds=remaining)
    proof=verify_producer(target/'integrated',load(target/'integrated/producer.json'),request)
    value=seal(dict(version='discovery-acceptance/v1',scope='progressive-lifecycle',qualified=True,
                    campaign=campaign['identity'],runtime=runtime['identity'],source=source,status='COMPLETE',
                    software=reference(output,'software/suite/acceptance.json'),
                    device=reference(output,'device/integrated/producer.json'),request=reference(output,'device/input.json'),
                    sustained_status='PENDING',scientific_status='UNMEASURED'))
    import shutil
    shutil.copyfile(confined(root,args.campaign),output/'campaign.json')
    shutil.copyfile(confined(root,args.campaign).parent/'runtime.json',output/'runtime.json')
    _publish_receipt(output,value,qroot,q)
    return publish(output,operation='acceptance',mechanical_status='COMPLETE',qualification_status='QUALIFIED',
                   campaign=campaign['identity'],reason='ORIGINAL_CPU_AND_HIP_FEATURE_ACCEPTANCE',
                   metrics={'acceptance':reference(output,'acceptance.json')}),0


def consume_receipt(root, receipt):
    """Consumers replay original proofs rather than trusting a new qualified bit."""
    from .identity import confined
    if (not isinstance(receipt,dict) or seal(receipt)!=receipt or receipt.get('version')!='discovery-acceptance/v1' or
        receipt.get('qualified') is not True or receipt.get('status')!='COMPLETE' or
        receipt.get('scope')!='progressive-lifecycle'):
        raise ValueError('intact completed feature receipt required')
    if not {'software','device','request'}<=set(receipt):
        raise ValueError('original feature producers required')
    verify_refs(root,[receipt[k] for k in ('software','device','request')],max_bytes=32<<20)
    software_path=confined(root,receipt['software']['path'])
    software=load(software_path)
    verify_refs(software_path.parent,[software['junit'],software['cases']],max_bytes=8<<20)
    from .progressive_acceptance import reconcile_suite
    if reconcile_suite(confined(software_path.parent,software['junit']['path'])) != load(confined(software_path.parent,software['cases']['path'])):
        raise ValueError('original software case reconciliation differs')
    if software['status']!='COMPLETE' or software['qualified'] is not False or software['source']!=receipt['source']:
        raise ValueError('original CPU/native feature producer required')
    request=load(confined(root,receipt['request']['path']))
    if (request['campaign'],request['runtime'],request['source'])!=(receipt['campaign'],receipt['runtime'],receipt['source']):
        raise ValueError('feature campaign changed')
    proof_path=confined(root,receipt['device']['path'])
    verify_producer(proof_path.parent,load(proof_path),request)
    return receipt


def _publish_receipt(output, value, qualification_root, qualification):
    output=Path(output)
    atomic_write(output/'acceptance.pending.json',value)
    export_package(output,qualification_root,qualification,receipt_path=output/'acceptance.pending.json')
    atomic_write(output/'acceptance.json',value)


def export_package(output, qualification_root, qualification, *, receipt_path=None):
    """One bounded metadata package; never export weights or private work trees."""
    import zipfile
    from .generalization_qualification_io import public_closure
    from .identity import confined, file_hash
    output=Path(output)
    consume_receipt(output,load(receipt_path if receipt_path is not None else output/'acceptance.json'))
    members={name:output/name for name in (
        'acceptance.json','campaign.json','runtime.json','software/suite/acceptance.json','software/suite/cases.json','software/suite/junit.xml',
        'device/input.json','device/reuse.json','device/integrated/producer.json','device/integrated/cases.xml')}
    if receipt_path is not None:
        members['acceptance.json']=receipt_path
    members['qualification/qualification.json']=Path(qualification_root)/'qualification.json'
    for ref in public_closure(qualification_root,qualification):
        members['qualification/'+ref['path']]=confined(qualification_root,ref['path'])
    total=0; inventory=[]
    for name,path in sorted(members.items()):
        if path.is_symlink() or any(part in {'private','checkpoints','inputs'} for part in Path(name).parts) or path.suffix in {'.pt','.pth','.ckpt','.bin'}:
            raise ValueError('private or tensor evidence cannot enter public package')
        total+=path.stat().st_size
        if total>512<<20:
            raise ValueError('complete acceptance metadata exceeds public package cap')
        inventory.append(dict(path=name,bytes=path.stat().st_size,sha256=file_hash(path)))
    manifest=seal(dict(version='progressive-acceptance-package/v1',files=inventory,
                       qualified=True,sustained_status='PENDING',scientific_status='UNMEASURED'))
    import json
    with zipfile.ZipFile(output/'diagnostics.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(members.items()): archive.write(path,name)
        archive.writestr('manifest.json',json.dumps(manifest,sort_keys=True))
    return manifest
