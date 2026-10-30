"""Fresh inspection and original-producer I/O; no host Torch/jsonschema imports."""
import json
from pathlib import Path
import platform
import uuid

from oeis_learn.generalization.contracts import artifact_refs,verify_artifact
from .identity import seal,write,file_hash
from .generalization_preparation import reference


def inspect_device(host,image,output,*,seconds=120):
    from .supervision import launch,reclaim
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    name='generalization-inspect-'+uuid.uuid4().hex
    code="""import json,platform,torch
from pathlib import Path
assert torch.version.hip and torch.cuda.is_available(), 'actual HIP required'
p=torch.cuda.get_device_properties(0)
driver=Path('/sys/module/amdgpu/srcversion')
result=dict(runtime=dict(hip=str(torch.version.hip),driver=driver.read_text().strip() if driver.is_file() else platform.release(),python=platform.python_version(),torch=str(torch.__version__)),hardware=dict(hip_available=True,device=p.name,memory_bytes=p.total_memory))
Path('/output/inspection.json').write_text(json.dumps(result,sort_keys=True))
"""
    command=host.engine.command(image=image,name=name,mounts=[(output,'/output',True)],argv=['python','-c',code],
        gpu=True,render_node=host.policy['render_node'],memory_bytes=host.policy['memory_bytes'],cpus=host.policy['cpus'])
    cleanup={}
    try:
        with host.gpu_lock():launch(command,output/'logs',seconds=seconds,stream_cap=1<<20,cleanup_state=cleanup)
    finally:reclaim(host.engine,name,output/'logs',cleanup)
    return json.loads((output/'inspection.json').read_text())


def current_facts(host,root,output,*,images,source_ref,build_refs,producer_root,device_inspector=inspect_device):
    """Inspect the current machine/package; never copy claimed runtime/device facts."""
    from oeis_learn.container_validation import inspect_current_image
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    if len(images)!=2:raise ValueError('qualification and GPU images must be named in that order')
    snapshot=json.loads(verify_artifact(producer_root,source_ref))
    if snapshot!=collect_snapshot(root):raise ValueError('current source closure changed')
    for ref in build_refs:
        verify_artifact(producer_root,ref)
        if not ref['path'].startswith('build/') or file_hash(Path(root)/ref['path'][6:])!=ref['sha256']:
            raise ValueError('build input changed')
    for position,image in enumerate(images):
        if host.engine.image(image)!=image:raise ValueError('immutable image changed')
        inspect_current_image(host,image,Path(output)/('image-'+str(position)),seconds=120)
    host.select_gpu()
    device=device_inspector(host,images[1],Path(output)/'device')
    engine=host.engine.facts()
    resources=dict(memory_bytes=host.policy['memory_bytes'],cpus=host.policy['cpus'],torch_intraop=2,torch_interop=1,native_threads=1,learners=1)
    security=dict(engine=engine['engine'],runtime_socket=False,engine_version=engine['version'],runtime=engine['runtime'],
        rootless=engine['rootless'],selinux=engine['selinux'],network=False,read_only_root=True,no_new_privileges=True,capabilities='none')
    return dict(source_closure=source_ref,build_dependencies=build_refs,images=images,
        runtime=device['runtime'],hardware=device['hardware'],precision='fp32',resources=resources,engine_security=security)


def public_closure(root,authority):
    """Collect only the transitive hash-bound original proof, rejecting conflicts."""
    pending=list(artifact_refs(authority));refs={}
    while pending:
        ref=pending.pop()
        if ref['path'] in refs:
            if refs[ref['path']]!=ref:raise ValueError('conflicting qualification artifact')
            continue
        payload=verify_artifact(root,ref,max_bytes=512<<20);refs[ref['path']]=ref
        try:child=json.loads(payload)
        except (ValueError,UnicodeError):continue
        pending.extend(artifact_refs(child))
        if sum(p['bytes'] for p in refs.values())>512<<20:raise ValueError('qualification sharing cap')
    return list(refs.values())


def consume_original(host,workspace,producer_root,output,*,required_suites,required_capabilities,research=False,feature_receipt=None):
    """Consumers independently inspect and validate the same original producer."""
    from .generalization_roles import project_and_invoke
    producer_root=Path(producer_root);output=Path(output)
    path=producer_root/'qualification.json';authority=json.loads(path.read_text())
    if seal(authority)!=authority or authority.get('version')!='generalization-qualification/v2':
        raise ValueError('original v2 qualification authority required')
    facts=current_facts(host,workspace,output/'inspection',images=authority['images'],source_ref=authority['source_closure'],
        build_refs=authority['build_dependencies'],producer_root=producer_root)
    refs=[reference(path,producer_root),*public_closure(producer_root,authority)]
    if feature_receipt is not None:
        verify_artifact(producer_root,feature_receipt);refs.append(feature_receipt)
    result=project_and_invoke(host,authority['images'][0],workspace,'report',{'public_evidence':refs},
        source_roots={'public_evidence':producer_root},output=str((output/'verification').relative_to(workspace)),
        operation='qualification-consume',settings=dict(authority=refs[0],current=facts,required_suites=list(required_suites),
            required_capabilities=list(required_capabilities),research=research,feature_receipt=feature_receipt),seconds=120)
    if result['status']!='PASS':raise ValueError('INCOMPATIBLE_QUALIFICATION: '+result.get('error',result.get('reason','verification failed')))
    return dict(original=result['original'],authority=authority,root=producer_root)


def produce(host,workspace,output,*,images,contract):
    """Canonical runner executes each required CPU/native and HIP scope once."""
    from .registry import Stage
    from .runner import execute
    from oeis_learn.container_validation import _stages,inspect_current_image
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from .generalization_roles import project_and_invoke
    workspace=Path(workspace);output=Path(output)
    if output.exists():raise ValueError('fresh qualification output required')
    if len(images)!=2:raise ValueError('prebuilt qualification/GPU image pair required')
    output.mkdir(parents=True,exist_ok=False)
    class Adapter:
        def __call__(self,stage,target):
            if stage.adapter=='inspect':
                image=images[0 if stage.id=='image.qualification' else 1]
                if host.engine.image(image)!=image:raise ValueError('immutable image required')
                inspect_current_image(host,image,target,seconds=stage.seconds)
                return dict(image=image)
            gpu=stage.adapter=='hip';host.image=images[1 if gpu else 0]
            if gpu:
                host.select_gpu()
                with host.gpu_lock():result=_stages(host,'bigint-gpu',target,external_seconds=stage.seconds)[1]
            else:result=_stages(host,'bigint-cpu',target,external_seconds=stage.seconds)[1]
            return dict(suite='bigint-gpu' if gpu else 'bigint-cpu',result=result)
    stages=[Stage('image.qualification',1,seconds=240,adapter='inspect'),
        Stage('cpu.semantic',1,('image.qualification',),seconds=3300,adapter='cpu'),
        Stage('image.gpu',2,seconds=240,adapter='inspect'),
        Stage('hip.preflight',2,('image.gpu','cpu.semantic'),seconds=2400,adapter='hip')]
    code=execute(stages,output/'pipeline',adapter=Adapter(),binding=dict(source=collect_snapshot(workspace),contract=contract),
                 cap=6600,reserve=300)
    if code:return dict(status='FAIL',reason='QUALIFICATION_FAILED',exit_code=code)
    write(output/'source.json',collect_snapshot(workspace));source_ref=reference(output/'source.json',output)
    names=['Dockerfile','pyproject.toml','uv.lock']
    names.extend(str(p.relative_to(workspace)) for p in sorted((workspace/'crates').rglob('Cargo.lock')))
    builds=[]
    for name in names:
        source=workspace/name
        if not source.is_file():raise ValueError('missing pinned build input: '+name)
        target=output/'build'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        builds.append(reference(target,output))
    facts=current_facts(host,workspace,output/'current',images=images,source_ref=source_ref,build_refs=builds,producer_root=output)
    evidence=list((output/'pipeline/stages/hip.preflight').rglob('generalization-device.json'))
    if len(evidence)!=1:raise ValueError('missing unique actual load/update proof')
    device_ref=reference(evidence[0],output);dependencies=[]
    for stage in stages:
        path=output/'pipeline/stages'/stage.id/'receipt.json'
        receipt=json.loads(path.read_text())
        if receipt['state']!='PASS':raise ValueError('original PASS stage required')
        files=[reference(path.parent/name,output) for name in receipt['files']]
        proof=seal(dict(version='generalization-original-stage/v1',status='PASS',stage=stage.id,
            receipt=reference(path,output),files=files))
        path=output/'originals'/(stage.id+'.json');write(path,proof);dependencies.append(reference(path,output))
    producer=seal(dict(version='generalization-producer/v1',status='PASS',facts=facts,suites=['bigint-cpu','bigint-gpu'],
        capabilities=['weight-load','finite-update','native-exact'],artifacts=[device_ref],dependencies=dependencies,feature_acceptance=False))
    path=output/'producer.json';write(path,producer);producer_ref=reference(path,output)
    authority=seal(dict(version='generalization-qualification/v2',suite_selection=producer['suites'],covered_capabilities=producer['capabilities'],
        **facts,producer=dict(receipt=producer_ref,status='PASS',artifacts=[device_ref]),artifact_inventory=public_closure(output,producer),status='PASS'))
    path=output/'qualification.json';write(path,authority)
    refs=[reference(path,output),*public_closure(output,authority)]
    validated=project_and_invoke(host,images[0],workspace,'report',{'public_evidence':refs},source_roots={'public_evidence':output},
        output=str((output/'authority-validation').relative_to(workspace)),operation='qualification-consume',
        settings=dict(authority=refs[0],current=facts,required_suites=producer['suites'],required_capabilities=producer['capabilities'],research=False,feature_receipt=None),seconds=120)
    if validated['status']!='PASS':raise ValueError('qualification authority validation failed')
    return dict(status='PASS',original=producer['identity'],qualification=reference(path,output))
