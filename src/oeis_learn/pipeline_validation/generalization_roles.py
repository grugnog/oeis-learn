"""Resolve only explicitly authorized role mounts, never a shared input root."""
from pathlib import Path
from oeis_learn.generalization.role_contracts import Role, ROLE_INPUTS, ROLE_OUTPUTS
from .identity import confined


def mounts(root,role,channels,*,output):
    role=Role(role); declarations=ROLE_INPUTS[role]
    if set(channels)!={m.channel for m in declarations}:
        raise ValueError('exact role channels required')
    result=[];paths=[]
    for declaration in declarations:
        path=confined(root,channels[declaration.channel])
        if any(path==other or path in other.parents or other in path.parents for other in paths):
            raise ValueError('overlapping role input roots')
        paths.append(path)
        result.append(dict(source=str(path),destination=declaration.destination,read_only=True))
    output_path=confined(root,output,exists=False)
    if output_path.exists() or any(output_path==p or output_path in p.parents or p in output_path.parents for p in paths):
        raise ValueError('fresh disjoint role output required')
    result.append(dict(source=str(output_path),destination=ROLE_OUTPUTS[role].destination,read_only=False))
    return tuple(result)


def invoke(host,image,root,role,channels,*,output,operation,settings,seconds,gpu=False,launcher=None):
    """Launch one installed role with exact read-only channels and bounded cleanup.

    Settings are public execution metadata only; data and tensor payloads travel
    through declared channels. The checkout and container socket are never mounted.
    """
    import json,uuid
    from contextlib import nullcontext
    from .supervision import launch,reclaim
    if not 0<seconds<=21600:raise ValueError('bounded role duration required')
    role=Role(role)
    if gpu and role not in {Role.LEARNER,Role.GENERATOR}:raise ValueError('GPU forbidden for this role')
    declared=mounts(root,role,channels,output=output)
    for entry in declared[:-1]:
        path=Path(entry['source'])
        if not path.is_dir() or any(p.is_symlink() for p in path.rglob('*')):
            raise ValueError('regular explicit input directory required')
    target=Path(declared[-1]['source']);target.mkdir(parents=True,exist_ok=False)
    name='generalization-'+uuid.uuid4().hex
    runtime=dict(settings=settings,operation=operation,seconds=seconds,device='cuda' if gpu else 'cpu')
    command=host.engine.command(image=image,name=name,
        mounts=[(m['source'],m['destination'],not m['read_only']) for m in declared],
        argv=['python','-m','oeis_learn.pipeline_validation.generalization_role',role.value,json.dumps(runtime,sort_keys=True,allow_nan=False)],
        gpu=gpu,render_node=host.policy.get('render_node') if gpu else None,
        memory_bytes=host.policy['memory_bytes'],cpus=host.policy['cpus'],network=False)
    cleanup={}
    try:
        with host.gpu_lock() if gpu else nullcontext():
            (launcher or launch)(command,target/'supervision',seconds=seconds,stream_cap=1<<20,cleanup_state=cleanup)
    finally:
        reclaim(host.engine,name,target/'supervision',cleanup)
    result=json.loads((target/'result.json').read_text())
    if result.get('status') not in {'PASS','FAIL','BLOCKED','PARTIAL','INCONCLUSIVE','INTERRUPTED'}:
        raise ValueError('role omitted terminal status')
    return result


def project_and_invoke(host,image,root,role,artifacts,*,output,operation,settings,seconds,gpu=False,source_roots=None):
    """Copy the declared verified files into disjoint, read-only channel views."""
    from oeis_learn.generalization.contracts import verify_artifact
    from .identity import confined
    import shutil
    root=Path(root);output=confined(root,output,exists=False)
    if output.exists():raise ValueError('fresh role invocation required')
    output.mkdir(parents=True,exist_ok=False)
    channels={}
    try:
        for channel,refs in artifacts.items():
            if channel not in {m.channel for m in ROLE_INPUTS[Role(role)]}:raise ValueError('undeclared role channel')
            folder=output/'private/input'/channel;folder.mkdir(parents=True)
            for ref in refs:
                source=Path((source_roots or {}).get(channel,root)).resolve()
                source.relative_to(root.resolve())
                payload=verify_artifact(source,ref)
                path=confined(folder,ref['path'],exists=False);path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:stream.write(payload)
            channels[channel]=str(folder.relative_to(root))
        return invoke(host,image,root,role,channels,output=str((output/'role').relative_to(root)),
            operation=operation,settings=settings,seconds=seconds,gpu=gpu)
    finally:
        # Keep operator originals and role outputs; delete only disposable views.
        shutil.rmtree(output/'private/input',ignore_errors=True)
