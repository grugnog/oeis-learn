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
