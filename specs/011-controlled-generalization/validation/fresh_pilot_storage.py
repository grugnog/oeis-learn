"""Temporary exact execution projection; preserve immutable preparation evidence."""
from pathlib import Path
import hashlib
from oeis_learn.pipeline_validation.identity import confined,seal
from oeis_learn.pipeline_validation.generalization_preparation import reference


def peak_plan(refs,limits):
    """Bound source view plus the largest serial validation/learner projection.

    Validation projects every execution reference. Learner channels are subsets;
    disposable channels are removed by canonical project_and_invoke on exit.
    Reserve the complete checkpoint/raw allowance in addition to these inputs.
    """
    total=0;cells=0
    for ref in refs:
        path=Path(ref['path']);size=ref['bytes']
        if type(size) is not int or size<0:raise ValueError('invalid projection size')
        if 'checkpoints' in path.parts or 'selected' in path.parts or path.suffix in {'.pt','.pth','.safetensors'}:
            raise ValueError('fresh random projection cannot include checkpoint payloads')
        total+=size
        if 'cells' in path.parts and path.suffix=='.json':cells+=size
    peak=2*cells
    if peak+(16<<20)>limits['shard_bytes']:raise ValueError('execution projection exceeds peak shard reservation')
    study_peak=2*total+limits['checkpoint_bytes']+limits['raw_bytes']+(16<<20)
    if study_peak>limits['study_bytes']:raise ValueError('execution projection exceeds peak study reservation')
    return dict(input_bytes=total,cell_bytes=cells,peak_shard_bytes=peak,
        peak_study_bytes=study_peak,margin_bytes=16<<20,
        policy='exact-execution-view-plus-one-serial-copy/v1')


def project(workspace,manifest,output):
    """Copy only the canonical custodian closure; keep authority bytes unchanged."""
    from oeis_learn.pipeline_validation.generalization_runtime import load,execution_refs
    workspace=Path(workspace);base,study,runtime=load(workspace,manifest)
    if runtime['preparation'] is not None:raise ValueError('resolved preparation required')
    destination=confined(workspace,output,exists=False)
    if destination.exists() or base==destination or base in destination.parents or destination in base.parents:
        raise ValueError('fresh disjoint execution view required')
    refs=execution_refs(base,study,runtime)
    unique={r['path']:r for r in refs}
    for name in ('study.json','runtime.json'):
        ref=reference(base/name,base)
        if name in unique and unique[name]!=ref:raise ValueError('conflicting execution authority')
        unique[name]=ref
    refs=list(unique.values())
    if len(refs)>10000:raise ValueError('execution projection member cap')
    plan=peak_plan(refs,runtime['storage'])
    destination.mkdir(parents=True)
    for ref in refs:
        source=confined(base,ref['path']);target=confined(destination,ref['path'],exists=False)
        target.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256();size=0
        with source.open('rb') as src,target.open('xb') as dst:
            for block in iter(lambda:src.read(1<<20),b''):
                size+=len(block)
                if size>ref['bytes']:raise ValueError('execution input changed during copy')
                h.update(block);dst.write(block)
        if size!=ref['bytes'] or 'sha256:'+h.hexdigest()!=ref['sha256']:
            raise ValueError('execution input changed during copy')
    receipt=seal(dict(version='fresh-pilot-execution-view/v1',
        original_manifest=reference(base/'study.json',workspace),
        manifest=reference(destination/'study.json',workspace),
        runtime=reference(destination/'runtime.json',workspace),
        files=refs,storage=plan,scope='identical canonical inputs; preparation audit tree retained separately'))
    return str((destination/'study.json').relative_to(workspace)),receipt
