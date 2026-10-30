"""Bounded materialization and immutable per-cell publication adapters."""
from pathlib import Path
import time
from oeis_learn.data.bigint_programs import materialize
from oeis_learn.generalization.coverage import observe
from oeis_learn.generalization.cohorts import choose,project,publish
from oeis_learn.generalization.exclusions import ExclusionIndex
from oeis_learn.generalization.contracts import digest
from .identity import write,seal,file_hash


def reference(path,root):
    return dict(path=str(path.relative_to(root)),sha256=file_hash(path),bytes=path.stat().st_size)


def prepare_cell(candidates,*,role,holdout,count,quotas,structural,output,execute,
                 exclusions=None,draw_cap=1_000_000,seconds=14400,length_boundary=32):
    """execute is the qualified native client's externally deadline-bounded adapter."""
    if role not in {'training','selection','assessment'} or holdout not in {'parameter','structural'}:
        raise ValueError('invalid cohort role/holdout')
    if not 0<seconds<=14400 or not 1<=draw_cap<=1_000_000: raise ValueError('preparation limits')
    output=Path(output)
    if output.exists(): raise ValueError('fresh preparation output required')
    output.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+seconds;admitted=[];rejections=[];seen=0
    exclusions=exclusions if exclusions is not None else ExclusionIndex()
    for row in candidates:
        if seen>=draw_cap or time.monotonic()>=deadline: break
        seen+=1
        try:
            if row['origin']!='generic_random': raise ValueError('nonmechanical origin')
            exact=materialize(row)
            native=execute(exact['source'],tuple(exact['indices']))
            if native.get('outcome')!='ok' or tuple(native['values'])!=tuple(exact['values']): raise ValueError('native/reference disagreement')
            observation=observe(exact['source'],tuple(exact['indices']),stage=row['stage'])
            tags=[row['stage'],'long' if observation.nodes>=length_boundary else 'short','above64' if observation.above_64_bits else 'within64']
            if observation.loop_active:tags.append('loop_active')
            if observation.branch_active:tags.append('branch_active')
            exclusion=dict(id=exact['id'],role=role,family=row['family'],holdout=holdout,source=exact['source'],values=exact['values'],group=row['group'],variant=row['variant'])
            if exclusions.reasons(exclusion): raise ValueError('cross-role exclusion')
            admitted.append(dict(exact,role=role,tags=tags))
        except ValueError as exc:
            rejections.append(dict(draw=seen,reason=str(exc)))
    choice=choose(admitted,count=count,quotas=quotas,structural=structural,draw_cap=draw_cap)
    ledger=seal(dict(version='generalization-cell-preparation/v1',role=role,draws=seen,rejections=rejections,shortfalls=list(choice.shortfalls),status='PASS' if choice.complete else 'BLOCKED'))
    write(output/'ledger.json',ledger)
    if not choice.complete:return ledger
    visible=[];protected=[]
    for row in choice.rows:
        # Hash-based opaque id contains no protected labels.
        row=dict(row,id=digest([role,row['id'],row['source']]))
        prompt,target=project(row);visible.append(prompt);protected.append(target)
    write(output/'prompts.json',visible)
    write(output/'private/targets.json',protected)
    write(output/'inventory.json',seal(dict(version='generalization-cell-files/v1',files=[reference(output/name,output) for name in ('ledger.json','prompts.json','private/targets.json')])))
    return ledger


def publish_study(choices,preparation,proposed,manifests,*,root,output):
    """Resolve only after all cells and exclusion commitments pass Phase B."""
    resolved=publish(choices,preparation,proposed,manifests,root=root)
    write(Path(output)/'study.json',resolved.record())
    return resolved
