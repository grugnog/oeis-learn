"""Selection-only I/O around the existing independent exact scorer."""
import json
import time
from pathlib import Path
from oeis_learn.generalization.contracts import verify_artifact,shape
from .identity import seal,write


def score_selection(truth_root,truth_ref,candidate_root,candidate_ref,*,client,output,seconds):
    from oeis_learn.fresh.bigint_diagnostics import score
    if not 0<seconds<=21600:raise ValueError('bounded scoring duration required')
    truth_root=Path(truth_root).resolve();candidate_root=Path(candidate_root).resolve()
    if truth_root==candidate_root or truth_root in candidate_root.parents or candidate_root in truth_root.parents:
        raise ValueError('disjoint scoring inputs required')
    rows=json.loads(verify_artifact(truth_root,truth_ref))
    selections=json.loads(verify_artifact(candidate_root,candidate_ref))
    if not isinstance(rows,list) or not rows or not isinstance(selections,dict):
        raise ValueError('nonempty selection roster and candidate mapping required')
    truth={}
    for row in rows:
        shape('scorer-target',row)
        if row['role']!='selection':raise ValueError('selection supervision required')
        if row['id'] in truth:raise ValueError('duplicate selection target')
        truth[row['id']]=row
    if set(selections)-set(truth):raise ValueError('candidate outside selection roster')
    output=Path(output)
    if output.exists():raise ValueError('fresh scoring output required')
    output.mkdir(parents=True,exist_ok=False)
    result=score(selections,truth,client,deadline=time.monotonic()+seconds)
    # Software completion does not imply positive scientific evidence.
    record=seal(dict(version='generalization-selection-scoring/v1',status='PASS',
        scientific_status='UNMEASURED',targets=truth_ref,candidates=candidate_ref,result=result))
    write(output/'scoring.json',record)
    return record
