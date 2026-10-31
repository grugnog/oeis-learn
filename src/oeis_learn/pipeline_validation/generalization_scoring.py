"""Selection-only I/O around the existing independent exact scorer."""
import json
import time
from pathlib import Path
from oeis_learn.generalization.contracts import verify_artifact,shape
from .identity import seal,write


def score_frozen(selections,truth,client,*,deadline):
    """Use the canonical scorer for historical controls and the shared bank."""
    from oeis_learn.fresh.bigint_diagnostics import score
    result=score(selections,truth,client,deadline=deadline)
    if any('bank' in row.get('selections',{}) for row in selections.values()):
        projected={key:dict(selections={'equal_budget':row.get('selections',{}).get('bank')},
            arm_evidence={'equal_budget':row.get('arm_evidence',{}).get('bank')}) for key,row in selections.items()}
        bank=score(projected,truth,client,deadline=deadline)
        result['splits']['bank']=bank['splits']['equal_budget']
        result['visible_fit20']['bank']=bank['visible_fit20']['equal_budget']
        for key,outcomes in result['outcomes'].items():outcomes['bank']=bank['outcomes'][key]['equal_budget']
    return result


def score_assessment(truth_root,authorization_root,candidate_root,authorization_ref,*,client,output,seconds):
    """Consume a frozen host authorization after its shared exposure debit.

    No ranking API is available here. The host must use expose_final before
    mounting truth and finish_final after the returned terminal result.
    """
    from oeis_learn.fresh.bigint_state import final_plan
    from oeis_learn.synthesis.state import FinalStudy
    from .identity import validate
    if not 0<seconds<=21600:raise ValueError('bounded assessment duration required')
    roots=[Path(r).resolve() for r in (truth_root,authorization_root,candidate_root)]
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError('disjoint assessment channels required')
    authorization=json.loads(verify_artifact(authorization_root,authorization_ref,max_bytes=1<<20))
    validate(authorization,{'version','plan','exposure','arm','targets','candidates','checkpoint'},'generalization-assessment-authorization/v1')
    plan=final_plan(authorization['plan'])
    state=FinalStudy.restore(authorization['exposure'],plan)
    arm=next((a for a in plan.arms if a[0]==authorization['arm']),None)
    if arm is None or arm[2]!=authorization['checkpoint'] or state.status[arm[0]]!='active' or arm[0] not in state.active:
        raise ValueError('frozen active assessment arm required')
    if plan.cohort!=authorization['targets']['sha256']:
        raise ValueError('assessment cohort differs from authorized plan')
    rows=json.loads(verify_artifact(truth_root,authorization['targets']))
    selections=json.loads(verify_artifact(candidate_root,authorization['candidates']))
    if not isinstance(rows,list) or len(rows)!=plan.denominator or not isinstance(selections,dict):
        raise ValueError('authorized fixed assessment denominator required')
    truth={}
    for row in rows:
        shape('scorer-target',row)
        if row['role']!='assessment' or row['id'] in truth:raise ValueError('unique assessment targets required')
        truth[row['id']]=row
    if set(selections)-set(truth):raise ValueError('candidate outside assessment roster')
    output=Path(output)
    if output.exists():raise ValueError('fresh assessment output required')
    output.mkdir(parents=True,exist_ok=False)
    result=score_frozen(selections,truth,client,deadline=time.monotonic()+seconds)
    record=seal(dict(version='generalization-assessment-scoring/v1',status='PASS',
        scientific_status='UNMEASURED',authorization=authorization_ref,checkpoint=arm[2],result=result))
    write(output/'scoring.json',record)
    return record


def select_checkpoints(truth_root,truth_ref,candidate_root,candidates,*,recipe,client,output,seconds,directories=None):
    """Compare one frozen candidate set under selection supervision only.

    The role receives exact weights and generated selections, never learner
    full-state payloads. Loss uses the existing token-weighted teacher-forced
    measurement; behavior ordering delegates to the Phase B selection policy.
    """
    import math
    from oeis_learn.generalization.contracts import validate
    from oeis_learn.generalization.selection import select
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.fresh.pilot_metrics import measure
    from .generalization_evaluation import load_retained
    if not 0<seconds<=21600:raise ValueError('bounded selection duration required')
    if not isinstance(candidates,list) or not 1<=len(candidates)<=66:
        raise ValueError('bounded frozen checkpoint roster required')
    rows=json.loads(verify_artifact(truth_root,truth_ref))
    rosters={name:{} for name in ('structural','parameter')};seen=set()
    for row in rows:
        shape('scorer-target',row)
        if row['role']!='selection' or row['stratum'] not in rosters or row['id'] in seen:
            raise ValueError('unique selection-only primary splits required')
        seen.add(row['id']);rosters[row['stratum']][row['id']]=row['family']
    if not all(rosters.values()):raise ValueError('both primary selection splits required')
    from .identity import confined
    ids=set();roots={}
    for candidate in candidates:
        if set(candidate)!={'checkpoint','selections'}:raise ValueError('closed checkpoint selection input required')
        checkpoint=validate('checkpoint',candidate['checkpoint']).record()
        if checkpoint['identity'] in ids:raise ValueError('duplicate checkpoint candidate')
        ids.add(checkpoint['identity'])
        key=checkpoint['identity']
        roots[key]=confined(candidate_root,directories[key]) if directories is not None else candidate_root
        verify_artifact(roots[key],checkpoint['weights'])
        verify_artifact(roots[key],candidate['selections'])
    if directories is not None and set(directories)!=ids:raise ValueError('exact candidate directory mapping required')
    output=Path(output)
    if output.exists():raise ValueError('fresh selection output required')
    output.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+seconds
    measured=[];behavior=[];best=None;best_loss=float('inf')
    loss_rows=[dict(values=r['values'][:20],tokens=encode_body(r['source'])) for r in rows]
    for ordinal,candidate in enumerate(sorted(candidates,key=lambda c:(c['checkpoint']['tokens'],c['checkpoint']['identity']))):
        checkpoint=candidate['checkpoint'];remaining=deadline-time.monotonic()
        outcomes={};loss=None
        if remaining>0:
            scoring=score_selection(truth_root,truth_ref,roots[checkpoint['identity']],candidate['selections'],client=client,
                output=output/str(ordinal),seconds=remaining)
            outcomes=scoring['result']['outcomes']
        if time.monotonic()<deadline:
            encoder,decoder,_=load_retained(roots[checkpoint['identity']],checkpoint['weights'],recipe,device='cpu')
            metrics=measure(encoder,decoder,loss_rows,'cpu',deadline=deadline)
            if metrics['complete'] and metrics['loss'] is not None and math.isfinite(metrics['loss']):
                loss=metrics['loss']
            del encoder,decoder
        key=checkpoint['identity']
        if loss is not None and loss<best_loss:best=key;best_loss=loss
        value=dict(id=key,tokens=checkpoint['tokens'],role='selection')
        for split,roster in rosters.items():
            values={}
            for row in roster:
                outcome=outcomes.get(row,{}).get('equal_budget',{})
                values[row]=outcome.get('success') if outcome.get('complete') is True else None
            value[split]=(roster,values)
        behavior.append(value);measured.append(dict(checkpoint=key,loss=loss,behavior=value))
    chosen=select(behavior)
    # An incomplete candidate set cannot support a best-loss comparison.
    if any(row['loss'] is None for row in measured):best=None
    record=seal(dict(version='generalization-checkpoint-selection/v1',status='PASS',
        scientific_status='UNMEASURED',targets=truth_ref,candidates=candidates,
        measurements=measured,best_loss=best,selected=chosen.checkpoint_identity,
        eligible=list(chosen.eligible),reason=chosen.reason))
    write(output/'decision.json',record)
    return record


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
    result=score_frozen(selections,truth,client,deadline=time.monotonic()+seconds)
    # Software completion does not imply positive scientific evidence.
    record=seal(dict(version='generalization-selection-scoring/v1',status='PASS',
        scientific_status='UNMEASURED',targets=truth_ref,candidates=candidate_ref,result=result))
    write(output/'scoring.json',record)
    return record
