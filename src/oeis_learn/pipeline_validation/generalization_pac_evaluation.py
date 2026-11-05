"""PAC structural delivery over the existing generator, fitter and exact scorer."""
import json
from pathlib import Path
from .identity import seal,digest,write,confined
from .generalization_preparation import reference,verify_pac_structural
from oeis_learn.generalization.contracts import verify_artifact,shape


def publish_views(inputs,output,settings,*,seconds):
    if set(settings)!={'admission'}:raise ValueError('closed PAC publication request')
    root=inputs['mechanical_sources']
    admission=json.loads(verify_artifact(root,settings['admission'],max_bytes=16<<20))
    if seal(admission)!=admission:raise ValueError('admission seal changed')
    rows=verify_pac_structural(root,admission);prompts=[];targets=[]
    from oeis_learn.generalization.cohorts import project
    for row in rows:
        key=digest(['selection',row['id'],row['source']])
        visible,truth=project(dict(row,id=key,role='selection',holdout='structural'))
        prompts.append(visible);targets.append(truth)
    output=Path(output)
    write(output/'visible/prompts.json',prompts);write(output/'private/targets.json',targets)
    write(output/'roster.json',admission['roster'])
    return dict(status='PASS',prompts=reference(output/'visible/prompts.json',output),
        targets=reference(output/'private/targets.json',output),roster=reference(output/'roster.json',output),admission=admission['identity'])


def checkpoint_request(workspace,request,study,runtime,recipe,prepared):
    """Explicit arm=descriptor path; latest new weights or frozen retained A only."""
    from .generalization_runtime import learner_config
    from .generalization_execution import bind_compatibility
    try:arm,path=request.split('=',1)
    except ValueError:raise ValueError('PAC checkpoint requires arm=descriptor path') from None
    if arm not in set(study['arms'])|{'A'}:raise ValueError('unfrozen evaluation arm')
    file=confined(workspace,path)
    if file.stat().st_size>1<<20:raise ValueError('checkpoint descriptor cap')
    descriptor=json.loads(file.read_text());shape('checkpoint',descriptor)
    if seal(descriptor)!=descriptor:raise ValueError('checkpoint seal changed')
    if arm=='A' and study['control_mode']!='fresh-control':
        if descriptor['identity']!=runtime['control']['checkpoint']:raise ValueError('retained A differs from frozen control')
        if descriptor.get('version')=='generalization-checkpoint/v2':
            from oeis_learn.generalization.legacy import verify_legacy
            verify_legacy(file.parent,descriptor)
    else:
        config=learner_config(study,runtime,recipe,arm,study['seed'],prepared=prepared)
        data=digest(dict(prepared=prepared['identity'],arm=arm,batch_size=config['batch_size']))
        expected=digest(dict(config=config,data=data,**bind_compatibility('cuda',{})))
        if descriptor['kind']!='full_state' or descriptor['compatibility']!=expected or 'latest' not in descriptor['pins']:
            raise ValueError('new arm requires latest full-state descriptor from frozen learner')
        if not 0<descriptor['tokens']<=study['target_tokens']:raise ValueError('invalid committed exposure')
        verify_artifact(file.parent,descriptor['state'],max_bytes=16<<20)
    verify_artifact(file.parent,descriptor['weights'],max_bytes=32<<30)
    return arm,file.parent,descriptor


def score_views(inputs,output,settings,*,seconds):
    import time
    from oeis_learn.synthesis.client import SynthesisClient
    from .generalization_scoring import score_frozen
    from oeis_learn.generalization.metrics import pac_structural_metrics
    if set(settings)!={'targets','candidates','roster','study'}:raise ValueError('closed PAC score request')
    truth=json.loads(verify_artifact(inputs['selection_supervision'],settings['targets']))
    roster=json.loads(verify_artifact(inputs['selection_supervision'],settings['roster']))
    for row in truth:
        shape('scorer-target',row)
        if row['role']!='selection' or row['stratum']!='structural':raise ValueError('PAC structural selection only')
    if {r['id'] for r in truth}!=set(roster):raise ValueError('truth/roster mismatch')
    candidates=json.loads(verify_artifact(inputs['selection_candidates'],settings['candidates']))
    with SynthesisClient(settings['study'],cache_root=Path(output)/'private/native') as client:
        scores=score_frozen(candidates,{r['id']:r for r in truth},client,deadline=time.monotonic()+seconds)
    controls={control:{key:(values[control]['success'] if values[control]['complete'] else None) for key,values in scores['outcomes'].items() if control in values}
              for control in {c for values in scores['outcomes'].values() for c in values}}
    result=seal(dict(version='pac-structural-scoring/v1',status='PASS',study=settings['study'],
        controls={c:pac_structural_metrics(roster,v) for c,v in controls.items()},outcomes=controls,scored_cases=scores['outcomes'],roster=roster,
        parameter='DEFERRED',assessment='NOT_EXECUTED'))
    write(Path(output)/'scoring.json',result);return result


def evaluate(host,workspace,output,base,study,runtime,plan,*,checkpoints,bank=None):
    from .generalization_roles import project_and_invoke
    admission_ref=runtime['prepared']['admission'];admission=json.loads(verify_artifact(base,admission_ref))
    if runtime['purpose']!='research':raise ValueError('structural evaluation requires original research admission')
    if bank is None:bank=digest(list(range(20261101,20261109)))
    if bank!=digest(list(range(20261101,20261109))):raise ValueError('frozen bank seeds required')
    available=[checkpoint_request(workspace,c,study,runtime,plan['recipe'],plan['training']) for c in checkpoints]
    if not available or len({a for a,_,_ in available})!=len(available):raise ValueError('explicit unique arm checkpoints required')
    output=Path(output)
    limits=runtime['envelope']['validation']
    publication=project_and_invoke(host,runtime['images'][0],workspace,'preparation',
        {'mechanical_sources':[admission_ref,admission['structural']],'isolation_truth':[]},source_roots={'mechanical_sources':base},
        output=str((output/'views').relative_to(workspace)),operation='pac-publish-evaluation',settings=dict(admission=admission_ref),
        seconds=limits['outer'],envelope=limits)
    if publication['status']!='PASS':return publication
    views=output/'views/role';runs=[]
    for arm,root,descriptor in available:
        limits=runtime['envelope']['generation'];folder=output/arm
        generated=project_and_invoke(host,runtime['images'][1],workspace,'generator',
            {'visible_prompts':[publication['prompts']],'frozen_weights':[descriptor['weights']]},
            source_roots={'visible_prompts':views,'frozen_weights':root},output=str((folder/'generation').relative_to(workspace)),
            operation='pac-evaluate',settings=dict(checkpoint=descriptor['weights'],prompts=publication['prompts'],recipe=plan['recipe'],
                study=study['identity'],source=descriptor['identity'],device='cuda',bank_seeds=list(range(20261101,20261109)) if bank else None),
            seconds=limits['outer'],envelope=limits,gpu=True)
        if generated['status'] not in {'PASS','PARTIAL'}:return dict(status=generated['status'],runs=runs,failed_arm=arm)
        gen=folder/'generation/role/generation';candidates=reference(gen/'selections.json',gen)
        limits=runtime['envelope']['proof']
        scored=project_and_invoke(host,runtime['images'][0],workspace,'selection',
            {'selection_supervision':[publication['targets'],publication['roster']],'selection_candidates':[candidates]},
            source_roots={'selection_supervision':views,'selection_candidates':gen},output=str((folder/'scoring').relative_to(workspace)),
            operation='pac-score',settings=dict(targets=publication['targets'],candidates=candidates,roster=publication['roster'],study=study['identity']),
            seconds=limits['outer'],envelope=limits)
        runs.append(dict(arm=arm,checkpoint=descriptor['identity'],generation=generated,scoring=scored))
    return seal(dict(status='PASS' if all(r['generation']['status']==r['scoring']['status']=='PASS' for r in runs) else 'PARTIAL',runs=runs))
