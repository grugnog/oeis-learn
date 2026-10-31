"""Publish selection-designated retained initialization from actual learner origins."""
import json
from pathlib import Path
from .identity import seal,write
from .generalization_preparation import reference
from .generalization_runtime import initializer_refs
from oeis_learn.generalization.contracts import verify_artifact


def remember_origin(base,checkpoint_root,study,arm):
    from .generalization_commands import _copy
    _copy(checkpoint_root,[(base,r) for r in initializer_refs(base,study['model']['initialization'])])
    write(checkpoint_root/'origin-study.json',study)
    write(checkpoint_root/'origin.json',seal(dict(version='generalization-learner-origin/v1',study=reference(checkpoint_root/'origin-study.json',checkpoint_root),arm=arm)))


def publish_role(inputs,output,settings,*,seconds):
    from oeis_learn.generalization.contracts import validate
    from oeis_learn.generalization.initializers import _selection,verify_initializer
    from .generalization_commands import _copy
    from .receipts import Ledger
    if set(settings)!={'origin','checkpoint','selection'}:raise ValueError('closed initializer publication inputs required')
    root=inputs['mechanical_sources']
    origin=json.loads(verify_artifact(root,settings['origin']))
    if seal(origin)!=origin or origin.get('version')!='generalization-learner-origin/v1':raise ValueError('original learner binding required')
    study=validate('study',json.loads(verify_artifact(root,origin['study']))).record()
    descriptor=validate('checkpoint',json.loads(verify_artifact(root,settings['checkpoint']))).record()
    decision=json.loads(verify_artifact(root,settings['selection']))
    if study['data']['target_format']!='literal' or origin['arm']=='parameterized' or origin['arm'] not in study['arms']:
        raise ValueError('literal mechanical learner origin required')
    _selection(decision,descriptor)
    prior=verify_initializer(root,study)
    ancestry=['mechanical','training','random_initialization']
    if study['progression']['stage']=='P4' or prior and any('verified_training_replay' in p['authority']['ancestry'] for p in prior['lineage']):ancestry.append('verified_training_replay')
    target=output/'private/initializer'
    refs=[origin['study'],settings['checkpoint'],settings['selection'],descriptor['weights'],*initializer_refs(root,study['model']['initialization'])]
    _copy(target,[(root,r) for r in refs])
    binding=dict(study=study['identity'],checkpoint=descriptor['identity'],ancestry=ancestry,parent=prior['authority'] if prior else None)
    ledger=Ledger(target,'generalization.initializer',binding);ledger.transition('READY');ledger.transition('RUNNING')
    ledger.transition('PASS',files={r['path']:r['sha256'] for r in refs})
    authority=seal(dict(version='generalization-initializer/v1',study=origin['study'],checkpoint=settings['checkpoint'],selection=settings['selection'],
        producer=reference(target/'receipt.json',target),ancestry=ancestry,parent=study['model']['initialization']['artifact'] if prior else None))
    write(target/'initializer.json',authority)
    artifact=reference(target/'initializer.json',target)
    expected=seal(dict(study,version='generalization-study/v3' if study['version']=='generalization-study/v3' else 'generalization-study/v2',model=dict(study['model'],initialization=dict(kind='retained',artifact=artifact))))
    verify_initializer(target,expected)
    return dict(status='PASS',authority=authority['identity'],checkpoint=descriptor['identity'],artifact=reference(target/'initializer.json',output))


def publish_selected(host,workspace,output,base,study,runtime,decision,available,*,seconds=60):
    from .generalization_commands import _copy
    from .generalization_roles import project_and_invoke
    if decision is None or decision.get('selected') is None:return dict(status='UNAVAILABLE',reason='NO_ELIGIBLE_CHECKPOINT')
    descriptor,parent=next((c,r) for c,r in available if c['identity']==decision['selected'])
    if not (parent/'origin.json').is_file():return dict(status='UNAVAILABLE',reason='ORIGINAL_LEARNER_METADATA_ABSENT')
    origin=json.loads((parent/'origin.json').read_text())
    origin_study=json.loads(verify_artifact(parent,origin['study']))
    if origin['arm']=='parameterized' or origin_study['data']['target_format']!='literal':return dict(status='UNAVAILABLE',reason='NON_LITERAL_INITIALIZER')
    inputs=output/'private/input';inputs.mkdir(parents=True)
    refs=[reference(parent/'origin.json',parent),origin['study'],descriptor['weights'],*initializer_refs(parent,origin_study['model']['initialization'])]
    _copy(inputs,[(parent,r) for r in refs])
    write(inputs/'selected-checkpoint.json',descriptor);write(inputs/'selection-decision.json',decision)
    checkpoint=reference(inputs/'selected-checkpoint.json',inputs);selection=reference(inputs/'selection-decision.json',inputs)
    refs.extend([checkpoint,selection])
    return project_and_invoke(host,runtime['images'][0],workspace,'preparation',{'mechanical_sources':refs,'isolation_truth':[]},
        source_roots={'mechanical_sources':inputs,'isolation_truth':inputs},output=str((output/'publication').relative_to(workspace)),
        operation='publish-initializer',settings=dict(origin=reference(inputs/'origin.json',inputs),checkpoint=checkpoint,selection=selection),seconds=seconds)


def persist_selected(host,workspace,output,runtime,decision,available,*,seconds):
    """Only sanitized selection identities cross back into the learner role."""
    from .generalization_commands import _resume_inputs,_copy
    from .generalization_roles import project_and_invoke
    wanted={decision[k] for k in ('selected','best_loss') if decision.get(k) is not None}
    parents={parent for descriptor,parent in available if descriptor['identity'] in wanted}
    results=[]
    for ordinal,parent in enumerate(sorted(parents)):
        if not (parent/'retained.json').is_file() or not (parent/'generalization-latest.json').is_file():
            results.append(dict(status='UNAVAILABLE',reason='FULL_STATE_ABSENT'));continue
        roster_ref=reference(parent/'retained.json',parent);roster=json.loads(verify_artifact(parent,roster_ref));ids={c['identity'] for c in roster['checkpoints']}
        pointer=json.loads((parent/'generalization-latest.json').read_text())
        original,_,refs=_resume_inputs(workspace,str((parent/pointer['path']).relative_to(workspace)))
        state=json.loads(verify_artifact(parent,pointer))
        folder=output/str(ordinal);inputs=folder/'private/input'
        _copy(inputs,[(parent,r) for r in [*refs,roster_ref]])
        pins=seal(dict(version='generalization-selected-pins/v1',roster=roster['identity'],decision=decision['identity'],
            best_loss=decision['best_loss'] if decision['best_loss'] in ids else None,selected=decision['selected'] if decision['selected'] in ids else None,reason=decision['reason']))
        write(inputs/'selection-pins.json',pins);pins_ref=reference(inputs/'selection-pins.json',inputs)
        result=project_and_invoke(host,runtime['images'][0],workspace,'learner',{'training_targets':[],'verified_training_replay':[*refs,roster_ref,pins_ref]},
            source_roots={'training_targets':inputs,'verified_training_replay':inputs},output=str((folder/'persistence').relative_to(workspace)),
            operation='pin-selection',settings=dict(roster=roster_ref,decision=pins_ref,compatibility=state['contract'],target_tokens=state['ledger']['cap']),seconds=seconds/max(1,len(parents)))
        results.append(dict(result,output=str((folder/'persistence/role').relative_to(workspace))))
    return results
