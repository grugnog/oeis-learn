"""Resolve an explicit session dependency from an original measured E0 selection."""
import json
from pathlib import Path
from .identity import seal,validate,write,confined
from .generalization_preparation import reference
from .generalization_commands import _copy
from .generalization_runtime import execution_refs,initializer_refs
from oeis_learn.generalization.contracts import verify_artifact,artifact_refs
from oeis_learn.generalization.execution import bind
from oeis_learn.generalization.bootstrap import verify_transition
from oeis_learn.generalization.initializers import verify_initializer


def resolve(workspace,output,base,study,runtime,*,session,children,prepared,qualification):
    """No inference/selection here: consume the original preceding case result."""
    source=children['retained-eval'][1]
    if study['model']['initialization']['study']!=source['identity']:raise ValueError('bootstrap source differs from frozen E0')
    stage=output.parent/'retained-eval'
    record=json.loads((stage/'result.json').read_text())
    validate(record,{'version','session','operation','status','result'},'generalization-session-result/v1')
    if record['session']!=session['identity'] or record['operation']!='acceptance' or record['status']!='PASS' or record['result'].get('case')!='retained-eval':
        raise ValueError('original successful retained-evaluation dependency required')
    from oeis_learn.generalization.feature_acceptance import verify_cases
    verify_cases({'retained-eval':reference(stage/'case/receipt.json',output.parent)},qualification=qualification,root=output.parent)
    ref=record['result'].get('initializer')
    if ref is None:raise ValueError('retained evaluation has no eligible initializer')
    original=json.loads(verify_artifact(workspace,ref));origin=confined(workspace,ref['path']).parent
    resolved_source=json.loads(verify_artifact(workspace,prepared['cases']['retained-eval']['manifest']))
    original_source=json.loads(verify_artifact(origin,original['study']))
    if original_source!=resolved_source:raise ValueError('initializer producer differs from prepared E0')
    target=output/'private/resolved'
    refs=execution_refs(base,study,runtime)
    refs.extend(artifact_refs(study))
    if runtime['evaluation'] is not None:refs.extend(artifact_refs(runtime['evaluation']))
    _copy(target,[(base,r) for r in refs])
    initialization=dict(kind='retained',artifact=reference(origin/'initializer.json',origin))
    _copy(target/'initializer',[(origin,r) for r in initializer_refs(origin,initialization)])
    candidate=dict(study,model=dict(study['model'],initialization=dict(kind='retained',artifact=reference(target/'initializer/initializer.json',target))))
    resolved,sidecar=bind(candidate,runtime)
    verify_transition(study,runtime,resolved,sidecar)
    checked=verify_initializer(target,resolved)
    write(target/'study.json',resolved);write(target/'runtime.json',sidecar)
    write(output/'bootstrap.json',seal(dict(version='generalization-session-bootstrap/v1',session=session['identity'],qualification=qualification,
        original_study=study['identity'],resolved_study=resolved['identity'],source_study=resolved_source['identity'],
        initializer=checked['authority'],checkpoint=checked['checkpoint'],selection=original['selection']['sha256'])))
    return target,resolved,sidecar
