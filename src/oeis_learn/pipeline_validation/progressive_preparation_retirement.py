"""Retire only successful diagnostic tensors in a time-blocked preparation.

The canonical evidence closure has no tensor dependency. Calibration checkpoints,
research checkpoints, input corpora and every original metadata file stay intact.
"""
from .identity import confined, digest
from .progressive_workstation_state import read
from .progressive_research_inputs import checked_file, _json
from .progressive_trajectory import clean_stage
from oeis_learn.generalization.progressive.contracts import reference, CAMPAIGN_DOCUMENT_BYTES


def roots(root,folder):
    from .progressive_handoff import original_authority, validate_device_proofs
    base=folder/'prepare-research';journal=read(folder/'journal.json')
    steps=journal['steps']
    if (any(not k.startswith('prepare-research/') for k in steps)
            or any(r.get('status') in {'RUNNING','INTERRUPTED'} or r.get('history') for r in steps.values())
            or steps.get('prepare-research/prepare',{}).get('status')!='BLOCKED'
            or any((base/n).exists() for n in ('frozen','acceptance','construction/execution'))):
        raise ValueError('inactive pre-freeze preparation required for diagnostic retirement')
    prepared=_json(base/'prepared/prepared.json',maximum=CAMPAIGN_DOCUMENT_BYTES)
    deficits=prepared['payload']['deficits']
    if not deficits or any(not x.startswith('training_time:') for x in deficits):
        raise ValueError('original time-only preparation stop required')
    evidence=[reference(root,folder/n) for n in ('session.json','journal.json')]
    evidence.append(reference(root,base/'prepared/prepared.json'))
    qroot=base/'qualification';qref=reference(root,qroot/'qualification.json')
    step=steps.get('prepare-research/qualify',{})
    if step.get('status')!='COMPLETE' or reference(root,qroot/'result.json') not in step.get('outputs',[]):
        raise ValueError('journal-bound completed qualification required')
    result=read(qroot/'result.json')
    if (result.get('mechanical_status')!='COMPLETE' or result.get('qualification_status')!='QUALIFIED'
            or result.get('metrics',{}).get('qualification')!=reference(qroot,'qualification.json')):
        raise ValueError('original completed qualification publication required')
    evidence.append(reference(root,qroot/'result.json'))
    q,qroot=original_authority(root,qref)
    source=digest(_json(checked_file(qroot,q['source_closure'])))
    if source!=read(folder/'session.json')['binding']['source']:
        raise ValueError('original diagnostic qualification source differs')
    matrix,refs=validate_device_proofs(qroot,q,source=source,image=q['images'][1],engine=q['engine_security']['engine'])
    clean_stage(qroot/'device')
    evidence.extend([qref,reference(root,qroot/'device/process.json'),reference(root,qroot/'device/cleanup.json'),
                     reference(root,qroot/'device/container-inspect.json')])
    result=[]
    for ref in refs:
        path=checked_file(qroot,ref)
        if not path.is_relative_to(qroot/'device/cases'):
            raise ValueError('only locally executed qualification cases may be retired')
        result.append(confined(root,path.parent/'checkpoints',exists=False));evidence.append(reference(root,path))
    trajectory=base/'training-trajectory'
    if trajectory.exists():
        from .progressive_runtime_timing import validate_request
        step=steps.get('prepare-research/training-trajectory',{})
        if step.get('status')!='COMPLETE':raise ValueError('completed training-only trajectory required')
        clean_stage(trajectory)
        req=validate_request(read(trajectory/'request.json'));batch=read(trajectory/'runtime-cases.json')
        if (req['source']!=source or req['image']!=q['images'][1] or not req['scope'].startswith('DIAGNOSTIC_ONLY')
                or batch['request']!=req['identity'] or batch['status']!='PASS' or batch['device_block']
                or [r['index'] for r in batch['cases']]!=list(range(len(req['cases'])))):
            raise ValueError('complete original diagnostic trajectory required')
        owned=step['outputs']
        for case,item in zip(req['cases'],batch['cases']):
            path=checked_file(trajectory,item['producer']);p=read(path)
            if (reference(root,path) not in owned or item['status']!='PASS' or p['status']!='PASS'
                    or p['scope']!='DIAGNOSTIC_ONLY' or p['request']!=req['identity'] or p['case']!=case):
                raise ValueError('original diagnostic producer required')
            evidence.append(reference(root,path))
            diagnostics=checked_file(path.parent,p['diagnostics']);evidence.append(reference(root,diagnostics))
            campaign_path=confined(root,path.parent/'private/learner/campaign.json');campaign=read(campaign_path)
            if campaign['purpose']!='acceptance':raise ValueError('diagnostic-only campaign required')
            evidence.append(reference(root,campaign_path))
            from oeis_learn.generalization.progressive.contracts import record
            for task in campaign['payload']['tasks']:
                bound=record('task',dict(task,campaign=campaign['identity']),created_at=campaign['created_at'],
                    purpose=campaign['purpose'],track=campaign['track'])
                result.append(path.parent/'private/learner/learning'/bound['identity'][7:]/'checkpoints')
        evidence.extend(reference(root,trajectory/n) for n in ('request.json','runtime-cases.json','process.json','cleanup.json','container-inspect.json'))
    return result,evidence
