"""Recover clean pre-science time or original-reservation reuse failures.

Historical replay authenticates evidence; it never supplies current qualification.
The next owner still points directly to the first complete calibration donor.
"""
from .identity import confined, seal
from .progressive_workstation_state import Blocked, read
from .progressive_research_inputs import checked_file, _json
from .progressive_process_result import stdout_result
from oeis_learn.generalization.progressive.contracts import reference, CAMPAIGN_DOCUMENT_BYTES, validate


def proof(root, receipt, session, journal):
    from .progressive_preparation_owner import EARLY
    from .progressive_construction_recovery import prerequisites
    from .progressive_calibration_reuse import validate_prepared, named, source_authority
    from .progressive_trajectory import clean_stage
    folder=confined(root,receipt['recipient']);base=folder/'prepare-research'
    allowed=EARLY|{'base-qualification','matrix','qualification-plan','qualify',
        'evaluation-plan','evaluation-timing','evaluation-feasibility','construction',
        'trajectory-plan','training-trajectory','calibration-reuse-plan','prepare'}
    for key,row in journal['steps'].items():
        name=key.split('/',1)[-1]
        if not key.startswith('prepare-research/'):
            raise Blocked('scientific work cannot transfer preparation ownership')
        for attempt in [row,*row.get('history',[])]:
            if attempt['status'] in {'RUNNING','INTERRUPTED'} or (name not in allowed and attempt['status']!='BLOCKED'):
                raise Blocked('downstream or unresolved work cannot transfer preparation ownership')
    step=journal['steps'].get('prepare-research/prepare',{})
    failed = step.get('status') == 'FAILED'
    if step.get('status') not in {'BLOCKED','FAILED'} or step.get('history'):
        raise Blocked('one original clean stopped prepare required')
    if any((base/p).exists() for p in ('frozen','acceptance','construction/execution')):
        raise Blocked('unstarted scientific campaign required')
    req,contract=prerequisites(root,receipt,journal)
    owned={}
    for name in ('construction','trajectory-plan','training-trajectory','calibration-reuse-plan'):
        row=journal['steps'].get('prepare-research/'+name,{})
        if row.get('status')!='COMPLETE' or row.get('history'):
            raise Blocked('one complete original preparation stage required: '+name)
        for ref in row.get('outputs',[]):checked_file(root,ref);owned[ref['path']]=ref
    blueprint_path=base/'calibration-reuse/blueprint.json'
    if owned.get(str(blueprint_path.relative_to(root)))!=reference(root,blueprint_path):
        raise Blocked('original recovery blueprint must be journal-bound')
    blueprint=validate('blueprint',_json(blueprint_path,maximum=CAMPAIGN_DOCUMENT_BYTES))
    extra = {}
    if failed:
        from .progressive_calibration_reuse import raw_prepared
        prepared = raw_prepared(root, blueprint)
        extra = reservation_failure(root, base, blueprint, prepared)
    else:
        prepared=validate('prepared',_json(base/'prepared/prepared.json',maximum=CAMPAIGN_DOCUMENT_BYTES))
        deficits=prepared['payload']['deficits']
        if not deficits or any(not d.startswith('training_time:') for d in deficits):
            raise Blocked('only a training-time deficit can transfer original calibration')
        # Replay historical evidence without granting current-source authority.
        validate_prepared(root,blueprint,prepared,historical=True)
    reuse=_json(checked_file(root,named(prepared,'calibration_reuse')))
    source_authority(root,reuse,req['qualification'])
    donor=confined(root,reuse['blueprint']['path']).parent.parent.parent
    if str(donor.relative_to(root))!=session['binding'].get('reuse_calibration_session'):
        raise Blocked('original complete calibration donor differs')
    clean_stage(base/'training-trajectory')
    allowance=read(base/'training-trajectory/trajectory-allowance.json')
    trajectory=read(base/'training-trajectory/request.json')
    if (allowance.get('active') is not None or allowance.get('binding')!=trajectory['identity']
            or allowance.get('seconds')!=trajectory['seconds']
            or not allowance.get('attempts')
            or any(a.get('outcome')!='COMPLETE' for a in allowance['attempts'])
            or not 0<allowance['spent_seconds']<=allowance['seconds']):
        raise Blocked('original closed trajectory allowance required')
    binding=seal(dict(operation='prepare',blueprint=blueprint['identity'],qualification=req['qualification']))
    owner=read(base/'prepared/owner.json')
    if (read(base/'prepared/operation-binding.json')!=binding or owner.get('version')!='discovery-owner/v1'
            or owner.get('output')!=str((base/'prepared').relative_to(root))):
        raise Blocked('original reused preparation owner/binding differs')
    result=read(base/'prepared/result.json')
    if not failed and (result.get('mechanical_status')!='PARTIAL' or result.get('reason')!='BOUND_OUTCOME_BLIND_PREPARATION'
            or result.get('operation')!='prepare' or result.get('producer_error') is not None
            or result.get('scientific_status')!='UNMEASURED' or result.get('missing_evidence')!=deficits
            or result.get('metrics',{}).get('prepared')!=reference(base/'prepared',base/'prepared/prepared.json')):
        raise Blocked('original time-only preparation result required')
    logs=base/'logs/prepare'
    if sorted(p.name for p in logs.iterdir())!=['0000']:
        raise Blocked('one original canonical prepare invocation required')
    logs=logs/'0000';process=_json(logs/'process.json');command=_json(logs/'command.json')
    b=session['binding']
    from .progressive_workstation import command as canonical_command
    from types import SimpleNamespace
    expected=canonical_command(root,SimpleNamespace(engine=b['engine'],memory_gib=b['memory_gib'],
        cpus=b['cpus'],render_node=b.get('render_node')),'prepare',
        blueprint=str(blueprint_path.relative_to(root)),qualification=str((base/'qualification').relative_to(root)),
        output=str((base/'prepared').relative_to(root)))
    if (command['argv'][1:]!=expected[1:] or process.get('returncode')!=(3 if failed else 2) or process.get('cleanup_failure')
            or process.get('failure') or process.get('outcome')!='COMPLETE'
            or process.get('omitted_bytes',{}).get('stdout',0)!=0 or stdout_result(logs/'stdout.log')!=result):
        raise Blocked('clean original time-blocked canonical prepare required')
    return seal(dict(version='progressive-preparation-owner-proof/v7' if failed else 'progressive-preparation-owner-proof/v6',predecessor=receipt,
        session=session,journal=journal,qualification=req['qualification'],execution_contract=contract,
        blueprint=reference(root,blueprint_path),prepared=None if failed else reference(root,base/'prepared/prepared.json'),
        calibration_reuse=named(prepared,'calibration_reuse'),
        trajectory_allowance=reference(root,base/'training-trajectory/trajectory-allowance.json'),
        result=reference(root,base/'prepared/result.json'),process=reference(root,logs/'process.json'),
        reason='original calibration reservation mismatch before publication' if failed else 'complete reused calibration and trajectory; pre-freeze training-time deficit only',
        original_costs_retained=True,scientific_allowance_transferred=False,
        historical_qualification_only=True,current_source_requalification_required=True,**extra))


def reservation_failure(root, base, blueprint, prepared):
    """Admit only the reproduced debit mismatch after terminated inspections."""
    import re
    from .progressive_calibration_reuse import named, bind_original_reservation, raw_prepared, assemble
    from .progressive_handoff import named_plan
    output=base/'prepared';result=read(output/'result.json');error=result.get('producer_error') or {}
    message='full original calibration debit must remain reserved'
    if (result.get('operation')!='prepare' or result.get('mechanical_status')!='INVALID'
            or result.get('scientific_status')!='INVALID' or result.get('reason')!='CONTRACT_MISMATCH'
            or result.get('missing_evidence')!=[] or error.get('code')!='PRODUCER_FAILURE'
            or error.get('producer')!='host' or error.get('type')!='ValueError' or error.get('message')!=message):
        raise Blocked('only the original calibration reservation mismatch can be recovered')
    trace_ref=error['local_traceback']
    if trace_ref['path']!='private/host-traceback.txt' or trace_ref['bytes']>64<<10:
        raise Blocked('bounded original reservation traceback required')
    trace=checked_file(output,trace_ref).read_text()
    frames=re.findall(r'  File "([^"]+)", line \d+, in ([^\n]+)',trace)
    if (not frames or frames[-1]!=(str(root/'src/oeis_learn/pipeline_validation/progressive_calibration_reuse.py'),'assemble')
            or not trace.rstrip().endswith('ValueError: '+message)):
        raise Blocked('original reuse assembly failure location required')
    allowed={'owner.json','operation-binding.json','result.json','private/host-traceback.txt'}
    evidence=[]
    for name in ('image-0','image-1','device'):
        folder=output/'reuse-inspection'/name
        process=folder/'logs/process.json'
        cleanup=folder/('logs/cleanup.json' if name=='device' else 'cleanup.json')
        p=_json(process);c=_json(cleanup)
        if (p.get('returncode')!=0 or p.get('outcome')!='COMPLETE' or p.get('failure') or p.get('cleanup_failure')
                or c.get('acknowledged') is not True or c.get('within_bound') is not True or c.get('failure')):
            raise Blocked('terminated original reuse inspections required')
        evidence.extend(reference(root,x) for x in (process,cleanup))
        allowed.update(str((folder/'logs'/n).relative_to(output)) for n in ('process.json','command.json','stdout.log','stderr.log'))
        allowed.add(str(cleanup.relative_to(output)))
    allowed.add('reuse-inspection/device/inspection.json')
    # Every directory must lead to an allowlisted file; no calibration/training
    # output, published prepared state or recovery attempt may have started.
    for path in output.rglob('*'):
        confined(root,path)
        relative=str(path.relative_to(output))
        if path.is_file():
            if relative not in allowed:raise Blocked('unexpected reuse preparation artifact')
        elif not any(n.startswith(relative+'/') for n in allowed):
            raise Blocked('unexpected reuse preparation directory')
    reuse=_json(checked_file(root,named(prepared,'calibration_reuse')))
    request=_json(checked_file(root,reuse['request']))
    plan=named_plan(root,blueprint,'iteration_calibration_plan','progressive-calibration-plan/v1',
        {'seconds','caps','checkpoint_bytes','maximum_inodes'},optional={'checkpoint_retention'})
    if plan is None or plan['caps']==request['caps']:
        raise Blocked('original reservation mismatch must be independently reproducible')
    repaired=bind_original_reservation(root,blueprint,reuse)
    # Validate the original complete donor and exact current inputs/trajectory.
    # This is metadata-only historical replay, never a new prepared publication.
    _,resources=assemble(root,repaired,raw_prepared(root,repaired),output,historical=True)
    return dict(traceback=reference(root,output/trace_ref['path']),inspection_termination=evidence,
        original_reservation=named(blueprint,'iteration_calibration_plan'),
        retained_reservation=named(repaired,'iteration_calibration_plan'),replayed_resources=resources)
