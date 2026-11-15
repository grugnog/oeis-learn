"""Transfer data ownership only after complete calibration and a time-only stop."""
from pathlib import Path

from .progressive_process_result import stdout_result

from .identity import confined, digest, seal
from .progressive_workstation_state import Blocked, read
from .progressive_research_inputs import checked_file, _json
from oeis_learn.generalization.progressive.contracts import reference, CAMPAIGN_DOCUMENT_BYTES, validate


def originals(root, base):
    """Replay the old publication in its own original authority, without edits."""
    from .progressive_calibration import validate_prepared, verify_result
    from .progressive_handoff import original_authority, validate_device_proofs, named_plan
    from .progressive_trajectory import clean_stage
    base = Path(base)
    blueprint = validate('blueprint', _json(base/'construction/blueprint.json', maximum=CAMPAIGN_DOCUMENT_BYTES))
    prepared = validate('prepared', _json(base/'prepared/prepared.json', maximum=CAMPAIGN_DOCUMENT_BYTES))
    if any(r['name'] == 'calibration_reuse' for r in prepared['payload']['references']):
        raise ValueError('calibration reuse chains are not authorized')
    deficits = prepared['payload']['deficits']
    if not deficits or any(not d.startswith('training_time:') for d in deficits):
        raise ValueError('only a whole-training-time feasibility deficit is eligible')
    validate_prepared(root, blueprint, prepared)
    device = base/'prepared/calibration'
    req = read(device/'request.json')
    result = read(device/'producer.json')
    # validate_prepared already replayed the complete inputs, producers and
    # retirement inventories. Read its authenticated small producer records.
    producers = [_json(checked_file(device,result['producers'][c['compatibility']])) for c in req['cases']]
    clean_stage(device)
    qref = reference(root, base/'qualification/qualification.json')
    q, qroot = original_authority(root, qref)
    matrix, _ = validate_device_proofs(qroot, q, source=req['source'], image=req['image'], engine=req['engine'])
    if (req['hardware'] != matrix['hardware'] or req['execution_policy'] != q['runtime']['execution_policy']
            or req['source'] != digest(_json(checked_file(qroot, q['source_closure'])))):
        raise ValueError('original calibration hardware/source authority differs')
    plan = named_plan(root, blueprint, 'iteration_calibration_plan', 'progressive-calibration-plan/v1',
        {'seconds', 'caps', 'checkpoint_bytes', 'maximum_inodes'}, optional={'checkpoint_retention'})
    if not plan or any(req[k] != plan[k] for k in ('caps', 'checkpoint_bytes', 'maximum_inodes', 'checkpoint_retention')):
        raise ValueError('original complete calibration reservation required')
    state = Path(root)/'run-state/discovery-calibration'/digest(['iteration-calibration',blueprint['identity']])[7:]
    allowance = read(state/'preparation-allowance.json')
    attempts = allowance.get('attempts', [])
    if (allowance.get('active') is not None or allowance.get('seconds') != plan['seconds']
            or allowance.get('binding') != dict(blueprint=blueprint['identity'], prepared=req['prepared'], output=str((base/'prepared').relative_to(root)))
            or len(attempts) != 1 or attempts[0].get('outcome') != 'COMPLETE'
            or allowance.get('spent_seconds') != attempts[0].get('charged_seconds')
            or not result['actual_elapsed_us']/1000000 <= allowance['spent_seconds'] <= plan['seconds']):
        raise ValueError('closed original calibration allowance and consumed time required')
    return blueprint, prepared, req, producers


def proof(root, receipt, session, journal):
    from .progressive_preparation_owner import EARLY
    from .progressive_construction_recovery import prerequisites
    from .progressive_calibration_equivalence import validate as equivalent
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    folder = confined(root, receipt['recipient']); base = folder/'prepare-research'
    allowed = EARLY | {'base-qualification', 'matrix', 'qualification-plan', 'qualify',
                      'evaluation-plan', 'evaluation-timing', 'evaluation-feasibility', 'construction', 'prepare'}
    for key, row in journal['steps'].items():
        if not key.startswith('prepare-research/'):
            raise Blocked('calibration reuse cannot transfer scientific execution')
        name = key.split('/', 1)[1]
        for attempt in [row, *row.get('history', [])]:
            if attempt['status'] in {'RUNNING', 'INTERRUPTED'} or (name not in allowed and attempt['status'] != 'BLOCKED'):
                raise Blocked('started downstream or unresolved preparation work')
    step = journal['steps'].get('prepare-research/prepare', {})
    if step.get('status') != 'BLOCKED' or step.get('history'):
        raise Blocked('one original time-blocked preparation required')
    if any((base/p).exists() for p in ('frozen', 'acceptance', 'construction/execution')):
        raise Blocked('calibration reuse requires an unstarted scientific campaign')
    req, contract = prerequisites(root, receipt, journal)
    construction = journal['steps'].get('prepare-research/construction', {})
    if construction.get('status') != 'COMPLETE':
        raise Blocked('complete original construction required')
    owned = {r['path']: r for r in construction.get('outputs', [])}
    for r in owned.values(): checked_file(root, r)
    if owned.get(str((base/'construction/blueprint.json').relative_to(root))) != reference(root, base/'construction/blueprint.json'):
        raise Blocked('original construction blueprint must be journal-bound')
    blueprint, prepared, calibration, producers = originals(root, base)
    summary_path=base/'construction/construction-summary.json'
    if owned.get(str(summary_path.relative_to(root)))!=reference(root,summary_path):
        raise Blocked('original construction summary must be journal-bound')
    summary=read(summary_path)
    if (summary.get('status')!='COMPLETE' or summary.get('assessment_exposed') is not False
            or summary.get('executable') is not False or summary.get('exact_calibration_cases')!=len(producers)
            or summary.get('blueprint')!=reference(root,base/'construction/blueprint.json')):
        raise Blocked('complete unexposed original construction inventory required')
    expected_binding=seal(dict(operation='prepare',blueprint=blueprint['identity'],
        qualification=reference(root,base/'qualification/qualification.json')))
    owner=read(base/'prepared/owner.json')
    if (read(base/'prepared/operation-binding.json')!=expected_binding
            or owner.get('version')!='discovery-owner/v1' or owner.get('output')!=str((base/'prepared').relative_to(root))):
        raise Blocked('original preparation owner/binding differs')
    q = _json(checked_file(root, req['qualification']))
    qroot = checked_file(root, req['qualification']).parent
    changed = equivalent(_json(checked_file(qroot, q['source_closure'])), collect_snapshot(root))
    result = read(base/'prepared/result.json')
    if (result.get('mechanical_status') != 'PARTIAL' or result.get('reason') != 'BOUND_OUTCOME_BLIND_PREPARATION'
            or result.get('operation') != 'prepare' or result.get('producer_error') is not None
            or result.get('scientific_status') != 'UNMEASURED'
            or result.get('missing_evidence') != prepared['payload']['deficits']
            or result.get('metrics', {}).get('prepared') != reference(base/'prepared', base/'prepared/prepared.json')):
        raise Blocked('original calibration-complete time-only preparation result required')
    logs = base/'logs/prepare'
    if sorted(p.name for p in logs.iterdir()) != ['0000']:
        raise Blocked('one original canonical prepare invocation required')
    logs = logs/'0000'; process = _json(logs/'process.json'); command = _json(logs/'command.json')
    b = session['binding']
    expected = ['-S', str(root/'scripts/foundation_run.py'), '--engine', b['engine'],
        '--memory-gib', str(b['memory_gib']), '--cpus', str(b['cpus']), '--host-free-memory-gib', '24',
        'discovery', 'prepare', '--blueprint', str((base/'construction/blueprint.json').relative_to(root)),
        '--qualification', str((base/'qualification').relative_to(root)), '--output', str((base/'prepared').relative_to(root))]
    if (command['argv'][1:] != expected or process.get('returncode') != 2 or process.get('cleanup_failure')
            or process.get('failure') or process.get('outcome') != 'COMPLETE' or process.get('omitted_bytes', {}).get('stdout', 0) != 0
            or stdout_result(logs/'stdout.log') != result):
        raise Blocked('cleanly terminated original time-blocked canonical prepare required')
    return seal(dict(version='progressive-preparation-owner-proof/v5', predecessor=receipt,
        session=session, journal=journal, qualification=req['qualification'], execution_contract=contract,
        blueprint=reference(root, base/'construction/blueprint.json'), prepared=reference(root, base/'prepared/prepared.json'),
        calibration=reference(root, base/'prepared/calibration/producer.json'),
        request=reference(root, base/'prepared/calibration/request.json'),
        result=reference(root, base/'prepared/result.json'), process=reference(root, logs/'process.json'),
        reason='complete exact-input calibration; whole-trajectory time feasibility only',
        reviewed_host_changes=changed, original_costs_retained=True, scientific_allowance_transferred=False,
        historical_qualification_only=True, current_source_requalification_required=True))
