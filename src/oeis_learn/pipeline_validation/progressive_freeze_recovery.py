"""Authenticate a terminated host freeze defect before fresh qualification.

This transfers unscored inputs and historical calibration provenance, never
execution authority. Canonical preparation still verifies every measurement.
"""
import re
from types import SimpleNamespace

from .identity import confined, seal
from .progressive_workstation_state import Blocked, read
from .progressive_research_inputs import checked_file, _json
from .progressive_process_result import stdout_result
from oeis_learn.generalization.progressive.contracts import reference, CAMPAIGN_DOCUMENT_BYTES

VERSION = 'progressive-preparation-owner-proof/v8'
ERROR = 'artifact inventory cap'


def revised_materials(root, journal, source):
    row = journal['steps'].get('panel-revision/regroup', {})
    report = row.get('value', {})
    if (row.get('status') != 'COMPLETE' or seal(report) != report
            or report.get('status') != 'PANELS_READY' or report.get('source') != source
            or report.get('deficits') or report.get('training_support_unchanged') is not True
            or report.get('assessment_scored') is not False
            or report.get('model_calls') != 0 or report.get('new_draws') != 0):
        raise Blocked('original unscored panel revision required')
    outputs = row.get('outputs', [])
    for ref in outputs:
        checked_file(root, ref)
    if report['materials'] not in outputs:
        raise Blocked('revised materials must be journal-bound')
    original = _json(checked_file(root, report['original_materials']))
    materials = _json(checked_file(root, report['materials']))
    if (materials.get('status') != 'COMPLETE' or materials.get('diagnostic') is not False
            or materials.get('assessment_exposed') is not False or materials.get('deficits')):
        raise Blocked('complete unexposed research materials required')
    for key in ('support', 'support_summary'):
        for value in (original, materials):
            checked_file(root, value[key])
        if any(original[key][field] != materials[key][field] for field in ('bytes', 'sha256')):
            raise Blocked('original training support changed')
    return report, original, materials


def proof(root, receipt, session, journal):
    from .progressive_preparation_owner import EARLY
    from .progressive_construction_recovery import prerequisites
    from .progressive_calibration_reuse import named, source_authority
    from .progressive_workstation import command
    folder = confined(root, receipt['recipient']); base = folder/'prepare-research'
    steps = journal['steps']
    allowed = EARLY | {'base-qualification', 'matrix', 'qualification-plan', 'qualify',
        'evaluation-plan', 'evaluation-timing', 'evaluation-feasibility', 'construction',
        'trajectory-plan', 'training-trajectory', 'calibration-reuse-plan', 'prepare',
        'reservation-repair', 'panel-campaign-enroll', 'panel-campaign-helper',
        'panel-campaign-roster', 'panel-campaign-estimates', 'panel-campaign-construction',
        'panel-campaign-calibration', 'panel-campaign-archive', 'freeze', 'acceptance', 'readiness'}
    for key, row in steps.items():
        if key not in {'panel-audit/inspect', 'panel-audit/inspect-v2', 'panel-revision/regroup'} and (
                not key.startswith('prepare-research/') or key.split('/', 1)[1] not in allowed):
            raise Blocked('scientific or unknown work cannot transfer freeze recovery')
        for attempt in [row, *row.get('history', [])]:
            if attempt['status'] not in {'COMPLETE', 'FAILED', 'BLOCKED'}:
                raise Blocked('unresolved work cannot transfer freeze recovery')
            if key in {'prepare-research/acceptance', 'prepare-research/readiness'} and attempt['status'] != 'BLOCKED':
                raise Blocked('started downstream work cannot transfer freeze recovery')
        if row['status'] == 'COMPLETE':
            for ref in row.get('outputs', []):
                checked_file(root, ref)
    if (steps.get('prepare-research/prepare', {}).get('status') != 'COMPLETE'
            or steps.get('prepare-research/freeze', {}).get('status') != 'FAILED'):
        raise Blocked('complete prepare followed by failed freeze required')
    for path in (base/'frozen/campaign.json', base/'acceptance', folder/'run-research', folder/'assess',
                 base/'construction/execution', base/'panel-campaign/construction/execution'):
        if path.exists():
            raise Blocked('unstarted scientific campaign required')
    req, contract = prerequisites(root, receipt, journal)
    report, _, _ = revised_materials(root, journal, receipt['current_source'])
    binding = steps['prepare-research/freeze'].get('binding', {})
    arguments = binding.get('arguments', {})
    expected = dict(blueprint=str((base/'panel-campaign/calibration-reuse/blueprint.json').relative_to(root)),
        prepared=str((base/'panel-campaign/prepared').relative_to(root)),
        qualification=str((base/'qualification').relative_to(root)), output=str((base/'frozen').relative_to(root)))
    if binding != seal(dict(operation='freeze', arguments=expected)):
        raise Blocked('original panel-campaign freeze binding required')
    prepared_base = confined(root, arguments['prepared'])
    prepared_result = read(prepared_base/'result.json')
    if (reference(root, prepared_base/'result.json') not in steps['prepare-research/prepare']['outputs']
            or seal(prepared_result) != prepared_result or prepared_result.get('mechanical_status') != 'COMPLETE'):
        raise Blocked('original completed preparation result required')
    prepared_path = checked_file(prepared_base, prepared_result['metrics']['prepared'])
    prepared = _json(prepared_path, maximum=CAMPAIGN_DOCUMENT_BYTES)
    if seal(prepared) != prepared or prepared['payload']['deficits']:
        raise Blocked('original complete prepared authority required')
    reuse_ref = named(prepared, 'calibration_reuse')
    reuse = _json(checked_file(root, reuse_ref))
    source_authority(root, reuse, req['qualification'])
    donor = confined(root, reuse['blueprint']['path']).parent.parent.parent
    if str(donor.relative_to(root)) != session['binding'].get('reuse_calibration_session'):
        raise Blocked('original complete calibration donor differs')
    result = read(base/'frozen/result.json'); error = result.get('producer_error') or {}
    if (seal(result) != result or result.get('operation') != 'freeze'
            or result.get('mechanical_status') != 'INVALID' or result.get('reason') != 'CONTRACT_MISMATCH'
            or error.get('type') != 'ValueError' or error.get('message') != ERROR or error.get('producer') != 'host'):
        raise Blocked('only the original host policy byte-cap failure can recover')
    trace_ref = error['local_traceback']
    if trace_ref['path'] != 'private/host-traceback.txt' or trace_ref['bytes'] > 64 << 10:
        raise Blocked('bounded original freeze traceback required')
    trace_path = checked_file(base/'frozen', trace_ref); trace = trace_path.read_text()
    frames = re.findall(r'  File "([^"]+)", line \d+, in ([^\n]+)', trace)
    suffix = [(str(root/'src/oeis_learn'/name), function) for name, function in (
        ('pipeline_validation/progressive_study.py', 'validate_frozen'),
        ('pipeline_validation/progressive_study.py', 'policy'),
        ('generalization/progressive/contracts.py', 'verify_refs'))]
    if frames[-3:] != suffix or not trace.rstrip().endswith('ValueError: '+ERROR):
        raise Blocked('original study-policy failure location required')
    options = SimpleNamespace(**{k:session['binding'].get(k) for k in ('engine', 'memory_gib', 'cpus', 'render_node')})
    expected_command = command(root, options, 'freeze', **expected)[1:]
    candidates = []
    for path in (base/'panel-campaign/logs/freeze').glob('*/command.json'):
        if _json(path).get('argv', [])[1:] == expected_command:
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise Blocked('one original canonical freeze process required')
    logs = candidates[0]; process = _json(logs/'process.json')
    if (process.get('returncode') != 3 or process.get('outcome') != 'COMPLETE'
            or process.get('failure') or process.get('cleanup_failure')
            or process.get('omitted_bytes', {}).get('stdout', 0) or stdout_result(logs/'stdout.log') != result):
        raise Blocked('clean terminated original freeze required')
    allowance_path = base/'panel-campaign/continuation-allowance.json'
    allowance = read(allowance_path)
    enrolled = steps['prepare-research/panel-campaign-enroll']['value']
    if (seal(allowance) != allowance or allowance.get('active') is not None
            or allowance.get('binding') != enrolled['identity']
            or not 0 < allowance['spent_seconds'] <= allowance['seconds']):
        raise Blocked('closed original continuation debit required')
    return seal(dict(version=VERSION, predecessor=receipt, session=session, journal=journal,
        qualification=req['qualification'], execution_contract=contract,
        blueprint=reference(root, confined(root, arguments['blueprint'])), prepared=reference(root, prepared_path),
        calibration_reuse=reuse_ref, result=reference(root, base/'frozen/result.json'),
        process=reference(root, logs/'process.json'), traceback=reference(root, trace_path),
        failure_frames=[dict(file=path, function=function) for path, function in frames],
        continuation_allowance=reference(root, allowance_path), panel_revision=report,
        reason='terminated study-policy byte-cap failure before campaign publication or science',
        original_costs_retained=True, scientific_allowance_transferred=False,
        historical_qualification_only=True, current_source_requalification_required=True))


# Exact reviewed changes from the qualified producer to the panel-isolation
# repair. They alter grouping/early validation, not native generation; reuse
# therefore requires the already successful, unscored panel correction.
PANEL_PRODUCER_REVISIONS = {
    'src/oeis_learn/pipeline_validation/progressive_research_census.py': (
        'sha256:7be162c87d512b66dfe44fd01966bf2744fec51984332aed40ac234a02e26daa',
        'sha256:37b97c90067043e6e80b859e1ec365f714f4ab700ad54b5235d62af90ff0087f'),
    'src/oeis_learn/pipeline_validation/progressive_research_materials.py': (
        'sha256:7c0169e08f06422cf7b3cffe22eba8ff004209560f791184e7dbd631b1c9dc94',
        'sha256:ff4959beca474ad689abd350ef339f75e1a51d7f03725bfc6901f247569c3bf9'),
}


def panel_proof(root, owner):
    """Locate the immutable correction through bounded ownership ancestry."""
    from .progressive_preparation_owner import VERSION as OWNER_VERSION
    evidence = None; path = None
    for _ in range(16):
        if owner.get('version') != OWNER_VERSION:
            return None
        path = checked_file(root, owner['proof'])
        candidate = read(path)
        previous = read(checked_file(root, owner['previous']))
        predecessor = previous['receipt'] if previous.get('version') == OWNER_VERSION else previous
        if candidate.get('predecessor') != predecessor:
            raise Blocked('original owner-bound panel recovery proof required')
        if candidate.get('version') == VERSION:
            evidence = candidate
            break
        owner = previous
    if evidence is None:
        raise Blocked('bounded panel recovery ownership history exhausted')
    return evidence, path


def verify_panel_data_reuse(root, claim, original_materials):
    """Review before ownership changes; the transfer rechecks termination."""
    import fcntl
    from .progressive_preparation_owner import VERSION as OWNER_VERSION, failure_proof
    if not claim.is_file():
        raise Blocked('reviewed producer changes require original panel recovery')
    owner = read(claim)
    found = panel_proof(root, owner)
    if found is None:
        active = owner['receipt'] if owner.get('version') == OWNER_VERSION else owner
        folder = confined(root, active['recipient'])
        with confined(root, folder/'session.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            evidence = failure_proof(root, active)
        if evidence.get('version') != VERSION:
            raise Blocked('reviewed producer changes require original panel recovery')
    else:
        evidence, _ = found
    report, original, _ = revised_materials(root, evidence['journal'], evidence['predecessor']['current_source'])
    if original != original_materials or report != evidence['panel_revision']:
        raise Blocked('reviewed producer changes require the same corrected data donor')


def adopt_panels(root, recipient, values, outputs, *, required=False):
    """The new owner's immutable proof selects the already corrected panels."""
    from .preparation_recovery import retain
    found = panel_proof(root, read(recipient/'preparation-owner.json'))
    if found is None:
        if required:
            raise Blocked('reviewed producer changes require original panel recovery')
        return
    evidence, path = found
    report, original, materials = revised_materials(root, evidence['journal'], evidence['predecessor']['current_source'])
    if original != values['materials'] or report != evidence['panel_revision']:
        raise Blocked('panel correction must retain the original unscored data donor')
    receipt = seal(dict(version='progressive-panel-data-reuse/v1', original_proof=reference(root, path),
        original_materials=report['original_materials'], materials=report['materials'],
        revision=report, training_support_unchanged=True, current_source_requalification_required=True))
    retain(recipient/'panel-data-reuse.json', receipt)
    values['materials'] = materials
    outputs['materials'] = [checked_file(root, ref) for ref in evidence['journal']['steps']['panel-revision/regroup']['outputs']]
    outputs['materials'].extend([path, recipient/'panel-data-reuse.json'])
