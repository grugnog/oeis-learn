"""Replace only an evidenced metadata preflight failure before calibration work."""
import re

from .identity import confined, digest, seal
from .progressive_workstation_state import Blocked, read
from .progressive_research_inputs import checked_file, _json
from oeis_learn.generalization.progressive.contracts import reference, record, CAMPAIGN_DOCUMENT_BYTES

ERROR = 'owned storage byte/inode reservation exhausted'


def proof(root, receipt, session, journal):
    from .progressive_preparation_owner import EARLY
    from .progressive_construction_recovery import prerequisites
    from .progressive_handoff import named_plan

    folder = confined(root, receipt['recipient'])
    base = folder/'prepare-research'
    output = base/'prepared'
    allowed = EARLY | {'base-qualification', 'matrix', 'qualification-plan', 'qualify',
                      'evaluation-plan', 'evaluation-timing', 'evaluation-feasibility', 'construction', 'prepare'}
    steps = journal['steps']
    for key, row in steps.items():
        if not key.startswith('prepare-research/'):
            raise Blocked('scientific execution cannot receive calibration preflight replacement')
        name = key.split('/', 1)[1]
        for attempt in [row, *row.get('history', [])]:
            status = attempt['status']
            if status in {'RUNNING', 'INTERRUPTED'} or (name not in allowed and status != 'BLOCKED'):
                raise Blocked('started downstream or unresolved preparation work')
        if name == 'prepare' and (row['status'] != 'FAILED' or row.get('history')):
            raise Blocked('one original failed preparation attempt required')
    if any((base/name).exists() for name in ('frozen', 'acceptance', 'construction/execution')):
        raise Blocked('campaign execution already exists')
    # An inspection directory, request, recovery attempt or even an empty case
    # directory means this narrowly pre-dispatch exception does not apply.
    allowed_files = {'owner.json', 'operation-binding.json', 'result.json', 'private', 'private/host-traceback.txt'}
    for path in output.rglob('*'):
        confined(root, path)
        if str(path.relative_to(output)) not in allowed_files:
            raise Blocked('calibration preflight has downstream or recovery artifacts')
    req, contract = prerequisites(root, receipt, journal)
    construction = steps.get('prepare-research/construction', {})
    if construction.get('status') != 'COMPLETE':
        raise Blocked('original completed construction required')
    originals = {ref['path']: ref for ref in construction.get('outputs', [])}

    def owned(path):
        ref = reference(root, path)
        if originals.get(ref['path']) != ref:
            raise Blocked('original construction publication is not journal-bound')
        return _json(checked_file(root, ref), maximum=CAMPAIGN_DOCUMENT_BYTES)

    summary = owned(base/'construction/construction-summary.json')
    blueprint = owned(base/'construction/blueprint.json')
    if (seal(summary) != summary or summary.get('status') != 'COMPLETE'
            or summary.get('assessment_exposed') is not False or summary.get('executable') is not False
            or summary['blueprint'] != reference(root, base/'construction/blueprint.json')
            or seal(blueprint) != blueprint or blueprint.get('purpose') != 'research'):
        raise Blocked('original unexposed metadata construction required')
    plan = named_plan(root, blueprint, 'iteration_calibration_plan', 'progressive-calibration-plan/v1',
                      {'seconds', 'caps', 'checkpoint_bytes', 'maximum_inodes'}, optional={'checkpoint_retention'})
    if plan is None or plan['maximum_inodes'] != 64*summary['exact_calibration_cases']+256:
        raise Blocked('original exact calibration inode reservation required')
    qualification = reference(root, base/'qualification/qualification.json')
    if qualification != req['qualification'] or read(output/'operation-binding.json') != seal(dict(
            operation='prepare', blueprint=blueprint['identity'], qualification=qualification)):
        raise Blocked('original preparation output binding differs')
    owner = read(output/'owner.json')
    if owner.get('version') != 'discovery-owner/v1' or owner.get('output') != str(output.relative_to(root)):
        raise Blocked('original preparation owner differs')
    result = read(output/'result.json')
    error = result.get('producer_error', {})
    if (result.get('operation') != 'prepare' or result.get('mechanical_status') != 'FAILED'
            or result.get('scientific_status') != 'UNMEASURED' or result.get('reason') != 'HOST_FAILURE'
            or error.get('type') != 'TimeoutError' or error.get('message') != ERROR
            or error.get('producer') != 'host' or error.get('code') != 'PRODUCER_FAILURE'):
        raise Blocked('original host calibration storage preflight failure required')
    trace_ref = error['local_traceback']
    if trace_ref['path'] != 'private/host-traceback.txt' or trace_ref['bytes'] > 64 << 10:
        raise Blocked('bounded original local preflight traceback required')
    trace = checked_file(output, trace_ref).read_text()
    frames = re.findall(r'  File "([^"]+)", line \d+, in ([^\n]+)\n([^\n]*)', trace)
    expected = [(str(root/'src/oeis_learn/pipeline_validation/progressive_calibration.py'), '_prepare'),
                (str(root/'src/oeis_learn/generalization/progressive/storage.py'), 'check')]
    if ([(p, f) for p, f, _ in frames[-2:]] != expected
            or ''.join(frames[-2][2].split()) != 'guard.check(required_bytes,64*len(cases)+256)'
            or not trace.rstrip().endswith('TimeoutError: '+ERROR)):
        raise Blocked('failure must be the original total-as-additional preflight before dispatch')
    state = confined(root, 'run-state/discovery-calibration/'+digest(['iteration-calibration', blueprint['identity']])[7:])
    if {p.name for p in state.iterdir()} != {'preparation-allowance.json', 'preparation-allowance.lock'}:
        raise Blocked('calibration budget reservation or unknown state already exists')
    allowance = read(state/'preparation-allowance.json')
    for path in state.iterdir():
        confined(root, path)
    preparation = owned(base/'construction/preparation-plan.json')
    prepared = record('prepared', dict(preparation['payload'], blueprint=blueprint['identity']),
                      created_at=blueprint['created_at'], purpose=blueprint['purpose'], track=blueprint['track'])
    attempts = allowance.get('attempts', [])
    if (allowance.get('active') is not None or allowance.get('seconds') != plan['seconds'] or len(attempts) != 1
            or allowance.get('binding') != dict(blueprint=blueprint['identity'], prepared=prepared['identity'], output=str(output.relative_to(root)))
            or attempts[0].get('outcome') != 'TimeoutError' or attempts[0].get('error') != ERROR
            or allowance.get('spent_seconds') != attempts[0].get('charged_seconds')
            or not 0 < allowance['spent_seconds'] <= plan['seconds']):
        raise Blocked('closed original preflight allowance and consumed time required')
    logs = base/'logs/prepare'
    if sorted(p.name for p in logs.iterdir()) != ['0000']:
        raise Blocked('one original prepare process required')
    logs = logs/'0000'
    command = _json(logs/'command.json')
    process = _json(logs/'process.json')
    binding = session['binding']
    expected = ['-S', str(root/'scripts/foundation_run.py'), '--engine', binding['engine'],
                '--memory-gib', str(binding['memory_gib']), '--cpus', str(binding['cpus']),
                '--host-free-memory-gib', '24', 'discovery', 'prepare',
                '--blueprint', str((base/'construction/blueprint.json').relative_to(root)),
                '--qualification', str((base/'qualification').relative_to(root)), '--output', str(output.relative_to(root))]
    if (command['argv'][1:] != expected or process.get('returncode') != 1
            or process.get('cleanup_failure') is not None or process.get('failure') is not None
            or process.get('outcome') != 'COMPLETE' or process.get('elapsed_seconds', 0) <= 0
            or _json(logs/'stdout.log') != result):
        raise Blocked('original cleanly terminated canonical prepare process required')
    return seal(dict(version='progressive-preparation-owner-proof/v4', predecessor=receipt, session=session, journal=journal,
        qualification=req['qualification'], execution_contract=contract, construction=summary,
        result=dict(original=reference(root, output/'result.json'), value=result),
        # Keep private traceback bytes local; only their original identity travels.
        traceback=reference(root, output/trace_ref['path']),
        allowance=dict(original=reference(root, state/'preparation-allowance.json'), value=allowance),
        command=dict(original=reference(root, logs/'command.json'), value=command),
        process=dict(original=reference(root, logs/'process.json'), value=process),
        reason='exact-total storage preflight failed before calibration reservation or device dispatch',
        original_costs_retained=True, scientific_allowance_transferred=False,
        historical_qualification_only=True, current_source_requalification_required=True))
