"""Single-owner data reuse with evidenced pre-campaign diagnostic replacement."""
import fcntl
from pathlib import Path

from .identity import confined, file_hash, seal
from .preparation_recovery import retain
from .progressive_research_inputs import _json, checked_file
from .progressive_workstation_state import Blocked, read, write
from oeis_learn.generalization.progressive.contracts import reference

VERSION = 'progressive-preparation-owner/v2'
EARLY = {'execution-preflight', 'registered-data', 'rehearsal-reference', 'census-plan', 'environment',
         'image-qualification', 'image-gpu', 'native-census', 'materials',
         'role-roster', 'sensitivity', 'research-prerequisites', 'qualification-profiles'}


def failure_proof(root, receipt):
    """Caller holds the previous session lock. No scientific allowance transfers."""
    folder = confined(root, receipt['recipient'])
    session = read(folder/'session.json')
    journal = read(folder/'journal.json')
    if session['binding']['source'] != receipt['current_source']:
        raise Blocked('previous preparation source differs')
    from . import progressive_round019_successor as round019
    if receipt['recipient'] == round019.SESSION:
        return round019.proof(root, receipt, session, journal)
    from . import progressive_round020_successor as round020
    if receipt['recipient'] == round020.SESSION:
        return round020.proof(root, receipt, session, journal)
    from . import progressive_round022_successor as round022
    if receipt['recipient'] == round022.SESSION:
        return round022.proof(root, receipt, session, journal)
    from . import progressive_round023_successor as round023
    if receipt['recipient'] == round023.SESSION:
        return round023.proof(root, receipt, session, journal)
    from . import progressive_round024_successor as round024
    if receipt['recipient'] == round024.SESSION:
        return round024.proof(root, receipt, session, journal)
    from . import progressive_round025_successor as round025
    if receipt['recipient'] == round025.SESSION:
        return round025.proof(root, receipt, session, journal)
    if receipt['recipient'] == round025.SUCCESSOR:
        raise Blocked('round026 learner replacement has no further successor')
    steps = journal['steps']
    if steps.get('prepare-research/freeze', {}).get('status') == 'FAILED' and steps.get('prepare-research/prepare', {}).get('status') == 'COMPLETE':
        from .progressive_freeze_recovery import proof
        return proof(root, receipt, session, journal)
    if (steps.get('prepare-research/prepare', {}).get('status') in {'BLOCKED','FAILED'}
            and (folder/'prepare-research/calibration-reuse/blueprint.json').exists()):
        from .progressive_reused_preparation_recovery import proof
        return proof(root,receipt,session,journal)
    if steps.get('prepare-research/prepare', {}).get('status') == 'BLOCKED' and (folder/'prepare-research/prepared/calibration/producer.json').exists():
        from .progressive_completed_calibration_recovery import proof
        return proof(root,receipt,session,journal)
    if steps.get('prepare-research/prepare', {}).get('status') == 'FAILED':
        from .progressive_calibration_recovery import proof
        return proof(root,receipt,session,journal)
    if steps.get('prepare-research/construction', {}).get('status') == 'FAILED':
        from .progressive_construction_recovery import proof
        return proof(root,receipt,session,journal)
    if steps.get('prepare-research/base-qualification', {}).get('status') == 'COMPLETE':
        from .progressive_preparation_feasibility_recovery import proof
        return proof(root,receipt,session,journal)
    if steps.get('prepare-research/base-qualification', {}).get('status') != 'FAILED':
        raise Blocked('previous preparation must have failed base qualification')
    for key, row in steps.items():
        if not key.startswith('prepare-research/'):
            raise Blocked('previous preparation has work outside pre-research qualification')
        name = key.split('/', 1)[1]
        for attempt in [row, *row.get('history', [])]:
            state = attempt['status']
            if state in {'RUNNING', 'INTERRUPTED'} or (
                name not in EARLY | {'base-qualification'} and state != 'BLOCKED'
            ):
                raise Blocked('previous preparation has started downstream or unresolved work')
    base = folder/'prepare-research'
    if any((base/name).exists() for name in
           ('matrix/matrix.json', 'qualification', 'evaluation-timing', 'construction', 'prepared', 'frozen', 'acceptance')):
        raise Blocked('previous preparation has downstream execution artifacts')
    pipeline = base/'base-qualification/pipeline'
    summary = _json(confined(root, pipeline/'summary.json'), maximum=32 << 20)
    states = {'cpu.semantic': 'FAIL', 'hip.preflight': 'BLOCKED',
              'image.gpu': 'PASS', 'image.qualification': 'PASS'}
    if {k: v['state'] for k, v in summary['stages'].items()} != states:
        raise Blocked('only a CPU base-qualification failure before HIP can be replaced')
    evidence = []
    for name, expected in states.items():
        stage = pipeline/'stages'/name
        original = read(confined(root, stage/'receipt.json'))
        if (original != summary['stages'][name] or
                original['binding']['contract'] != receipt['current_source']):
            raise Blocked('original base-qualification receipt differs')
        processes = cleanups = 0
        for path, identity in original['files'].items():
            if not path.endswith(('process.json', 'cleanup.json')):
                continue
            selected = confined(stage, path)
            if file_hash(selected) != identity:
                raise Blocked('original qualification cleanup evidence changed')
            value = _json(selected, maximum=1 << 20)
            if path.endswith('process.json'):
                processes += 1
                if type(value.get('returncode')) is not int or value.get('cleanup_failure'):
                    raise Blocked('previous qualification process termination is unproven')
            else:
                cleanups += 1
                if (value.get('acknowledged') is not True or value.get('within_bound') is not True
                        or value.get('failure')):
                    raise Blocked('previous qualification cleanup is unproven')
            evidence.append(dict(original=reference(root, selected), value=value))
        if expected != 'BLOCKED' and (not processes or processes != cleanups):
            raise Blocked('complete qualification process and cleanup evidence required')
    return seal(dict(version='progressive-preparation-owner-proof/v1', predecessor=receipt,
        session=session, journal=journal, qualification=summary['stages'], processes=evidence,
        reason='CPU base qualification failed before HIP or research execution',
        original_costs_retained=True, scientific_allowance_transferred=False))


def claim_owner(root, claim, receipt):
    """Caller holds the donor lock; atomically fence old clients on replacement."""
    root, claim = Path(root), Path(claim)
    lock_path = confined(root, claim.with_suffix('.lock'), exists=False)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _claim_owner(root, claim, receipt)


def _claim_owner(root, claim, receipt):
    if not claim.exists():
        retain(claim, receipt)
    current = read(claim)
    node = current
    from . import progressive_round022_successor as round022
    # Allow exactly one additional reviewed ownership link for this successor.
    from . import progressive_round023_successor as round023
    from . import progressive_round024_successor as round024
    from . import progressive_round025_successor as round025
    history_limit = {round025.SUCCESSOR:20, round022.SUCCESSOR:17, round023.SUCCESSOR:18, round024.SUCCESSOR:19}.get(receipt['recipient'],16)
    ancestors = []
    for _ in range(history_limit):
        previous = node['receipt'] if node.get('version') == VERSION else node
        if previous.get('version') != 'progressive-preparation-data-reuse/v1' or seal(previous) != previous:
            raise Blocked('original preparation ownership receipt required')
        ancestors.append(previous)
        if node.get('version') != VERSION:
            break
        checked_file(root, node['proof'])
        node = read(checked_file(root, node['previous']))
    else:
        raise Blocked('bounded preparation ownership history exhausted')
    active = ancestors[0]
    if receipt != active:
        if len(ancestors) >= history_limit:
            raise Blocked('bounded preparation ownership history exhausted')
        if receipt['recipient'] in {r['recipient'] for r in ancestors}:
            raise Blocked('preparation recipient is superseded or its source changed; use a fresh session')
        stable = lambda r: {k: v for k, v in r.items() if k not in {'identity', 'recipient', 'current_source'}}
        if stable(receipt) != stable(active):
            raise Blocked('original preparation data ownership differs')
        previous_folder = confined(root, active['recipient'])
        if not previous_folder.is_relative_to(root/'run-state'):
            raise Blocked('original preparation owner must be a run-state session')
        if not all((previous_folder/name).is_file() for name in ('session.json','session.lock','journal.json')):
            raise Blocked('original failed preparation session evidence required')
        with confined(root, previous_folder/'session.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            proof = failure_proof(root, active)
            if proof['version'] == round022.VERSION and receipt['recipient'] != round022.SUCCESSOR:
                raise Blocked('round022 audit replacement belongs to round023')
            if proof['version'] == round023.VERSION and receipt['recipient'] != round023.SUCCESSOR:
                raise Blocked('round023 generator replacement belongs to round024')
            if proof['version'] == round024.VERSION and receipt['recipient'] != round024.SUCCESSOR:
                raise Blocked('round024 random-control replacement belongs to round025')
            if proof['version'] == round025.VERSION and receipt['recipient'] != round025.SUCCESSOR:
                raise Blocked('round025 capability import belongs to round026')
            history = claim.parent/(claim.stem+'-history')
            prior_path = history/(current['identity'][7:]+'.json')
            proof_path = history/(proof['identity'][7:]+'.json')
            retain(prior_path, current)
            retain(proof_path, proof)
            current = seal(dict(version=VERSION, receipt=receipt,
                previous=reference(root, prior_path), proof=reference(root, proof_path)))
            # Old versions use immutable retain() here, so cannot restart the
            # superseded recipient after this atomic pointer publication.
            write(claim, current)
    destination = confined(root, receipt['recipient'], exists=False)
    retain(destination/'preparation-owner.json', current)
    if current.get('version') == VERSION:
        retain(destination/'preparation-owner-proof.json', read(checked_file(root, current['proof'])))
