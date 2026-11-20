"""Transfer round020's unscored inputs after its reviewed HIP acceptance failure."""
import math

from oeis_learn.generalization.progressive.contracts import reference
from .identity import confined, digest, seal
from .progressive_research_inputs import checked_file
from .progressive_workstation_state import Blocked, read

SESSION = 'run-state/discovery/013-research-round-020'
ROUND021 = 'run-state/discovery/013-research-round-021'
VERSION = 'progressive-preparation-owner-proof/v10'
FILES = (
    'session.json', 'journal.json', 'preparation-owner.json', 'preparation-owner-proof.json',
    'panel-data-reuse.json', 'prepare-research/preparation-data-reuse.json',
    'prepare-research/calibration-reuse/calibration-reuse.json',
    'prepare-research/prepared/result.json', 'prepare-research/prepare-allowance.json',
    'prepare-research/frozen/freeze-publication.json', 'prepare-research/frozen/result.json',
    'prepare-research/frozen.freeze-progress/freeze-progress.json',
    'prepare-research/acceptance/software/suite/acceptance.json',
    'prepare-research/acceptance/device/integrated/request.json',
    'prepare-research/acceptance/device/integrated/case-outcomes.json',
    'prepare-research/acceptance/device/process.json', 'prepare-research/acceptance/device/cleanup.json',
    'prepare-research/acceptance/result.json', 'prepare-research/logs/acceptance/process.json',
)
REVIEWED = 'sha256:501c2a5a600085c96a4371d37d1336a8e99661fae6b97ad5c0c2a3115c449f1c'


def proof(root, receipt, session, journal):
    folder = confined(root, SESSION)
    base = folder / 'prepare-research'
    if digest([reference(folder, name) for name in FILES]) != REVIEWED:
        raise Blocked('round020 freeze, acceptance or accounting differs from the reviewed originals')
    if (receipt != read(base / 'preparation-data-reuse.json') or receipt['recipient'] != SESSION
            or read(folder / 'preparation-owner.json')['receipt'] != receipt
            or session != read(folder / 'session.json') or journal != read(folder / 'journal.json')):
        raise Blocked('round020 owner or session differs')
    campaign = read(base / 'frozen/result.json')['campaign']
    scientific_paths = [folder / 'run-research', folder / 'assess', base / 'construction/execution',
                        base / 'acceptance/acceptance.json', base / 'acceptance/device/integrated/producer.json',
                        root / 'run-state/discovery-control' / campaign[7:],
                        root / 'run-state/discovery-accounting' / campaign[7:],
                        root / 'run-state/discovery-learning' / campaign[7:],
                        root / 'run-state/discovery-components' / campaign[7:]]
    if any(confined(root, path, exists=False).exists() for path in scientific_paths):
        raise Blocked('round020 has downstream execution or acceptance authority')
    steps = journal['steps']
    for ref in steps['prepare-research/freeze']['outputs']:
        checked_file(root, ref)
    checked_file(base / 'acceptance', read(base / 'acceptance/result.json')['producer_error']['local_traceback'])
    reused = read(base / 'calibration-reuse/calibration-reuse.json')
    for key in ('original_qualification', 'qualification', 'blueprint', 'prepared', 'request', 'batch'):
        checked_file(root, reused[key])
    prepared = read(base / 'prepared/result.json')['metrics']['prepared']
    checked_file(base / 'prepared', prepared)
    panels = read(folder / 'panel-data-reuse.json')
    for key in ('original_proof', 'materials', 'original_materials'):
        checked_file(root, panels[key])
    prior = read(folder / 'preparation-owner-proof.json')
    # Account for the entire measured round, including blocked/failed attempts.
    elapsed = sum(attempt.get('elapsed_seconds', 0) for row in steps.values()
                  for attempt in [*row.get('history', []), row])
    return seal(dict(version=VERSION, predecessor=receipt, reviewed_evidence=REVIEWED,
                     session=session['identity'], journal=journal['identity'], frozen_campaign=campaign,
                     prepared=prepared, calibration_batch=reused['batch'], corrected_materials=panels['materials'],
                     charged_seconds=prior['charged_seconds'] + elapsed, original_costs_retained=True,
                     scientific_allowance_transferred=False, current_source_requalification_required=True,
                     historical_qualification_only=True))


def setup_limits(setup, proof):
    if proof['version'] != VERSION or proof['reviewed_evidence'] != REVIEWED or seal(proof) != proof:
        raise Blocked('reviewed round020 owner proof required')
    # Include a full diagnostic rehearsal ceiling even if the operator used less.
    from .research_limits import DEVICE_ACCEPTANCE_SECONDS, CLEANUP_SECONDS
    freeze = 86400
    prepare = (setup['prepare'] + setup['freeze'] - freeze - math.ceil(proof['charged_seconds'])
               - DEVICE_ACCEPTANCE_SECONDS - 2 * CLEANUP_SECONDS)
    if prepare <= 120:
        raise Blocked('insufficient original preparation allowance')
    return dict(setup, prepare=prepare, freeze=freeze)


def replacement_limits(root, setup, failure):
    """Carry the round020 reservation through round021's pre-HIP qualification failure."""
    receipt = failure['predecessor']
    if (failure['version'] != 'progressive-preparation-owner-proof/v1' or seal(failure) != failure
            or receipt['recipient'] != ROUND021):
        raise Blocked('original round021 CPU base-qualification failure proof required')
    folder = confined(root, ROUND021)
    owner = read(folder / 'preparation-owner.json')
    if owner['receipt'] != receipt:
        raise Blocked('round021 preparation owner differs')
    previous = read(checked_file(root, owner['proof']))
    limits = setup_limits(setup, previous)
    attempts = (attempt for row in failure['journal']['steps'].values()
                for attempt in [*row.get('history', []), row])
    charged = sum(a.get('elapsed_seconds', 0) if a['status'] == 'BLOCKED'
                  else a['elapsed_seconds'] for a in attempts)
    limits['prepare'] -= math.ceil(charged)
    if limits['prepare'] <= 120:
        raise Blocked('round021 spent the remaining preparation allowance')
    return limits
