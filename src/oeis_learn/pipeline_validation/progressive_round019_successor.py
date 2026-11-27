"""Transfer unscored data after the reviewed round019 template-budget failure."""
import math
import runpy

from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES, load, reference
from .identity import confined, digest, seal
from .progressive_research_inputs import checked_file
from .progressive_workstation_state import Blocked, read

SESSION = 'run-state/discovery/013-research-round-019'
VERSION = 'progressive-preparation-owner-proof/v9'
# Exact metadata from the reviewed workstation ZIP, including all prior failed
# attempts, the clean exit, stdout, the original debit chain and panel adoption.
FILES = (
    'session.json', 'journal.json', 'preparation-owner.json', 'panel-data-reuse.json',
    'prepare-research/preparation-data-reuse.json', 'prepare-research/prepared/result.json',
    'prepare-research/qualification/qualification.json',
    'prepare-research/calibration-reuse/calibration-reuse.json',
    'prepare-research/freeze-continuation/continuation.json',
    'prepare-research/freeze-continuation/continuation-allowance.json',
    'prepare-research/storage-continuation/continuation.json',
    'prepare-research/storage-continuation/continuation-allowance.json',
    'prepare-research/prepare-allowance.json', 'prepare-research/frozen/result.json',
    'prepare-research/segv-continuation/continuation.json',
    'prepare-research/segv-continuation/continuation-allowance.json',
    'prepare-research/segv-continuation/logs/0000/command.json',
    'prepare-research/segv-continuation/logs/0000/process.json',
    'prepare-research/segv-continuation/logs/0000/stdout.log',
    'prepare-research/segv-continuation/freeze-traceback.log',
)
REVIEWED = 'sha256:1a939317c7f8aac06bddbd86791c8306de1c947c8c607bdeaad0257a78b37757'


def proof(root, receipt, session, journal):
    folder = confined(root, SESSION)
    base = folder / 'prepare-research'
    refs = [reference(folder, name) for name in FILES]
    if digest(refs) != REVIEWED:
        raise Blocked('round019 failure, journal or accounting differs from the reviewed bundle')
    if (receipt != read(base / 'preparation-data-reuse.json') or receipt['recipient'] != SESSION
            or read(folder / 'preparation-owner.json')['receipt'] != receipt
            or session != read(folder / 'session.json') or journal != read(folder / 'journal.json')):
        raise Blocked('round019 owner or session differs')
    if any(confined(root, path, exists=False).exists() for path in (
            base / 'frozen/campaign.json', base / 'frozen/tasks', base / 'acceptance',
            base / 'construction/execution', folder / 'run-research', folder / 'assess')):
        raise Blocked('round019 has downstream or scientific work')
    failed = read(base / 'frozen/result.json')
    checked_file(base / 'frozen', failed['producer_error']['local_traceback'])
    prepared = read(base / 'prepared/result.json')['metrics']['prepared']
    checked_file(base / 'prepared', prepared)
    reused = read(base / 'calibration-reuse/calibration-reuse.json')
    for key in ('original_qualification', 'qualification', 'blueprint', 'prepared', 'request', 'batch'):
        checked_file(root, reused[key])
    panels = read(folder / 'panel-data-reuse.json')
    for key in ('original_proof', 'materials', 'original_materials'):
        checked_file(root, panels[key])
    prior = read(base / 'storage-continuation/continuation.json')
    latest = read(base / 'segv-continuation/continuation.json')
    for ref in (prior['original_script'], prior['script'], latest['script'], *latest['crash_logs']):
        checked_file(root, ref)
    debit = read(base / 'segv-continuation/continuation-allowance.json')
    # previous_seconds already includes the original preparation and earlier freezes.
    charged = latest['previous_seconds'] + debit['spent_seconds']
    return seal(dict(version=VERSION, predecessor=receipt, reviewed_evidence=REVIEWED,
                     session=session['identity'], journal=journal['identity'],
                     prepared=prepared, calibration_batch=reused['batch'], corrected_materials=panels['materials'],
                     charged_seconds=charged, original_costs_retained=True, scientific_allowance_transferred=False,
                     current_source_requalification_required=True, historical_qualification_only=True))


def setup_limits(setup, proof):
    """Reserve a full day for freeze inside the original shared setup allowance."""
    if proof['version'] != VERSION or proof['reviewed_evidence'] != REVIEWED or seal(proof) != proof:
        raise Blocked('reviewed round019 owner proof required')
    freeze = 86400
    prepare = setup['prepare'] + setup['freeze'] - freeze - math.ceil(proof['charged_seconds'])
    if prepare <= 120:
        raise Blocked('insufficient original preparation allowance')
    return dict(setup, prepare=prepare, freeze=freeze)


def check_storage(root, base):
    # Use the exact reviewed projection helper, kept byte-identical for old evidence.
    helper = 'specs/013-progressive-mathematical-discovery/validation/continue-round019-storage.py'
    checked_file(root, dict(path=helper, bytes=22284,
                            sha256='sha256:089328350d68b98e090c17ae55921e94d3cc5662eb942897204717128fdf616c'))
    prepared = load(confined(root, base / 'prepared/prepared.json'), maximum=CAMPAIGN_DOCUMENT_BYTES)
    report = runpy.run_path(str(root / helper))['capacity'](root, prepared)
    path = base / 'storage-preflight.json'
    from .preparation_recovery import revise
    revise(path, report)
    if report['status'] != 'READY':
        raise Blocked(f"freeze needs {report['additional_bytes']} more bytes and "
                      f"{report['additional_inodes']} more inodes; see {path}")
    return report
