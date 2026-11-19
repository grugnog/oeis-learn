"""Round019 successor checks with a small locally authenticated predecessor."""
from pathlib import Path
import pytest

from oeis_learn.generalization.progressive.contracts import reference
from oeis_learn.pipeline_validation import progressive_round019_successor as successor
from oeis_learn.pipeline_validation.identity import seal, digest
from oeis_learn.pipeline_validation.progressive_preparation_owner import failure_proof
from oeis_learn.pipeline_validation.progressive_workstation_state import write, read, Blocked


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    root = tmp_path
    folder = root / successor.SESSION
    base = folder / 'prepare-research'
    def put(path, **fields):
        write(path, seal(fields))
        return reference(root, path)
    for name in successor.FILES:
        put(folder / name, original=name)
    data = put(base / 'preparation-data-reuse.json', version='progressive-preparation-data-reuse/v1', recipient=successor.SESSION, current_source='old')
    receipt = read(root / data['path'])
    put(folder / 'session.json', binding={'source': 'old'})
    put(folder / 'preparation-owner.json', receipt=receipt)
    prepared = put(base / 'prepared/prepared.json', example=True)
    put(base / 'prepared/result.json', metrics=dict(prepared=reference(base / 'prepared', 'prepared.json')))
    trace = put(base / 'frozen/private/host-traceback.txt', failure='closed failure')
    put(base / 'frozen/result.json', producer_error=dict(local_traceback=reference(base / 'frozen', 'private/host-traceback.txt')))
    put(base / 'calibration-reuse/calibration-reuse.json', **{key: prepared for key in (
        'original_qualification', 'qualification', 'blueprint', 'prepared', 'request', 'batch')})
    put(folder / 'panel-data-reuse.json', original_proof=prepared, materials=prepared, original_materials=prepared)
    put(base / 'storage-continuation/continuation.json', original_script=trace, script=trace)
    put(base / 'segv-continuation/continuation.json', script=trace, crash_logs=[trace], previous_seconds=88015.346)
    put(base / 'segv-continuation/continuation-allowance.json', spent_seconds=39668.532)
    reviewed = digest([reference(folder, name) for name in successor.FILES])
    monkeypatch.setattr(successor, 'REVIEWED', reviewed)
    return root, folder, receipt


def test_exact_predecessor_replays_without_refunding_elapsed_time(stopped):
    root, folder, receipt = stopped
    record = failure_proof(root, receipt)
    assert record == failure_proof(root, receipt) and record['current_source_requalification_required']
    limits = successor.setup_limits(dict(prepare=261000, freeze=7200, acceptance=7800), record)
    assert limits == dict(prepare=54116, freeze=86400, acceptance=7800)
    assert not record['scientific_allowance_transferred']
    with pytest.raises(Blocked):
        successor.setup_limits(dict(prepare=5000, freeze=7200), record)


@pytest.mark.parametrize('changed', ['journal.json', 'prepare-research/frozen/private/host-traceback.txt',
    'prepare-research/prepared/prepared.json', 'prepare-research/segv-continuation/continuation-allowance.json',
    'prepare-research/acceptance', 'prepare-research/frozen/campaign.json', 'run-research', 'assess'])
def test_changed_or_started_predecessor_is_preserved(stopped, changed):
    root, folder, receipt = stopped
    write(folder / changed, seal(dict(changed=True)))
    with pytest.raises(ValueError):
        failure_proof(root, receipt)


def test_capacity_block_publishes_deficit_preserving_original(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_round019_successor as m
    from shutil import copyfile
    path = Path('specs/013-progressive-mathematical-discovery/validation/continue-round019-storage.py')
    (tmp_path / path.parent).mkdir(parents=True)
    copyfile(Path(__file__).resolve().parents[2] / path, tmp_path / path)
    base = tmp_path / 'session'
    write(base / 'prepared/prepared.json', seal(dict(example=True)))
    reports = iter([seal(dict(status='BLOCKED', additional_bytes=123, additional_inodes=0)), seal(dict(status='READY'))])
    monkeypatch.setattr(m.runpy, 'run_path', lambda path: dict(capacity=lambda root, prepared: next(reports)))
    with pytest.raises(Blocked, match='123 more bytes'):
        m.check_storage(tmp_path, base)
    assert m.check_storage(tmp_path, base)['status'] == 'READY'
    archived, = (base / 'recovery-records').iterdir()
    assert read(archived)['additional_bytes'] == 123


def test_transfer_fences_round019_and_keeps_calibration_donor(stopped, monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.progressive_preparation_owner import claim_owner, VERSION
    from oeis_learn.pipeline_validation.progressive_calibration_reuse_workstation import fenced_original

    root, folder, receipt = stopped
    donor = 'run-state/discovery/013-research-round-009'
    donor_receipt = seal(dict(receipt, recipient=donor, current_source='calibration'))
    previous = root / 'history/original.json'
    write(previous, donor_receipt)
    blueprint = root / donor / 'prepare-research/construction/blueprint.json'
    write(blueprint, seal(dict(prepared=True)))
    proof = seal(dict(version='progressive-preparation-owner-proof/v5', predecessor=donor_receipt,
                      blueprint=reference(root, blueprint), original_costs_retained=True,
                      scientific_allowance_transferred=False))
    write(root / 'history/calibration.json', proof)
    owner = seal(dict(version=VERSION, receipt=receipt, previous=reference(root, previous),
                      proof=reference(root, 'history/calibration.json')))
    claim = root / 'run-state/claims/active.json'
    write(claim, owner)
    write(folder / 'preparation-owner.json', owner)
    (folder / 'session.lock').touch()
    monkeypatch.setattr(successor, 'REVIEWED', digest([reference(folder, n) for n in successor.FILES]))
    destination = 'run-state/discovery/013-research-round-020'
    new = seal(dict(receipt, recipient=destination, current_source='fresh qualification required'))
    workflow = SimpleNamespace(root=root, s=SimpleNamespace(folder=root / destination),
                               args=SimpleNamespace(reuse_calibration_session=donor))
    for _ in range(2):
        claim_owner(root, claim, new)
        assert fenced_original(workflow) == blueprint.parent.parent
    with pytest.raises(Blocked, match='superseded'):
        claim_owner(root, claim, receipt)
