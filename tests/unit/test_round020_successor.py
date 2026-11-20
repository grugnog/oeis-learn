from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import reference
from oeis_learn.pipeline_validation import progressive_round020_successor as successor
from oeis_learn.pipeline_validation.identity import digest, seal
from oeis_learn.pipeline_validation.progressive_preparation_owner import failure_proof
from oeis_learn.pipeline_validation.progressive_workstation_state import Blocked, Session, read, write


@pytest.fixture
def original(tmp_path, monkeypatch):
    folder = tmp_path / successor.SESSION
    base = folder / 'prepare-research'
    def put(path, **fields):
        write(path, seal(fields))
        return reference(tmp_path, path)
    for name in successor.FILES:
        put(folder / name, name=name)
    receipt = seal(dict(recipient=successor.SESSION, current_source='old'))
    write(base / 'preparation-data-reuse.json', receipt)
    put(folder / 'preparation-owner.json', receipt=receipt)
    put(folder / 'session.json', binding=dict(source='old'))
    put(folder / 'preparation-owner-proof.json', charged_seconds=127683.8)
    task = put(base / 'frozen/tasks/first.json', original=True)
    put(folder / 'journal.json', steps={'prepare-research/freeze': dict(
        status='COMPLETE', outputs=[task], history=[dict(status='BLOCKED', elapsed_seconds=1)], elapsed_seconds=6845),
        'prepare-research/acceptance': dict(status='FAILED', elapsed_seconds=373)})
    put(base / 'frozen/result.json', campaign=digest('campaign'))
    put(base / 'acceptance/private/host-traceback.txt', failure='actual HIP test')
    put(base / 'acceptance/result.json', producer_error=dict(local_traceback=reference(
        base / 'acceptance', 'private/host-traceback.txt')))
    prepared = put(base / 'prepared/prepared.json', original=True)
    put(base / 'prepared/result.json', metrics=dict(prepared=reference(base / 'prepared', 'prepared.json')))
    put(base / 'calibration-reuse/calibration-reuse.json', **{k: prepared for k in
        ('original_qualification', 'qualification', 'blueprint', 'prepared', 'request', 'batch')})
    put(folder / 'panel-data-reuse.json', original_proof=prepared, materials=prepared, original_materials=prepared)
    monkeypatch.setattr(successor, 'REVIEWED', digest([reference(folder, n) for n in successor.FILES]))
    return tmp_path, folder, receipt


def test_reviewed_transfer_replays_and_preserves_costs(original):
    root, folder, receipt = original
    proof = failure_proof(root, receipt)
    assert proof == failure_proof(root, receipt)
    assert proof['charged_seconds'] == 134902.8
    assert proof['current_source_requalification_required'] and not proof['scientific_allowance_transferred']
    limits = successor.setup_limits(dict(prepare=261000, freeze=7200, acceptance=7800), proof)
    assert limits == dict(prepare=43057, freeze=86400, acceptance=7800)


@pytest.mark.parametrize('name', ['prepare-research/frozen/tasks/first.json', 'prepare-research/acceptance/result.json',
    'prepare-research/acceptance/acceptance.json', 'prepare-research/acceptance/device/integrated/producer.json',
    'prepare-research/prepared/prepared.json', 'run-research', 'assess', 'control'])
def test_changed_or_started_predecessor_blocks(original, name):
    root, folder, receipt = original
    changed = root / 'run-state/discovery-control' / digest('campaign')[7:] if name == 'control' else folder / name
    write(changed, seal(dict(changed=True)))
    with pytest.raises(ValueError):
        failure_proof(root, receipt)


@pytest.fixture
def retry(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_acceptance_retry as retry
    from oeis_learn.pipeline_validation import progressive_handoff, progressive_device_acceptance
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    session = Session(tmp_path, 'run-state/retry', {'source': 'source', 'engine':'podman'}, 'prepare-research')
    w = Workflow(session, SimpleNamespace(phase='prepare-research', engine='podman', memory_gib=64,
                                          cpus=8, render_node=None))
    frozen = w.base / 'frozen/campaign.json'
    runtime = seal(dict(payload=dict(scope=dict(model=digest({'width':1}), images=['cpu','gpu'], execution_policy='fp32'))))
    write(w.base / 'frozen/runtime.json', runtime)
    write(w.base / 'matrix/matrix.json', seal(dict(hardware='hardware')))
    write(frozen, seal({'payload': {'runtime': runtime['identity']}}))
    campaign = read(frozen)['identity']
    session.step('freeze', lambda: (True, [frozen]))
    monkeypatch.setattr(progressive_handoff, 'named_plan', lambda *args: dict(seconds=7200))
    monkeypatch.setattr(progressive_device_acceptance, 'consume_receipt', lambda root, receipt: receipt)
    return w, retry, campaign


def fake_attempt(w, campaign, mode, command, logs, **kwargs):
    target = w.root / command[command.index('--output')+1]
    for stage in ('software', 'device', 'runtime-inspection/device/logs', 'runtime-inspection/image-0/logs',
                  'runtime-inspection/image-1/logs', 'runtime-inspection/selected-device'):
        directory = target / stage
        write(directory / 'command.json', dict(argv=['podman', 'run']))
        write(directory / 'process.json', dict(returncode=0, cleanup_failure=None))
        cleanup = directory.parent if directory.parent.name.startswith('image-') else directory
        write(cleanup / 'cleanup.json', dict(acknowledged=True, within_bound=mode != 'unsafe', failure=None))
    code = 0 if mode == 'pass' else 1
    write(logs / 'process.json', dict(outcome='COMPLETE', returncode=code, cleanup_failure=None))
    if mode in ('interrupt', 'unsafe'):
        raise KeyboardInterrupt
    if code == 0:
        original = target / 'producer.json'
        write(original, seal(dict(actual='synthetic unit test')))
        ref = reference(target, original)
        runtime = read(w.base / 'frozen/runtime.json')
        write(target / 'request.json', seal(dict(profile={'model':{'width':1}}, runtime=runtime['identity'], source='source',
                                                engine='podman', image='gpu', hardware='hardware', execution_policy='fp32')))
        write(target / 'acceptance.json', seal(dict(source='source', campaign=campaign, software=ref, device=ref,
                                                    request=reference(target,'request.json'))))
        write(target / 'result.json', seal(dict(operation='acceptance', campaign=campaign, mechanical_status='COMPLETE',
                                                metrics=dict(acceptance=reference(target, 'acceptance.json')))))


@pytest.mark.parametrize('stop', ['fail', 'interrupt'])
def test_retry_preserves_attempts_then_reuses_publication(retry, monkeypatch, stop):
    from oeis_learn.pipeline_validation import supervision
    w, api, campaign = retry
    modes = iter((stop, 'pass'))
    monkeypatch.setattr(supervision, 'launch', lambda *a, **kw: fake_attempt(w, campaign, next(modes), *a, **kw))
    if stop == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            api.run(w, 7800)
    else:
        assert api.run(w, 7800) is None
    result = api.run(w, 7800)
    assert result['output'].endswith('/acceptance/attempts/0001')
    assert api.accepted_folder(w.root, w.base, w.s.journal) == w.root / result['output']
    assert api.run(w, 7800) == result
    # Simulate loss of the outer journal update after the producer successfully exited.
    row = w.s.journal['steps']['prepare-research/acceptance']
    row['status'] = 'INTERRUPTED'
    path = w.base / 'acceptance-allowance.json'
    ledger = read(path)
    last = ledger['attempts'].pop()
    ledger['active'] = {k:last[k] for k in ('ordinal','reserved_seconds','started_unix')}
    write(path, seal(ledger))
    assert api.run(w, 7800) == result
    assert read(path)['spent_seconds'] == 7200 and read(path)['active'] is None
    assert len(list((w.base / 'acceptance/attempts').iterdir())) == 2
    write(w.root / result['output'] / 'producer.json', seal(dict(changed=True)))
    with pytest.raises(ValueError, match='output changed'):
        api.run(w, 7800)


def test_unacknowledged_cleanup_blocks_a_second_device_attempt(retry, monkeypatch):
    from oeis_learn.pipeline_validation import supervision
    w, api, campaign = retry
    monkeypatch.setattr(supervision, 'launch', lambda *a, **kw: fake_attempt(w, campaign, 'unsafe', *a, **kw))
    assert api.run(w, 7800) is None
    monkeypatch.setattr(supervision, 'launch', lambda *a, **kw: pytest.fail('second device owner'))
    assert api.run(w, 7800) is None
    assert len(list((w.base / 'acceptance/attempts').iterdir())) == 1


@pytest.mark.parametrize('change', ['image', 'hardware', 'missing-device-proof'])
def test_orphaned_success_with_altered_proof_cannot_launch(retry, monkeypatch, change):
    from oeis_learn.pipeline_validation import supervision
    w, api, campaign = retry
    monkeypatch.setattr(supervision, 'launch', lambda *a, **kw: fake_attempt(w, campaign, 'pass', *a, **kw))
    result = api.run(w, 7800)
    target = w.root / result['output']
    if change == 'missing-device-proof':
        (target / 'device/process.json').unlink()
    else:
        request = read(target / 'request.json')
        write(target / 'request.json', seal(dict(request, **{change:'other'})))
        receipt = read(target / 'acceptance.json')
        write(target / 'acceptance.json', seal(dict(receipt, request=reference(target, 'request.json'))))
        terminal = read(target / 'result.json')
        write(target / 'result.json', seal(dict(terminal, metrics={'acceptance':reference(target, 'acceptance.json')})))
    w.s.journal['steps']['prepare-research/acceptance']['status'] = 'INTERRUPTED'
    monkeypatch.setattr(supervision, 'launch', lambda *a, **kw: pytest.fail('ran after tampering'))
    assert api.run(w, 7800) is None
    assert w.s.journal['steps']['prepare-research/acceptance']['status'] != 'COMPLETE'


def test_storage_block_never_opens_acceptance(tmp_path):
    from oeis_learn.pipeline_validation.progressive_acceptance_retry import run
    s = Session(tmp_path, 'run-state/blocked', {}, 'prepare-research')
    w = SimpleNamespace(root=tmp_path, base=s.folder / 'prepare-research', s=s)
    assert run(w, 7800) is None
    assert s.journal['steps']['prepare-research/acceptance']['status'] == 'BLOCKED'
    assert not (w.base / 'acceptance-allowance.json').exists()
