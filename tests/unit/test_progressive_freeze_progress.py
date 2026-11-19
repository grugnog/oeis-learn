"""Process death, evidence invalidation and commit boundaries for research freeze."""
import json
import multiprocessing
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import load, seal
from oeis_learn.generalization.progressive.freeze_progress import FreezeProgress
from oeis_learn.pipeline_validation.progressive_commands import freeze_command
from oeis_learn.pipeline_validation.progressive_workstation_state import write


def inputs(root):
    args = SimpleNamespace(blueprint='blueprint.json', prepared='prepared',
                           qualification='qualification', output='frozen')
    for path in ('blueprint.json', 'prepared/prepared.json', 'qualification/qualification.json'):
        write(root / path, {'input': path})
    return args


def test_killed_study_resumes_only_committed_choices(tmp_path, monkeypatch):
    from tests.helpers.progressive_study import build
    from oeis_learn.pipeline_validation import progressive_learner
    from oeis_learn.pipeline_validation.progressive_study import validate_frozen

    args = inputs(tmp_path)
    campaign, _, _, _ = build(tmp_path, monkeypatch, trials=6)
    def run_until_killed():
        with FreezeProgress(tmp_path, 'frozen').lock() as progress:
            progress.bind(args)
            progress.begin('killed-at-commit')
            save = progress.save
            def kill_after_save(key):
                save(key)
                os.kill(os.getpid(), signal.SIGKILL)
            progress.save = kill_after_save
            validate_frozen(tmp_path, campaign, progress=progress)
    child = multiprocessing.get_context('fork').Process(target=run_until_killed)
    child.start()
    child.join(40)
    assert child.exitcode == -signal.SIGKILL
    folder = tmp_path / 'frozen.freeze-progress'
    assert load(folder / 'freeze-progress.json')['validated_this_attempt'] == 1
    assert not (tmp_path / 'frozen/campaign.json').exists()

    calls = []
    validate_input = progressive_learner.validate_input
    def observe(package, task, **kw):
        calls.append((task['payload']['id'], task['payload']['trial_choice']))
        return validate_input(package, task, **kw)
    monkeypatch.setattr(progressive_learner, 'validate_input', observe)
    with FreezeProgress(tmp_path, 'frozen').lock() as progress:
        progress.bind(args)
        progress.begin('resumed')
        validate_frozen(tmp_path, campaign, progress=progress)
        assert progress.reused == 1 and progress.validated == len(calls) > 0
        assert ('train', 0) not in calls and ('train', 1) in calls
        with pytest.raises(BlockingIOError), FreezeProgress(tmp_path, 'frozen').lock():
            pass

    target = tmp_path / 'study-training.json'
    target.write_bytes(target.read_bytes() + b' ')
    with FreezeProgress(tmp_path, 'frozen').lock() as progress:
        progress.bind(args)
        progress.begin('changed-package')
        with pytest.raises(ValueError):
            validate_frozen(tmp_path, campaign, progress=progress)
        assert progress.reused == 0
    write(tmp_path / 'prepared/prepared.json', {'changed': True})
    with FreezeProgress(tmp_path, 'frozen').lock() as progress, pytest.raises(ValueError, match='inputs or validator changed'):
        progress.bind(args)


@pytest.mark.parametrize('kill_at', ['publish', 'promote'])
def test_validated_result_survives_kill_during_publication(tmp_path, monkeypatch, kill_at):
    from oeis_learn.generalization.progressive import freeze as producer
    from oeis_learn.generalization.progressive.campaign import frozen_task_records
    from tests.unit.test_progressive_freeze import campaign as fixture
    args = inputs(tmp_path)
    runtime = seal(dict(version='test-runtime'))
    campaign = fixture()
    campaign['payload']['runtime'] = runtime['identity']
    campaign = seal(campaign)
    tasks = frozen_task_records(campaign)
    def prepared(*a, **kw):
        with (tmp_path / 'calls').open('a') as f:
            print('freeze', file=f)
        return campaign, runtime, tasks
    monkeypatch.setattr(producer, 'freeze', prepared)
    def die_before_rename():
        def kill(self, *args):
            os.kill(os.getpid(), signal.SIGKILL)
        setattr(FreezeProgress, kill_at, kill)
        freeze_command(args, tmp_path)
    child = multiprocessing.get_context('fork').Process(target=die_before_rename)
    child.start()
    child.join(30)
    assert child.exitcode == -signal.SIGKILL
    assert not (tmp_path / 'frozen/campaign.json').exists()
    for _ in range(2):
        result, code = freeze_command(args, tmp_path)
        assert code == 0 and result['campaign'] == campaign['identity']
    assert (tmp_path / 'calls').read_text() == 'freeze\n'
    assert load(tmp_path / 'frozen.freeze-progress/freeze-progress.json')['phase'] == 'complete'
    stored, = (tmp_path / 'frozen/tasks').iterdir()
    assert json.loads(stored.read_bytes()) == tasks[0]
    stored.write_text('{}')
    with pytest.raises(ValueError):
        freeze_command(args, tmp_path)
    assert (tmp_path / 'calls').read_text() == 'freeze\n'


def test_rejects_foreign_or_damaged_work(tmp_path):
    args = inputs(tmp_path)
    with FreezeProgress(tmp_path, 'frozen').lock() as p:
        p.bind(args)
        p.begin('writer')
        key = p.study_key('a', 1, 'channel')
        p.save(key)
        assert not p.completed(p.study_key('a', 1, 'different-channel'))
        with p.database:
            p.database.execute('UPDATE passed SET receipt=?', ('damaged',))
        with pytest.raises(ValueError, match='damaged'):
            p.completed(key)
    other = tmp_path / 'legacy'
    write(other / 'result.json', seal(dict(mechanical_status='FAILED')))
    args.output = 'legacy'
    with pytest.raises(ValueError, match='no checkpoint binding'):
        freeze_command(args, tmp_path)
    assert load(other / 'result.json')['mechanical_status'] == 'FAILED'


def test_workstation_freeze_spends_one_allowance_and_keeps_attempt_logs(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import supervision
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session, read
    from oeis_learn.generalization.progressive.contracts import reference

    session = Session(tmp_path, 'run-state/test', dict(source='test'), 'prepare-research')
    args = SimpleNamespace(phase=session.phase, engine='podman', memory_gib=64, cpus=8, render_node=None)
    workflow = Workflow(session, args)
    attempts = []
    def dispatch(argv, logs, **limits):
        attempts.append(logs.name)
        code = 2 if len(attempts) == 1 else 0
        write(logs / 'process.json', {'returncode': code})
        if code == 0:
            folder = tmp_path / 'frozen'
            write(folder / 'campaign.json', seal(dict(version='fixture')))
            write(folder / 'result.json', seal(dict(mechanical_status='COMPLETE', operation='freeze', campaign='fixture')))
            write(folder / 'freeze-publication.json', seal(dict(campaign='fixture', files=[reference(folder, 'campaign.json')])))
    monkeypatch.setattr(supervision, 'launch', dispatch)
    def run():
        return workflow.run_command('freeze', 'freeze', output='frozen', seconds=1200)

    with session.own():
        workflow.run_command('freeze', 'freeze', output='unresolved', dependencies=('prepare-research/prepare',))
        assert not attempts
        assert run() is None
        assert run()['exit_code'] == 0
        assert run()['exit_code'] == 0
        assert attempts == ['0000', '0001']
        allowance = workflow.base / 'freeze-allowance.json'
        spent = read(allowance)
        assert len(spent['attempts']) == 2 and spent['attempts'][1]['reserved_seconds'] < 1200
        # Retrying after the evidence changes cannot create another allowance.
        write(tmp_path / 'frozen/campaign.json', {'changed': True})
        allowance.unlink()
        assert run() is None
        assert 'allowance missing' in session.journal['steps']['prepare-research/freeze']['reason']
        assert attempts == ['0000', '0001']
