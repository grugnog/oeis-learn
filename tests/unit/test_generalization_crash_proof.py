import math
import pytest
from tests.unit.test_foundation_local import host
from oeis_learn.pipeline_validation.identity import digest,seal


def proof():
    from oeis_learn.pipeline_validation.generalization_execution import POLICY_ID
    return seal(dict(version='pac-crash-proof/v1',status='PASS',device='hip',diagnostic_only=True,
        source=digest('source'),execution_policy=POLICY_ID,cadence='pac-acceptance/v1',scale='production',
        crash_exit=86,actual_optimizer_calls=19,saved_updates=0,newer_valid_rewards=7,newer_graded_anchors=2,
        restored_old_weights=True,restored_old_policy=True,crashed_seconds=20.,discarded_tokens=900,
        charged_tokens=1652,useful_tokens=752,cap=4096,updates=16,learner_seconds=35.,segment='recovery-1',
        checkpoint=digest('checkpoint'),device_rng_present=True))


def test_actual_hip_crash_proof_is_required_and_closed():
    from oeis_learn.pipeline_validation.generalization_crash_proof import verify
    value=proof();assert verify(value,source=digest('source'))==value
    for changed in [dict(device='cpu'),dict(source=digest('other')),dict(device_rng_present=False),
        dict(restored_old_policy=False),dict(restored_old_weights=False),dict(crash_exit=0),
        dict(newer_valid_rewards=0),dict(newer_graded_anchors=0),dict(discarded_tokens=0),
        dict(charged_tokens=100),dict(learner_seconds=1.),dict(crashed_seconds=math.nan),
        dict(extra='unbound'),dict(scale='tiny')]:
        with pytest.raises(ValueError):verify(seal(dict(value,**changed)),source=digest('source'))
    with pytest.raises(ValueError):verify(dict(value,updates=17),source=digest('source'))


def test_focused_suite_is_explicit_in_portable_parser():
    from tests.unit.test_generalization_host import host_module
    args=host_module().parser().parse_args(['--engine','podman','validate','--suite','pac-crash-hip','--output','run-state/t072'])
    assert args.suite=='pac-crash-hip' and args.engine=='podman'


def test_focused_host_verifies_original_receipt_without_zip(tmp_path,monkeypatch,capsys):
    import importlib.util,json
    from pathlib import Path
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.identity import file_hash
    path=Path('specs/012-parameterized-adaptive-curriculum/validation/t072.py')
    spec=importlib.util.spec_from_file_location('focused_t072',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'collect_snapshot',lambda root:'source')
    def run(command,**kwargs):
        assert command[-4:]==['--suite','pac-crash-hip','--output','proof']
        assert kwargs['cwd']==tmp_path and kwargs['check'] is False
        stage=tmp_path/'proof/pac-crash-hip-regressions';stage.mkdir(parents=True)
        original=stage/'pac-crash-proof.json';original.write_text(json.dumps(proof()))
        receipt=dict(status='PASS',counts=dict(tests=1,skipped=0),files={'pac-crash-proof.json':file_hash(original)})
        (stage/'receipt.json').write_text(json.dumps(receipt))
        (tmp_path/'proof/validation.json').write_text(json.dumps(dict(suite='pac-crash-hip',image=digest('image'),results={'pac-crash-hip-regressions':receipt})))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',run)
    assert module.main(['--engine','podman','--output','proof'])==0
    value=json.loads((tmp_path/'proof/t072-summary.json').read_text())
    assert value['proof']['discarded_tokens']>0 and value['zip_upload']=='NOT_REQUESTED'
    assert len(capsys.readouterr().out)<10000 and not list(tmp_path.rglob('*.zip'))
    with pytest.raises(SystemExit):module.main(['--engine','podman','--output','proof'])


@pytest.mark.parametrize('evidence',['valid','missing','cpu'])
def test_focused_runner_is_gpu_serial_and_fails_without_hip_evidence(host,tmp_path,monkeypatch,evidence):
    import json
    from unittest.mock import Mock
    from oeis_learn.container_validation import validate
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.pipeline_validation.generalization_execution import startup_environment
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda root:'source')
    pending={}
    def command(**kwargs):
        assert kwargs['gpu'] is True
        assert all(kwargs['environment'][k]==v for k,v in startup_environment().items())
        assert kwargs['environment']['OEIS_REQUIRE_BIGINT_GPU']=='1'
        pending.update(kwargs);return ['engine','run']
    host.engine.command.side_effect=command
    process=Mock();process.wait.return_value=0;process.poll.return_value=0
    def spawn(*args,**kwargs):
        out=pending['mounts'][1][0]
        if 'pytest' in pending['argv']:
            selected=[a for a in pending['argv'] if a.startswith('tests/')]
            assert selected==['tests/integration/test_generalization_crash.py::test_pac_actual_hip_subprocess_crash_and_recovery']
            assert pending['environment']['OEIS_PAC_CRASH_EVIDENCE']=='/work/output/pac-crash-proof.json'
            (out/'results.xml').write_text('<testsuite><testcase name="actual"/></testsuite>')
            if evidence!='missing':
                value=proof() if evidence=='valid' else seal(dict(proof(),device='cpu'))
                (out/'pac-crash-proof.json').write_text(json.dumps(value))
        return process
    monkeypatch.setattr('oeis_learn.container_validation.subprocess.Popen',spawn)
    if evidence=='valid':
        validate(host,'pac-crash-hip',tmp_path/'out')
        assert json.loads((tmp_path/'out/validation.json').read_text())['qualified'] is False
    else:
        with pytest.raises(RuntimeError):validate(host,'pac-crash-hip',tmp_path/'out')
