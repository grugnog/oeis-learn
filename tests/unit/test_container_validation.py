"""Portable stage evidence cannot survive failure, skipped GPU work or tampering."""

from unittest.mock import Mock, patch

import pytest

from oeis_learn.container_validation import validate
from oeis_learn.fresh.common import read
from tests.unit import test_foundation_local as fixtures

host = fixtures.host


def launch(host, *, skipped=False, missing=False, failure=False, environment=b"GPU-A"):
    pending = {}

    def command(**options):
        pending.update(options)
        return ["engine", "run"]

    process = Mock()
    process.wait.return_value = 1 if failure else 0
    process.poll.return_value = 0

    def spawn(*args, **kwargs):
        out = pending["mounts"][1][0]
        assert pending["environment"]["TMPDIR"] == "/work/output/.runtime/tmp"
        (out / ".runtime/cache/compiler.json").write_text("not evidence")
        if "-c" in pending["argv"]:
            kwargs["stdout"].write(environment)
        if "pytest" in pending["argv"] and not missing:
            argv = pending["argv"]
            assert argv[argv.index("-o") + 1] == "cache_dir=/work/output/.runtime/cache/pytest"
            cache = out / ".runtime/cache/pytest"
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "nodeids.json").write_text("[]")
            (out / "results.xml").write_text(
                '<testsuite><testcase name="actual">'
                + ("<skipped/>" if skipped else "")
                + "</testcase></testsuite>"
            )
        return process

    host.engine.command.side_effect = command
    return patch("oeis_learn.container_validation.subprocess.Popen", side_effect=spawn)


def test_verified_receipts_skip_but_output_tampering_reruns(host, tmp_path):
    output = tmp_path / "validation"
    with launch(host) as spawn:
        validate(host, "cpu", output)
        assert spawn.call_count == 2
        validate(host, "cpu", output)
        assert spawn.call_count == 2
        (output / "cpu-regressions/stdout.log").write_text("changed")
        validate(host, "cpu", output)
        assert spawn.call_count == 3
    assert read(output / "validation.json")["qualified"] is False
    for stage in ("environment", "cpu-regressions"):
        assert all(
            ".runtime" not in str(path) for path in read(output / stage / "receipt.json")["files"]
        )


@pytest.mark.parametrize("problem", ["cleanup", "exit", "no-junit"])
def test_failure_never_records_success(host, tmp_path, problem):
    if problem == "cleanup":
        host.engine.remove.side_effect = RuntimeError("cleanup failed")
    with launch(host, failure=problem == "exit", missing=problem == "no-junit"):
        with pytest.raises(RuntimeError, match="failed; inspect"):
            validate(host, "cpu", tmp_path / "validation")
    receipts = list((tmp_path / "validation").glob("*/receipt.json"))
    assert receipts and any(read(receipt)["status"] == "FAIL" for receipt in receipts)
    assert read(tmp_path / "validation/validation.json")["status"] == "FAIL"


def test_gpu_skips_fail_and_do_not_run_containment(host, tmp_path):
    with launch(host, skipped=True), pytest.raises(RuntimeError):
        validate(host, "gpu", tmp_path / "validation")
    assert read(tmp_path / "validation/gpu-regressions/receipt.json")["status"] == "FAIL"


def test_gpu_environment_is_rechecked_and_changed_hardware_invalidates_receipts(host, tmp_path):
    output = tmp_path / "validation"
    with patch.object(host, "pipeline"), launch(host) as spawn:
        validate(host, "gpu", output)
        assert spawn.call_count == 3
        validate(host, "gpu", output)
        assert spawn.call_count == 4
    with patch.object(host, "pipeline"), launch(host, environment=b"GPU-B") as spawn:
        validate(host, "gpu", output)
        assert spawn.call_count == 3


def test_failed_rerun_removes_old_summary_and_corrupt_receipt_is_not_a_pass(host, tmp_path):
    output = tmp_path / "validation"
    with launch(host):
        validate(host, "cpu", output)
    (output / "cpu-regressions/receipt.json").write_text("truncated")
    with launch(host, missing=True), pytest.raises(RuntimeError):
        validate(host, "cpu", output)
    assert read(output / "validation.json")["status"] == "FAIL"
    assert read(output / "cpu-regressions/receipt.json")["status"] == "FAIL"


@pytest.mark.parametrize('suite', ['cpu', 'gpu', 'bigint-cpu', 'bigint-gpu', 'bigint-retirement'])
@pytest.mark.parametrize('engine_name', ['docker', 'podman'])
def test_all_validation_scratch_is_disk_backed_accounted_and_private(host, tmp_path, monkeypatch, suite, engine_name):
    from oeis_learn.container_runtime import Engine
    from oeis_learn.container_validation import _stages
    from oeis_learn.pipeline_validation.disk import DiskWatchdog
    from oeis_learn.pipeline_validation.bundle import public
    from pathlib import Path
    engine = Engine(engine_name, executable=engine_name, details={})
    monkeypatch.setattr(engine, 'facts', host.engine.facts)
    # Devices are a recorded external boundary; construct real commands only.
    monkeypatch.setattr(engine, 'device_args', lambda node: ['--device', node])
    monkeypatch.setattr(host, 'select_gpu', lambda: None)
    host.policy['render_node'] = '/dev/dri/renderD128'
    pending = {}; commands = []
    def command(**options):
        pending.clear();pending.update(options)
        built = engine.command(**options);commands.append(built)
        return built
    host.engine.command.side_effect = command
    process = Mock();process.wait.return_value = 0;process.poll.return_value = 0
    def spawn(*args, **kwargs):
        if 'pytest' in pending['argv']:
            argv = pending['argv']
            base = next(a.split('=', 1)[1] for a in argv if a.startswith('--basetemp='))
            assert base == '/work/output/.runtime/pytest'
            assert 'tmp_path_retention_policy=failed' in argv
            assert pending['tmp_bytes'] == 2 << 30  # No larger tmpfs workaround.
            mounts = pending['mounts'];assert len(mounts) == 2
            assert mounts[0][1:] == ('/work/input', False)
            assert mounts[1][1:] == ('/work/output', True)
            out = mounts[1][0];scratch = out / '.runtime/pytest';scratch.mkdir()
            watch = DiskWatchdog(out, out / 'disk');before = watch.usage()
            private = scratch / 'fixture.json';private.write_text('private checkpoint metadata')
            assert watch.usage() == before + private.stat().st_size
            assert not public(Path('.runtime/pytest/fixture.json'))
            (out / 'results.xml').write_text('<testsuite><testcase name="recorded"/></testsuite>')
        return process
    with patch('oeis_learn.container_validation.subprocess.Popen', side_effect=spawn):
        _, results = _stages(host, suite, tmp_path / 'validation')
    assert commands and all('--network=none' in c and '--read-only' in c for c in commands)
    receipt = results[suite + '-regressions']
    assert receipt['status'] == 'PASS' and 'results.xml' in receipt['files']
    assert not any('pytest' in Path(p).parts for p in receipt['files'])



def test_validation_pytest_releases_passed_scratch_and_retains_failures(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from oeis_learn.container_validation import PYTEST_SCRATCH_ARGS
    fixture = tmp_path / 'test_scratch.py'
    fixture.write_text("""def test_pass(tmp_path):
    (tmp_path / 'checkpoint.pt').write_bytes(b'passed fixture')
    (tmp_path.parent.parent / 'passed-path').write_text(str(tmp_path))

def test_fail(tmp_path):
    (tmp_path / 'checkpoint.pt').write_bytes(b'failed fixture')
    (tmp_path.parent.parent / 'failed-path').write_text(str(tmp_path))
    assert False, 'intentional fixture failure'
""")
    args = [('--basetemp=' + str(tmp_path / 'pytest')) if a.startswith('--basetemp=') else a
            for a in PYTEST_SCRATCH_ARGS]
    result = subprocess.run([sys.executable, '-m', 'pytest', str(fixture), *args,
        '-q', '-p', 'no:cacheprovider', '--tb=no'], cwd=tmp_path, timeout=30,
        env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'), capture_output=True, text=True)
    assert result.returncode == 1 and '1 failed, 1 passed' in result.stdout, result.stdout + result.stderr
    assert not Path((tmp_path / 'passed-path').read_text()).exists()
    assert (Path((tmp_path / 'failed-path').read_text()) / 'checkpoint.pt').read_bytes() == b'failed fixture'
