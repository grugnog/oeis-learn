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
    assert not (tmp_path / "validation/validation.json").exists()


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
    assert not (output / "validation.json").exists()
    assert read(output / "cpu-regressions/receipt.json")["status"] == "FAIL"
