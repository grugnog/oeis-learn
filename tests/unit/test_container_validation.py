"""Portable stage evidence cannot survive failure, skipped GPU work or tampering."""

from unittest.mock import Mock, patch

import pytest

from oeis_learn.container_validation import validate
from oeis_learn.fresh.common import read
from tests.unit import test_foundation_local as fixtures

host = fixtures.host


def launch(host, *, skipped=False, missing=False, failure=False):
    def command(**options):
        out = options["mounts"][1][0]
        if "pytest" in options["argv"] and not missing:
            (out / "results.xml").write_text(
                '<testsuite><testcase name="actual">'
                + ("<skipped/>" if skipped else "")
                + "</testcase></testsuite>"
            )
        return ["engine", "run"]

    host.engine.command.side_effect = command
    process = Mock()
    process.wait.return_value = 1 if failure else 0
    process.poll.return_value = 0
    return patch("oeis_learn.container_validation.subprocess.Popen", return_value=process)


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


@pytest.mark.parametrize("problem", ["cleanup", "exit", "no-junit"])
def test_failure_never_records_success(host, tmp_path, problem):
    if problem == "cleanup":
        host.engine.remove.side_effect = RuntimeError("cleanup failed")
    with launch(host, failure=problem == "exit", missing=problem == "no-junit"):
        with pytest.raises(RuntimeError, match="failed; inspect"):
            validate(host, "cpu", tmp_path / "validation")
    receipts = list((tmp_path / "validation").glob("*/receipt.json"))
    assert receipts and read(receipts[-1])["status"] == "FAIL"
    assert not (tmp_path / "validation/validation.json").exists()


def test_gpu_skips_fail_and_do_not_run_containment(host, tmp_path):
    with launch(host, skipped=True), pytest.raises(RuntimeError):
        validate(host, "gpu", tmp_path / "validation")
    assert read(tmp_path / "validation/gpu-regressions/receipt.json")["status"] == "FAIL"
