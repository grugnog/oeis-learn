import errno
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "fresh_cli", Path(__file__).parents[2] / "scripts/foundation_tasks.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def config():
    return yaml.safe_load((cli.ROOT / "configs/foundation/fresh_pilot.yaml").read_text())


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["run"].update(seconds=432000),
        lambda c: c["run"].update(training_seconds=18000),
        lambda c: c["archive"].update(training_prompts=4),
        lambda c: c["data"].update(max_records=True),
        lambda c: c["run"]["evaluation"].update(attempts=33),
    ],
)
def test_unknown_or_unbounded_configuration_rejects(tmp_path, change):
    value = config()
    change(value)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):
        cli.configuration(path)


def test_effective_model_is_explicit():
    value = cli.configuration(cli.ROOT / "configs/foundation/fresh_pilot.yaml")
    assert value["run"]["model"]["encoder"] == "integer_bytes/v1"
    assert value["run"]["model"]["attention_backend"] == "eager"


def test_resume_does_not_accept_new_settings_or_budget():
    with pytest.raises(SystemExit):
        cli.main(["resume", "--run", "runs/new", "--seconds", "7200"])


def test_paths_outside_shared_checkout_reject(tmp_path):
    with (
        patch.object(cli, "ROOT", tmp_path / "project"),
        pytest.raises(ValueError, match="shared checkout"),
    ):
        cli.relative(tmp_path)


@pytest.mark.parametrize("output", [None, "tmp-custom-inputs"])
def test_prepare_command_dispatches_to_input_preparation(tmp_path, monkeypatch, capsys, output):
    from oeis_learn.fresh import inputs

    value = config()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    monkeypatch.chdir(tmp_path)
    argv = ["prepare", "--config", str(path)]
    if output is not None:
        argv += ["--output", output]
    with (
        patch.object(cli, "ROOT", tmp_path),
        patch.object(inputs, "prepare", return_value={"identity": "prepared-identity"}) as prepare,
    ):
        assert cli.main(argv) == 0
        effective, cache, destination = prepare.call_args.args
        assert effective["run"]["model"]["encoder"] == "integer_bytes/v1"
        assert cache == tmp_path / value["data"]["cache"]
        assert destination == tmp_path / (output or "tmp-fresh-inputs")
    assert json.loads(capsys.readouterr().out) == {
        "identity": "prepared-identity",
        "output": output or "tmp-fresh-inputs",
        "qualified": False,
    }


def test_prepare_command_reports_preparation_failure(tmp_path, monkeypatch, capsys):
    from oeis_learn.fresh import inputs

    value = config()
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(cli, "ROOT", tmp_path),
        patch.object(cli, "configuration", return_value=value),
        patch.object(inputs, "prepare", side_effect=OSError("admission budget exhausted")),
    ):
        assert cli.main(["prepare"]) == 1
    assert "ERROR: admission budget exhausted" in capsys.readouterr().err


def test_prepare_open_file_exhaustion_reports_limit_and_traceback(tmp_path, monkeypatch, capsys):
    from oeis_learn.fresh import inputs

    value = config()
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(cli, "ROOT", tmp_path),
        patch.object(cli, "configuration", return_value=value),
        patch.object(inputs, "prepare", side_effect=OSError(errno.EMFILE, "too many open files")),
    ):
        assert cli.main(["prepare"]) == 1
    error = capsys.readouterr().err
    assert "Open-file limit: soft=" in error and "hard=" in error
    assert "Traceback (most recent call last)" in error
    assert "too many open files" in error


@pytest.mark.parametrize("inputs_only", [False, True])
def test_check_can_validate_cpu_inputs_without_nested_gpu_transport(
    tmp_path, monkeypatch, inputs_only
):
    from oeis_learn.fresh import inputs

    monkeypatch.chdir(tmp_path)
    argv = ["check", "--inputs", "tmp-inputs"] + (["--inputs-only"] if inputs_only else [])
    with (
        patch.object(cli, "ROOT", tmp_path),
        patch.object(inputs, "check_inputs", return_value={"identity": "input-identity"}) as check,
    ):
        assert cli.main(argv) == 0
        check.assert_called_once_with(tmp_path / "tmp-inputs")
