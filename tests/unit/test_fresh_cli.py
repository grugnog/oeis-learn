import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "fresh_cli", Path(__file__).parents[2] / "scripts/foundation_run.py"
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


def test_effective_model_is_explicit_and_copied_client_is_used(tmp_path):
    value = cli.configuration(cli.ROOT / "configs/foundation/fresh_pilot.yaml")
    assert value["run"]["model"]["encoder"] == "integer_bytes/v1"
    assert value["run"]["model"]["attention_backend"] == "eager"
    (tmp_path / "system").mkdir()
    (tmp_path / "system/lab-gpu.py").write_text("# updated client")
    with patch.object(cli, "ROOT", tmp_path), patch.object(cli.subprocess, "run") as run:
        cli.remote("run", "inputs", "runs/new", "smoke")
        argv = run.call_args.args[0]
        assert argv[:3] == [cli.sys.executable, str(tmp_path / "system/lab-gpu.py"), "client"]
        assert "foundation" in argv and "exec" not in argv


def test_resume_does_not_accept_new_settings_or_budget():
    with pytest.raises(SystemExit):
        cli.main(["resume", "--run", "runs/new", "--seconds", "7200"])


def test_paths_outside_shared_checkout_reject(tmp_path):
    with (
        patch.object(cli, "ROOT", tmp_path / "project"),
        pytest.raises(ValueError, match="shared checkout"),
    ):
        cli.relative(tmp_path)
