import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from oeis_learn.pipeline_validation.progressive_cli import COMMANDS
from oeis_learn.pipeline_validation.progressive_commands import main

ROOT = Path(__file__).resolve().parents[2]


def test_exact_twelve_commands_and_no_start_step():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "foundation_local", ROOT / "scripts/foundation_local.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = module.parser()
    assert len(COMMANDS) == 12
    for name, args in COMMANDS.items():
        command = ["discovery", name]
        for arg in args:
            command += ["--" + arg, "x"]
        assert p.parse_args(command).operation == name
    with pytest.raises(SystemExit):
        p.parse_args(["discovery", "run", "--start-step", "3"])


def test_stdlib_paths_under_no_site(tmp_path):
    code = """import sys
+sys.path.insert(0,'src')
+from oeis_learn.pipeline_validation.progressive_commands import main
+from types import SimpleNamespace
+from pathlib import Path
+root=Path(sys.argv[1])
+assert main(SimpleNamespace(operation='init',reference_campaign='missing',output='out'),root,None)==0
+for operation in ['status','freeze','report','verify-bundle']:
+    args=SimpleNamespace(operation=operation,run='out',blueprint='out/blueprint.json',prepared='missing',qualification='missing',output='unused',archive='missing')
+    assert main(args,root,None) in (0,1,2,3)
+assert not {'torch','jsonschema','optuna'} & set(sys.modules)
+""".replace("\n+", "\n")
    proc = subprocess.run(
        [sys.executable, "-S", "-c", code, str(tmp_path)], cwd=ROOT, text=True, capture_output=True
    )
    assert proc.returncode == 0, proc.stderr


def test_init_missing_inputs_together_status_is_read_only(tmp_path, capsys):
    args = SimpleNamespace(operation="init", reference_campaign="missing", output="out")
    assert main(args, tmp_path, None) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mechanical_status"] == "COMPLETE" and result["missing_evidence"]
    before = {str(p.relative_to(tmp_path)): p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    assert main(SimpleNamespace(operation="status", run="out"), tmp_path, None) == 0
    assert before == {
        str(p.relative_to(tmp_path)): p.stat().st_mtime_ns for p in tmp_path.rglob("*")
    }
    assert main(args, tmp_path, None) == 3


def test_init_audits_explicit_original_inventory_without_guessed_paths(tmp_path):
    from oeis_learn.generalization.progressive.freeze import inventory
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation.identity import seal

    base = tmp_path / "prior"
    base.mkdir()
    names = [
        "retained_checkpoint",
        "original_qualification",
        "fresh-002_exposure",
        "prepared_groups",
        "measured_feasibility",
    ]
    refs = []
    for name in names:
        path = base / (name + ".json")
        path.write_text("{}")
        refs.append(
            dict(name=name, role="custodian", kind="source", artifact=reference(tmp_path, path))
        )
    (base / "retained-inputs.json").write_text(
        json.dumps(seal(dict(version="discovery-retained-inputs/v1", references=refs)))
    )
    result = inventory(tmp_path, "prior")
    assert result["unresolved"] == []
    assert len(result["references"]) == 5
    (base / "retained_checkpoint.json").write_text("changed")
    result = inventory(tmp_path, "prior")
    assert "retained_checkpoint" in result["unresolved"]
    assert "invalid:retained_checkpoint" in result["unresolved"]
