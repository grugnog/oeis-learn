"""Only the installed proposal role depends on the exact tuning release."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path


def test_pinned_optuna_import_and_ask_tell():
    import optuna

    assert importlib.metadata.version("optuna") == "4.5.0"
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=13))
    trial = study.ask()
    trial.suggest_categorical("rate", [0.3, 1, 3])
    study.tell(trial, 0.25)
    assert study.trials[0].value == 0.25


def test_namespace_and_cli_have_no_model_or_proposer_imports():
    root = Path(__file__).resolve().parents[2]
    code = """import sys
sys.path.insert(0, 'src')
import oeis_learn.generalization.progressive
import oeis_learn.pipeline_validation.progressive_cli
assert not {'torch','optuna','jsonschema'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-S", "-c", code], cwd=root, check=True)


def test_cpu_inventory_covers_new_tests_without_legacy_name_collision():
    from oeis_learn.pipeline_validation.progressive_cli import CPU_TESTS

    assert "tests/contract/test_progressive_harness_contract.py" not in CPU_TESTS
    assert "tests/unit/test_progressive_readiness.py" not in CPU_TESTS
    assert "tests/unit/test_progressive_coverage.py" in CPU_TESTS
    assert len(CPU_TESTS) == len(set(CPU_TESTS))
    root = Path(__file__).resolve().parents[2]
    assert all((root / path).is_file() for path in CPU_TESTS)
