"""Retirement tests inspect real sources/distributions and injected regressions."""

from pathlib import Path
import os
import pytest

from oeis_learn.synthesis.retirement import audit

ROOT = Path(__file__).resolve().parents[2]


def test_source_tree_retirement():
    result = audit(ROOT / "src/oeis_learn", project_root=ROOT)
    assert result["status"] == "PASS", result["violations"]


@pytest.mark.parametrize(
    "body",
    [
        "import z3",
        "from z3 import Solver",
        "import importlib\nimportlib.import_module('oeis_learn.decoder.grounding')",
        "__import__('oeis_learn.decoder.constant_solver')",
        "from importlib import import_module as loader\nloader('oeis_learn.decoder.grounding')",
        "def solve_dixon_bounded(): pass",
        "def grounded_variants(): pass",
    ],
)
def test_renamed_or_dynamic_python_solver_is_rejected(tmp_path, body):
    (tmp_path / "innocent_name.py").write_text(body)
    assert audit(tmp_path)["status"] == "FAIL"


def test_installed_distribution_has_no_retired_surface():
    if not any(
        os.environ.get(k) == "1"
        for k in ("OEIS_REQUIRE_RETIREMENT_IMAGE", "OEIS_REQUIRE_BIGINT_BUILD")
    ):
        pytest.skip("requires clean installed qualification image outside checkout")
    import subprocess, sys, json

    result = subprocess.run(
        [sys.executable, "-I", "-m", "oeis_learn.synthesis.retirement", "--installed"],
        capture_output=True,
        text=True,
        cwd="/tmp",
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout)["status"] == "PASS"
