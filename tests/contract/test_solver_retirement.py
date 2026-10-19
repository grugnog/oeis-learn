"""Retirement tests inspect real sources/distributions and injected regressions."""

from pathlib import Path
import importlib.util
import os
import pytest

from oeis_learn.synthesis.retirement import audit, RETIRED_MODULES

ROOT = Path(__file__).resolve().parents[2]


def test_source_tree_retirement():
    result = audit(ROOT / "src/oeis_learn", project_root=ROOT)
    assert result["status"] == "PASS", result["violations"]


@pytest.mark.parametrize("body", [
    "import z3", "from z3 import Solver",
    "import importlib\nimportlib.import_module('oeis_learn.decoder.grounding')",
    "__import__('oeis_learn.decoder.constant_solver')",
    "def solve_dixon_bounded(): pass",
    "def grounded_variants(): pass",
])
def test_renamed_or_dynamic_python_solver_is_rejected(tmp_path, body):
    (tmp_path / "innocent_name.py").write_text(body)
    assert audit(tmp_path)["status"] == "FAIL"


def test_installed_distribution_has_no_retired_surface():
    if os.environ.get("OEIS_REQUIRE_RETIREMENT_IMAGE") != "1":
        pytest.skip("requires clean installed qualification image outside checkout")
    import oeis_learn
    root = Path(oeis_learn.__file__).resolve().parent
    assert "site-packages" in root.parts, root
    result = audit(root, installed=True)
    assert result["status"] == "PASS", result["violations"]
    for name in RETIRED_MODULES:
        assert importlib.util.find_spec("oeis_learn.decoder." + name) is None
    assert importlib.util.find_spec("z3") is None
