"""Caller routing, historical scope and independent certificate serialization."""

from pathlib import Path
import json
from types import SimpleNamespace
import pytest
from oeis_learn.synthesis.fitting import fit_program
from oeis_learn.discovery import native_obligations as smt


def test_facade_sends_only_visible_values_and_actual_indices():
    calls = []

    class Client:
        def fit(self, source, prompt, **kwargs):
            calls.append((source, prompt, kwargs))
            return dict(outcome="unknown", admitted=[])

    values = list(range(20)) + [987654321] * 80
    result = fit_program("i256.const 0", values, indices=list(range(-7, 93)), client=Client())
    assert result.outcome == "unknown" and result.source == result.original_source
    assert calls[0][1].values == tuple(map(str, range(20)))
    assert calls[0][1].indices == tuple(map(str, range(-7, 13)))
    assert calls[0][2]["policy"] == "eligible-literals/v1"
    assert calls[0][2]["profile"] == "wat_i256_checked_v1"


def test_retired_settings_and_profiles_fail_without_solver():
    with pytest.raises(ValueError, match="eight-parameter"):
        fit_program("i256.const 0", [1], max_parameters=4)
    assert fit_program("i64.const_?", [1]).outcome == "unsupported"
    assert fit_program("i256.const 0", [1], profile="modular_i64").outcome == "unsupported"


def test_certificate_serializer_preserves_exact_arithmetic_and_boolean_sort():
    n = smt.Int("n")
    huge = 1 << 300
    condition = smt.Implies(smt.And(n >= 0, n <= 100), smt.And(n + huge > 0, n != (-huge)))
    assert str(huge) in condition.text
    assert condition.symbols == frozenset(["n"])
    assert "(not (= n (- " in condition.text
    assert smt.And().text == "true"
    with pytest.raises(TypeError):
        bool(condition)
    with pytest.raises(ValueError):
        smt.Int("n) (assert false)")


@pytest.mark.parametrize(
    "reply",
    [
        b"not json",
        b'{"result":"verified","solver":"fake"}',
        b'{"result":"unsat","solver":"fake","extra":true}',
    ],
)
def test_certificate_protocol_failures_cannot_discharge(monkeypatch, tmp_path, reply):
    binary = tmp_path / "solver"
    binary.write_text("fixture")
    monkeypatch.setenv("OEIS_CERTIFICATE_SOLVER", str(binary))
    monkeypatch.setattr(
        smt.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=reply)
    )
    assert smt.discharge(smt.Int("n") >= 0)["result"] == "unknown"


def test_missing_native_certificate_checker_never_proves(monkeypatch, tmp_path):
    monkeypatch.setenv("OEIS_CERTIFICATE_SOLVER", str(tmp_path / "missing"))
    assert smt.discharge(smt.Int("n") >= 0)["result"] == "unknown"


def test_native_obligation_request_is_bounded_and_has_no_assignment_interface(
    monkeypatch, tmp_path
):
    binary = tmp_path / "solver"
    binary.write_text("fixture")
    monkeypatch.setenv("OEIS_CERTIFICATE_SOLVER", str(binary))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b'{"result":"unsat","solver":"native"}')

    monkeypatch.setattr(smt.subprocess, "run", run)
    assert smt.discharge(smt.Int("n") >= 0)["result"] == "unsat"
    payload = json.loads(calls[0][1]["input"])
    assert payload == dict(declarations=["n"], assertion="(not (>= n 0))")
    assert calls[0][1]["timeout"] == 2.5 and calls[0][0] == [str(binary)]


def test_no_retired_imports_in_current_scripts_or_tests():
    import ast
    from oeis_learn.synthesis.retirement import RETIRED_MODULES

    root = Path(__file__).resolve().parents[2]
    for base in ("scripts", "tests"):
        for p in (root / base).rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert node.module not in {
                        "oeis_learn.decoder." + n for n in RETIRED_MODULES
                    }, p
                    if node.module == "oeis_learn.decoder":
                        assert not set(a.name for a in node.names) & set(RETIRED_MODULES), p


@pytest.mark.parametrize("case", ["missing", "invalid_terms", "retired"])
def test_cli_reports_input_and_retired_fitting_failures(tmp_path, capsys, case):
    from oeis_learn.cli.main import cli

    source = tmp_path / "source.wat"
    if case != "missing":
        source.write_text("i64.const_?")
    terms = "oops" if case == "invalid_terms" else "4,4"
    assert cli(["solve-constants", "--wat-file", str(source), "--terms", terms]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["outcome"] == ("unsupported" if case == "retired" else "invalid_or_unavailable")


@pytest.mark.parametrize(
    "settings",
    [
        {"enable_smt": True},
        {"enable_z3_qfnia": True},
        {"tier": "three_tier_decoupled"},
        {"enable_dixon_diophantine": True},
        {"coefficient_bound": 1000},
    ],
)
def test_retired_solver_settings_are_rejected(settings):
    from oeis_learn.synthesis.fitting import require_native_solver_settings

    with pytest.raises(ValueError, match="retired solver settings"):
        require_native_solver_settings({"solver": settings})


def test_old_training_recipe_fails_before_constructing_models(tmp_path, capsys):
    from oeis_learn.cli.main import cli

    config = tmp_path / "old.yaml"
    config.write_text("solver: {enable_smt: true}")
    assert cli(["train", "--config", str(config)]) == 2
    assert "retired solver settings" in capsys.readouterr().out
