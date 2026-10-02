"""Full gate is a CLI run; these tests prevent weakened corpus/CLI contracts."""

from importlib.resources import files
from pathlib import Path
import pytest
from oeis_learn.cli.foundation import cmd_conformance
from oeis_learn.cli.main import build_parser
from oeis_learn.sandbox.conformance_runner import arithmetic_vectors, structured_vectors
from oeis_learn.sandbox.pipeline import Runtime


def test_packaged_regressions_match_reviewable_fixtures():
    fixtures = Path(__file__).parents[1] / "fixtures/foundation/conformance"
    for name in ("known_regressions.json", "manifest.json"):
        assert (
            files("oeis_learn.sandbox").joinpath("conformance", name).read_bytes()
            == (fixtures / name).read_bytes()
        )


def test_seeded_corpus_counts_bounds_and_determinism():
    first = list(arithmetic_vectors())
    assert len(first) == 10000 and first == list(arithmetic_vectors())
    assert {case[4] for case in first} == {None, "numeric_limit"}
    structured = list(structured_vectors())
    assert len(structured) == 256 and structured == list(structured_vectors())
    assert all(case[2] == list(range(100)) and len(case[3]) == 100 for case in structured)
    assert len({case[1] for case in structured}) == 256
    runtime = Runtime()
    for _, source, indices, expected, outcome in first[:30] + structured[:3]:
        actual = runtime.evaluate(source, indices)
        assert actual.outputs == actual.reference_outputs == expected and actual.outcome == outcome


def test_conformance_cli_requires_complete_inputs_and_keeps_diagnostics_separate(tmp_path, capsys):
    args = build_parser().parse_args(
        [
            "foundation",
            "conformance",
            "--profile",
            "profile.yaml",
            "--output",
            str(tmp_path),
            "--json",
        ]
    )
    assert args.profile == "profile.yaml" and args.as_json
    assert cmd_conformance(profile="missing", as_json=True) == 2
    assert cmd_conformance(profile="missing", output=str(tmp_path), artifacts=["x"]) == 2
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize(
    "status,interruptions,code", [("PASS", [], 0), ("FAIL", [], 4), ("FAIL", ["timeout"], 5)]
)
def test_gate_exit_codes(monkeypatch, tmp_path, status, interruptions, code):
    import oeis_learn.sandbox.conformance_runner as conformance

    monkeypatch.setattr(
        conformance,
        "run_conformance",
        lambda *args: {"status": status, "interruptions": interruptions},
    )
    assert cmd_conformance(profile="fixture", output=str(tmp_path)) == code
