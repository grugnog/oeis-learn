"""Scope receipts and one-shot orchestration; mock metadata never qualifies HIP."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oeis_learn.fresh.common import digest


def runner():
    path = Path(__file__).resolve().parents[2] / "scripts/run_bigint_diverse32.py"
    spec = importlib.util.spec_from_file_location("diverse32_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_fixture(tmp_path):
    from tests.unit.test_bigint_benchmark import benchmark
    from oeis_learn.fresh.bigint_config import defaults

    root = Path(__file__).resolve().parents[2]
    # Historical p95 values exercise the preregistration rationale, not current hardware.
    historical = json.loads(
        (
            root / "specs/008-bigint-constant-synthesis/validation/phase-5-batched-hip-review.json"
        ).read_text()
    )
    tiny = defaults()
    tiny["learning"]["validation_every"] = 100
    tiny["budgets"]["training_seconds"] = 14400
    screen = deepcopy(tiny)
    screen["learning"]["validation_every"] = 500
    screen["budgets"]["training_seconds"] = 7200
    shapes = dict(
        counts=dict(training=512, parameter_development=128, structural_development=128),
        tiny_count=32,
        native_admitted=True,
        stages={k: 1 for k in "ABCD"},
        max_input_bits=4096,
        max_tokens=1024,
        prepared_sha256="sha256:" + "1" * 64,
        split_sha256="sha256:" + "2" * 64,
    )
    report = dict(
        status="PASS",
        source_sha256=digest({"fixture": "source"}),
        image="sha256:" + "a" * 64,
        fitting=dict(complete=True),
        cases=[dict(case="mock")],
        costs=historical["costs"],
        neural=dict(
            hip="mock-unit-fixture",
            device="cuda:0",
            measurement_protocol="fixed-state-batched/v2",
            generation_parity=True,
            generation_batch_size=8,
            shape_sha256=digest(shapes),
            fixture_sha256="sha256:" + "f" * 64,
            frozen_states=[dict(parameters_sha256="sha256:" + "e" * 64)],
        ),
    )
    benchmark().qualify_scopes(
        report,
        {"diverse32": tiny, "transfer-screen": screen},
        shapes,
        tmp_path,
        primary="diverse32",
    )
    report["report_sha256"] = digest(report)
    return report, shapes


def test_smaller_receipt_can_pass_while_transfer_screen_remains_blocked(tmp_path):
    report, _ = report_fixture(tmp_path)
    assert report["preflight_eligible"]
    assert report["cost_receipts"]["diverse32"]["status"] == "PASS"
    assert report["cost_receipts"]["transfer-screen"]["status"] == "BLOCKED"
    assert (tmp_path / "measured-diverse32-recipe.yaml").exists()
    assert not (tmp_path / "measured-transfer-screen-recipe.yaml").exists()


@pytest.mark.parametrize("change", [None, "source", "image", "shape", "projection", "hash", "hip"])
def test_reviewed_receipt_is_revalidated_before_any_run(tmp_path, change):
    report, shapes = report_fixture(tmp_path)
    expected = dict(
        image=report["image"], source_sha256=report["source_sha256"], shape_sha256=digest(shapes)
    )
    if change in ("source", "image", "shape"):
        expected[{"source": "source_sha256", "shape": "shape_sha256", "image": "image"}[change]] = (
            "sha256:" + "b" * 64
        )
    if change == "projection":
        receipt = report["cost_receipts"]["diverse32"]
        receipt["projection"]["end_screen_prompts"] = 256
        receipt["receipt_sha256"] = digest(
            {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        )
    if change == "hip":
        report["neural"]["hip"] = None
    if change != "hash":
        report["report_sha256"] = digest({k: v for k, v in report.items() if k != "report_sha256"})
    else:
        report["report_sha256"] = "sha256:" + "0" * 64
    if change:
        with pytest.raises(ValueError):
            runner().reviewed_recipe(report, **expected)
    else:
        recipe, forecast = runner().reviewed_recipe(report, **expected)
        assert forecast["scope"] == "diverse32" and forecast["within_budget"]
        module = runner()
        module.claim_run(tmp_path, recipe, "first-run")
        with pytest.raises(ValueError, match="already claimed.*first-run"):
            module.claim_run(tmp_path, recipe, "fresh-budget-retry")


def test_diverse32_help_requires_only_standard_library():
    module = runner()
    subprocess.run(
        [sys.executable, "-S", str(Path(module.__file__)), "--help"],
        check=True,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize("interrupted", [False, True])
def test_runner_archives_negative_or_interrupted_attempt_without_remeasurement(
    tmp_path, monkeypatch, interrupted
):
    from types import SimpleNamespace
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.fresh.common import write

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import qualify_bigint_phase5 as qualification

    report, shapes = report_fixture(tmp_path)
    write(tmp_path / "reviewed.json", report)
    module = runner()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(source_snapshot, "collect_snapshot", lambda root: {"fixture": "source"})
    monkeypatch.setattr(qualification, "shape_manifest", lambda inputs: shapes)

    class Host:
        def __init__(self, args):
            self.engine = SimpleNamespace(name="podman")

        def ensure(self):
            return report["image"]

    monkeypatch.setitem(sys.modules, "foundation_local", SimpleNamespace(Host=Host, relative=str))
    launches = []

    def launch(argv, output, **kwargs):
        launches.append(argv[6])
        if argv[6] == "learnability":
            if interrupted:
                raise TimeoutError("injected interruption")
            run = tmp_path / "attempt/learnability-run"
            write(run / "summary.json", dict(memorization_gate=False))
            write(run / "checkpoint-identity.json", dict(selected="fixture-checkpoint"))
            write(run / "budget.json", dict(remaining=123, refunded=False))
        elif argv[6] == "pack":
            import zipfile

            with zipfile.ZipFile(tmp_path / "attempt/learnability.zip", "w"):
                pass

    monkeypatch.setattr(qualification, "launch", launch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diverse32",
            "--reviewed-preflight",
            "reviewed.json",
            "--inputs",
            "inputs",
            "--output",
            "attempt",
        ],
    )
    assert module.main() == (1 if interrupted else 2)
    result = json.loads((tmp_path / "attempt/summary.json").read_text())
    assert result["status"] == ("FAIL" if interrupted else "BLOCKED")
    assert result["qualified"] is False
    assert launches == (
        ["check", "learnability"] if interrupted else ["check", "learnability", "pack"]
    )
    assert (tmp_path / "attempt.zip").exists()
    assert len(list((tmp_path / "run-state/phase5-diverse32").glob("*.json"))) == 1
