"""Measurement fixtures must obey production widths and independent expectations."""

import importlib.util
from pathlib import Path

import pytest


def benchmark():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmark_bigint.py"
    spec = importlib.util.spec_from_file_location("bigint_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_width_matrix_uses_legal_inputs_and_independent_exact_expectations():
    from oeis_learn.synthesis.contracts import canonical_integer
    from oeis_learn.synthesis.reference import evaluate

    cases = benchmark().cases()
    assert {64, 256, 512, 1024, 4096} <= {c["supplied_bits"] for c in cases}
    assert max(c["result_bits"] for c in cases if c["expected_reason"] is None) == 16384
    assert any(c["expected_reason"] == "integer_limit" for c in cases)
    for case in cases:
        assert all(abs(canonical_integer(n)).bit_length() <= 4096 for n in case["indices"])
        result = evaluate(case["source"], case["indices"])
        if case["expected_reason"]:
            assert result.outcome != "ok" and result.reason == case["expected_reason"]
        else:
            assert result.outcome == "ok" and list(result.values) == case["expected"]
        assert case["old_source"] is None or case["supplied_bits"] <= 256


def test_measurement_summary_rejects_missing_failed_or_fabricated_samples():
    summarize = benchmark().summarize
    with pytest.raises(ValueError):
        summarize([])
    for sample in (float("nan"), -1, True):
        with pytest.raises(ValueError):
            summarize([sample])
    result = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["median_seconds"] == 3 and result["p95_seconds"] == 5


def test_cost_admission_rechecks_forecast_and_rejects_cpu_or_incomplete_dataset(tmp_path):
    from copy import deepcopy
    from oeis_learn.fresh.bigint_config import defaults, resolve

    module = benchmark()
    recipe = defaults()
    report = dict(
        status="PASS",
        source_sha256="sha256:" + "f" * 64,
        fitting=dict(complete=True),
        image="sha256:" + "a" * 64,
        neural=dict(
            hip="fixture",
            device="cuda:0",
            fixture_sha256="sha256:" + "b" * 64,
            measurement_protocol="fixed-state-batched/v2",
            generation_parity=True,
            generation_batch_size=8,
            frozen_states=[dict(parameters_sha256="sha256:" + "e" * 64)],
        ),
        costs={
            key + "_p95": 0.001
            for key in (
                "training",
                "teacher_forcing",
                "generation",
                "fit",
                "reference",
                "hidden_scoring",
                "checkpoint",
            )
        },
        cases=[dict(case="fixture")],
    )
    report["costs"].update(generation_batch_size=8, generation_batch_p95=0.008)
    shapes = dict(
        counts=dict(training=512, parameter_development=128, structural_development=128),
        tiny_count=32,
        max_input_bits=4096,
        max_tokens=1024,
        native_admitted=True,
        stages={s: 1 for s in "ABCD"},
        prepared_sha256="sha256:" + "c" * 64,
        split_sha256="sha256:" + "d" * 64,
    )
    from oeis_learn.synthesis.contracts import identity

    report["neural"]["shape_sha256"] = identity(shapes)
    for kind in (
        "cpu",
        "partial",
        "expensive",
        "failed",
        "not_admitted",
        "parity",
        "old_protocol",
        "unmeasured_batch",
    ):
        bad, data = deepcopy(report), deepcopy(shapes)
        if kind == "cpu":
            bad["neural"]["device"] = "cpu"
        if kind == "partial":
            data["counts"]["training"] = 511
        if kind == "expensive":
            bad["costs"]["generation_batch_p95"] = 480
        if kind == "failed":
            bad["status"] = "FAIL"
        if kind == "parity":
            bad["neural"]["generation_parity"] = False
        if kind == "old_protocol":
            bad["neural"]["measurement_protocol"] = "old"
        if kind == "unmeasured_batch":
            bad["costs"].pop("generation_batch_p95")
        if kind == "not_admitted":
            data["native_admitted"] = False
        with pytest.raises(ValueError):
            module.admit_cost_recipe(bad, recipe, data, tmp_path)
        assert not list(tmp_path.glob("measured-*-recipe.yaml"))
    result = module.admit_cost_recipe(report, recipe, shapes, tmp_path)
    import yaml

    value = resolve(yaml.safe_load((tmp_path / result["path"]).read_text()))
    assert value["learning"]["cost_preflight"]["scoring_data"] is False
    assert value["learning"]["cost_preflight"]["image"] == report["image"]


def test_neural_measurement_wiring_uses_full_checkpoint_and_legal_maximum_targets(
    tmp_path, monkeypatch
):
    """Small mocked tensors test plumbing, never qualify model throughput or HIP."""
    from types import SimpleNamespace
    import torch
    import oeis_learn.fresh.bigint_worker as worker
    import oeis_learn.decoder.sampler as sampler
    from oeis_learn.fresh.bigint_config import defaults, resolve

    class Encoder(torch.nn.Linear):
        def forward_from_sequences(self, sequences, device=None):
            return torch.zeros(len(sequences), 20, 2)

    encoder, decoder = Encoder(2, 2), torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        worker, "models", lambda recipe, device: (encoder, decoder, {"fixture_parameters": 12})
    )
    calls = []

    def loss(enc, dec, rows, device, optimizer=None):
        calls.append((len(rows), optimizer is not None))
        return 1.0

    monkeypatch.setattr(worker, "_batch_loss", loss)

    class Sampler:
        def __init__(self, *a, **kw):
            pass

        def sample_trajectories(self, *a, **kw):
            return [
                SimpleNamespace(tokens=(0, 1, 2), complete=True, source="fixture", failure=None)
                for _ in a[1]
            ]

    monkeypatch.setattr(sampler, "WatProgramSampler", Sampler)
    result = benchmark().neural_measurements(resolve(defaults()), tmp_path, "cpu", 2)
    assert result["device"] == "cpu" and result["scoring_data"] is False
    assert max(r["target_tokens"] for r in result["samples"]) == 1024
    assert max(r["input_bits"] for r in result["samples"]) == 4096
    assert max(r["difference_bits"] for r in result["samples"]) == 4097
    assert calls.count((8, False)) == 16
    assert result["measurement_protocol"] == "fixed-state-batched/v2"
    assert result["generation_parity"]
    assert [s["updates"] for s in result["frozen_states"]] == [12, 20]
    assert all(r["weights_unchanged"] for r in result["generation_samples"])
    assert not list(tmp_path.rglob("*.pt"))


def test_cost_fixture_matches_the_prepared_shape_separately_from_capacity_probes():
    rows = benchmark().neural_fixtures(dict(max_input_bits=75, max_tokens=777))
    matched = [r for r in rows if r["fixture_role"] == "preflight"]
    assert len(matched) == 1
    assert max(abs(int(v)).bit_length() for v in matched[0]["values"]) == 75
    assert 777 <= len(matched[0]["tokens"]) <= 1024
    assert max(len(r["tokens"]) for r in rows) == 1024


@pytest.mark.parametrize("wrong,reason", [(True, None), (False, "aggregate_fuel")])
def test_measurements_reject_wrong_answers_and_report_resource_exclusions(
    monkeypatch, wrong, reason
):
    from types import SimpleNamespace
    import json

    module = benchmark()
    case = dict(
        name="square",
        source="(program (return (mul (int 2) (int 2))))",
        indices=["0"],
        expected=["4"],
        expected_reason=None,
        supplied_bits=0,
        result_bits=3,
        old_source=None,
    )
    row = dict(
        outcome="ok" if wrong else "failure",
        values=["16"] if wrong else [],
        reason=reason,
        coverage=0,
    )
    reply = dict(
        version="bigint-measurements/v1",
        runtime_compile_seconds=0.1,
        samples=[
            dict(
                result=dict(row),
                timings=dict(prepare_seconds=0.1, execution_seconds=0.1, total_seconds=0.2),
            )
            for _ in range(2)
        ],
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stderr=b"", stdout=json.dumps(reply).encode()
        ),
    )
    if wrong:
        with pytest.raises(RuntimeError, match="disagreement"):
            module.runtime_measurements(case, 1, 2)
    else:
        result = module.runtime_measurements(case, 1, 2)
        assert (
            len(result["resource_exclusions"]) == 2
            and result["samples"][0]["result"]["outcome"] == "failure"
        )


def test_timing_repetition_count_cannot_change_calibration_weights(tmp_path, monkeypatch):
    """Exercise snapshot restoration with mutating mock training, not timing evidence."""
    from types import SimpleNamespace
    import torch
    import oeis_learn.fresh.bigint_worker as worker
    import oeis_learn.decoder.sampler as sampler
    from oeis_learn.fresh.bigint_config import defaults, resolve

    module = benchmark()
    monkeypatch.setattr(
        module,
        "neural_fixtures",
        lambda shapes: [
            dict(fixture_role="preflight", values=[str(i) for i in range(20)], tokens=[1, 2])
        ],
    )

    class Encoder(torch.nn.Linear):
        def forward_from_sequences(self, rows, device=None):
            return torch.zeros(len(rows), 20, 2)

    def models(recipe, device):
        torch.manual_seed(17)
        return Encoder(2, 2), torch.nn.Linear(2, 2), {}

    def loss(encoder, decoder, rows, device, optimizer=None):
        if optimizer is not None:
            with torch.no_grad():
                for model in (encoder, decoder):
                    for parameter in model.parameters():
                        parameter.add_(0.01)
        return 1.0

    class Sampler:
        def __init__(self, *a, **kw):
            pass

        def sample_trajectories(self, memory, seeds, **kw):
            return [
                SimpleNamespace(tokens=(0, 1, 2), source="fixture", complete=True, failure=None)
                for _ in seeds
            ]

    monkeypatch.setattr(worker, "models", models)
    monkeypatch.setattr(worker, "_batch_loss", loss)
    monkeypatch.setattr(sampler, "WatProgramSampler", Sampler)
    before = module.neural_measurements(resolve(defaults()), tmp_path / "two", "cpu", 2)
    after = module.neural_measurements(resolve(defaults()), tmp_path / "five", "cpu", 5)
    assert before["frozen_states"] == after["frozen_states"]
    hashes = [s["parameters_sha256"] for s in before["frozen_states"]]
    assert hashes[0] != hashes[1]
    assert all(
        row["weights_unchanged"]
        for row in before["generation_samples"] + after["generation_samples"]
    )
