import copy
import shutil
from unittest.mock import patch

import pytest
import torch

from oeis_learn.fresh import worker
from oeis_learn.fresh.bootstrap_config import learning_defaults
from oeis_learn.fresh.common import digest, read, write
from oeis_learn.fresh.diagnostics import evaluate
from oeis_learn.fresh.search import select, grounded_variants
from oeis_learn.decoder.program_codec import encode_body
from oeis_learn.sandbox.reference import execute_term
from oeis_learn.sandbox.wat_ast import parse
from tests.unit.test_fresh_workflow import (
    advance_test_ledger,
    examples,
    settings,
    small_models,
    host,
)


class ReferenceFixtureRunner:
    """Unit fixture only; production uses isolated native + reference workers."""

    def run_single(self, source, indices, **kwargs):
        from types import SimpleNamespace

        values = [execute_term(parse(source), i) for i in indices]
        failure = next((v.outcome for v in values if v.outcome), None)
        result = dict(
            outcome=failure,
            outputs=[str(v.value) for v in values],
            reference_outputs=[str(v.value) for v in values],
        )
        return SimpleNamespace(to_dict=lambda: result)


def test_prefix_selection_rejects_complete_wrong_program():
    candidates = [
        dict(source="i256.const 1", complete=True, prefix_fit=False),
        dict(source="i256.const 2", complete=True, prefix_fit=True),
    ]
    assert select(candidates, verified=True) == digest("i256.const 2")
    assert select(candidates[:1], verified=True) is None
    assert select(candidates, verified=False) == digest("i256.const 1")
    assert "i256.const 2" in grounded_variants("i256.const 1", [2] * 20)


def test_diagnostics_use_shuffled_prompts_and_retain_tokens(monkeypatch):
    encoder, decoder, _ = small_models(None, device="cpu")
    data = examples()
    observed = []

    def propose(self, visible, seed, ledger, greedy):
        observed.append(visible[0])
        source = f"i256.const {visible[0]}"
        return source, dict(
            method="greedy",
            tokens=encode_body(source),
            complete=True,
            failure=None,
            encoder_sha256=digest("e"),
            decoder_sha256=digest("d"),
        )

    monkeypatch.setattr("oeis_learn.evaluation.search.ModelProposer.stochastic", propose)
    learning = dict(learning_defaults(), diagnostic_rows=3, diagnostic_seconds=10)
    result = evaluate(encoder, decoder, data, learning, ReferenceFixtureRunner())
    training = result["splits"]["training"]
    assert training["denominator"] == training["exact"] == 3
    assert training["shuffled_exact"] == 0
    assert training["shuffled_denominator"] == 3
    assert all(row["tokens"] and row["emitted_eos"] for row in training["attempts"])
    assert result["cache_parity"]["passed"]
    assert len(observed) == 7


def test_full_validation_includes_rows_after_32():
    from oeis_learn.tracking.role_worker import validation_loss

    encoder, decoder, _ = small_models(None, device="cpu")
    rows = [
        {"tokens": encode_body(f"i256.const {n}"), "entry": {"conditioning_values": [str(n)] * 20}}
        for n in range(33)
    ]
    from oeis_learn.rl.sft_trainer import teacher_forced_logits

    with patch(
        "oeis_learn.rl.sft_trainer.teacher_forced_logits", wraps=teacher_forced_logits
    ) as call:
        validation_loss(encoder, decoder, rows, "cpu", limit=None)
        assert call.call_count == 33


def test_bootstrap_resume_preserves_stage_and_actual_weights(tmp_path, monkeypatch):
    data = examples()
    for row in data["training"] + data["development"]:
        row["stage"] = 0
    data["training"][-1]["stage"] = 1
    learning = dict(learning_defaults(), diagnostic_seconds=10, diagnostic_rows=2)
    # Deliberately do not promote; test real updates and durable restricted order.
    diagnostic = dict(
        stage=0,
        cache_parity={"passed": True},
        splits={s: dict(denominator=2, exact=0, attempts=[]) for s in data},
    )
    monkeypatch.setattr(
        "oeis_learn.fresh.diagnostics.evaluate", lambda *a, **kw: copy.deepcopy(diagnostic)
    )

    def run(root, cap, previous=None):
        inputs, output = root / "input", root / "output"
        inputs.mkdir(parents=True)
        output.mkdir()
        cfg = dict(
            settings(data, cap),
            learning=learning,
            cpu_threads=1,
            seconds=60,
            checkpoint_reserve_seconds=20,
        )
        if previous:
            shutil.copytree(previous / "checkpoints", inputs / "checkpoints")
            advance_test_ledger(cfg, inputs / "checkpoints")
        write(inputs / "settings.json", cfg)
        write(inputs / "examples.json", data)
        report = worker.train(inputs, output)
        from oeis_learn.tracking.training_checkpoint import QualifiedCheckpointStore

        _, _, state = QualifiedCheckpointStore(
            output / "checkpoints", free_floor_bytes=0
        ).load_learning(output / "checkpoints" / report["checkpoint"], **worker.identities(cfg))
        return output, report, state

    with patch("oeis_learn.experiments.profiles.construct_qualified_models", small_models):
        _, _, expected = run(tmp_path / "full", 3)
        first, _, _ = run(tmp_path / "first", 1)
        _, report, actual = run(tmp_path / "resume", 3, first)
    assert actual["sampler"]["stage"] == 0
    assert 7 not in actual["sampler"]["permutation"]
    assert report["validation_denominator"] == len(data["development"])
    for name in ("encoder", "decoder"):
        for key in expected[name]:
            torch.testing.assert_close(expected[name][key], actual[name][key], atol=0, rtol=0)


def test_synthetic_selection_precedes_loss(tmp_path):
    cfg = dict(
        profile="fresh-pilot-1h", protocol={"groups": 1}, learner={"learning": learning_defaults()}
    )
    write(
        tmp_path / "inputs/truth.json",
        {"final": [{"group_id": digest("hidden"), "values": ["9"] * 100}]},
    )
    for n, exact, loss in [(1, 4, 1.0), (2, 0, 0.01)]:
        stage = tmp_path / f"roles/train-{n:06d}"
        write(
            stage / "result.json",
            dict(
                checkpoint="q.json",
                completed_updates=n,
                validation_loss=loss,
                diagnostic=dict(stage=0, splits={"development": dict(exact=exact, denominator=8)}),
            ),
        )
        write(stage / "checkpoints/q.json", {"base": "base.json"})
        write(stage / "checkpoints/base.json", {"blob_sha256": digest(n)})
    result = host.finalize(tmp_path, cfg, "best-development")
    assert "train-000001" in result["checkpoint"]


def test_v2_configuration_is_explicit_and_v1_unchanged(tmp_path):
    import yaml
    from scripts.foundation_tasks import configuration

    old = configuration("configs/foundation/fresh_pilot.yaml")
    assert old["run"]["model"]["version"] == "qualified-model/v1"
    from pathlib import Path

    config = yaml.safe_load(Path("configs/foundation/bootstrap.yaml").read_text())
    for encoder in ("tri_stream", "hybrid", "integer_bytes"):
        config["run"]["encoder"] = encoder
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        assert configuration(path)["run"]["model"]["encoder"] == encoder
    del config["run"]["encoder"]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="explicit"):
        configuration(path)


@pytest.mark.parametrize("kind", ["integer_bytes", "tri_stream", "hybrid"])
def test_production_training_and_decoder_learn_generic_constant_microset(
    tmp_path, monkeypatch, kind
):
    # Reduced CPU shape, real optimizer/autoregressive grammar/EOS; no teacher stub.
    data = examples()
    data["development"] = [dict(data["training"][0]), dict(data["training"][1])]
    cfg = dict(
        settings(data, 300),
        cpu_threads=1,
        seconds=60,
        learning_rate=0.003,
        checkpoint_reserve_seconds=15,
        learning=dict(
            learning_defaults(),
            curriculum=False,
            diagnostic_rows=8,
            diagnostic_tokens=16,
            diagnostic_seconds=10,
        ),
    )
    inputs, output = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    write(inputs / "examples.json", data)
    write(inputs / "settings.json", cfg)
    monkeypatch.setattr(
        "oeis_learn.tracking.role_coordinator.QueueRunner", lambda _: ReferenceFixtureRunner()
    )

    def construct(profile, *, device):
        encoder, decoder, counts = small_models(profile, device=device)
        if kind != "integer_bytes":
            from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
            from oeis_learn.encoder.hybrid import HybridEncoder
            from oeis_learn.experiments.bootstrap_models import PRIMES

            cls = TriStreamEncoder if kind == "tri_stream" else HybridEncoder
            encoder = cls(
                d_model=16,
                n_heads=2,
                n_encoder_layers=1,
                d_ff=32,
                dropout=0.0,
                primes=PRIMES,
                base_moduli=PRIMES,
                max_valuation=32,
            ).to(device)
        return encoder, decoder, counts

    with patch("oeis_learn.experiments.profiles.construct_qualified_models", construct):
        report = worker.train(inputs, output)
    measured = report["diagnostic"]["splits"]["training"]
    assert report["completed_updates"] == 300
    assert measured["exact"] == 8
    assert measured["shuffled_exact"] == 0
    assert measured["completed"] == 8
    assert report["diagnostic"]["cache_parity"]["passed"]


@pytest.mark.parametrize("original_data", [False, True])
def test_host_learnability_freezes_encoder_override_and_does_not_score_final(
    tmp_path, original_data
):
    from tests.integration.test_fresh_lifecycle import run_environment, fixture_coordinator
    from tests.unit.test_fresh_workflow import FakeClock
    from oeis_learn.experiments.bootstrap_models import bootstrap_profile

    workspace, policy = run_environment(tmp_path)
    path = workspace / "inputs/prepared.json"
    manifest = read(path)
    if not original_data:
        manifest["config"]["run"].update(
            model=bootstrap_profile("tri_stream"), learning=learning_defaults()
        )
    manifest["identity"] = digest({k: v for k, v in manifest.items() if k != "identity"})
    write(path, manifest)
    clock = FakeClock()
    base = fixture_coordinator(clock)

    class Coordinator(base):
        def run(self, operation, role, inputs, output, seconds):
            assert operation == "train"  # No OEIS evaluation in the tiny gate.
            cfg = read(inputs / "settings.json")
            assert cfg["model"] == bootstrap_profile("hybrid")
            assert len(read(inputs / "examples.json")["training"]) == 32
            result = super().run(operation, role, inputs, output, seconds)
            result.update(
                validation_loss=1.0,
                diagnostic=dict(
                    stage=None,
                    cache_parity={"passed": True},
                    splits={
                            s: dict(denominator=32, exact=32, shuffled_exact=0, shuffled_denominator=32, coverage_complete=True, generation_parity={"passed": True})
                        for s in ("training", "development")
                    },
                ),
            )
            write(output / "result.json", result)
            return result

    with (
        patch.object(host, "clock", clock),
        patch.object(host, "process_identity", lambda pid: {"pid": pid}),
    ):
        host.execute(
            policy,
            "sha256:" + "a" * 64,
            ["run", "inputs", "runs/tiny", "bootstrap-learnability", "hybrid"],
            coordinator_factory=Coordinator,
        )
    status = read(workspace / "runs/tiny/status.json")
    assert status["status"] == "complete"
    assert status["final_score"]["learnability_passed"]
    assert not status["diagnostic_final_cohort_consumed"]
    assert not list((workspace / "runs/tiny/roles").glob("*-score"))


def test_checkpoint_diagnostic_verifies_old_binding_without_restoring_optimizer(
    tmp_path, monkeypatch
):
    from tests.unit.test_fresh_workflow import run_worker
    from oeis_learn.fresh.checkpoint_diagnostic import diagnose

    with patch("oeis_learn.experiments.profiles.construct_qualified_models", small_models):
        previous, result, _ = run_worker(tmp_path / "source", 1)
        inputs, output = tmp_path / "diagnostic-input", tmp_path / "diagnostic-output"
        inputs.mkdir()
        output.mkdir()
        shutil.copytree(previous / "checkpoints", inputs / "checkpoints")
        cfg = settings(examples(), 1)
        base = read(
            previous / "checkpoints" / (result["checkpoint"].removesuffix(".json") + "-base.json")
        )
        cfg.update(
            checkpoint=result["checkpoint"],
            source_contract=cfg["contract_sha256"],
            source_runtime=base["runtime_sha256"],
            learning=dict(
                learning_defaults(), diagnostic_seconds=10, diagnostic_tokens=8, diagnostic_rows=2
            ),
        )
        write(inputs / "settings.json", cfg)
        write(inputs / "examples.json", examples())
        monkeypatch.setattr(
            "oeis_learn.tracking.role_coordinator.QueueRunner", lambda _: ReferenceFixtureRunner()
        )
        report = diagnose(inputs, output)
        assert report["checkpoint_sha256"] == result["checkpoint_sha256"]
        assert report["validation_denominator"] == 1
        assert report["mode"].startswith("weights-only")
        assert not (output / "checkpoints").exists()
        cfg["source_contract"] = digest("wrong")
        write(inputs / "settings.json", cfg)
        with pytest.raises(ValueError):
            diagnose(inputs, output)


def test_interrupted_autoregressive_search_retains_emitted_prefix():
    from oeis_learn.evaluation.search import ModelProposer, SearchLedger, SearchExhausted
    from oeis_learn.decoder.program_codec import BOS_ID

    encoder, decoder, _ = small_models(None, device="cpu")
    ledger = SearchLedger(max_attempts=1, max_tokens=2, max_expansions=2, budget_ns=10**10)
    with pytest.raises(SearchExhausted) as failure:
        ModelProposer(encoder, decoder, context_limit=32).stochastic(
            [7] * 20, 7, ledger, greedy=True
        )
    assert failure.value.tokens[0] == BOS_ID
    assert len(failure.value.tokens) == 3


def test_full_validation_batching_preserves_per_program_mean():
    from oeis_learn.tracking.role_worker import validation_loss

    encoder, decoder, _ = small_models(None, device="cpu")
    rows = [
        {"tokens": encode_body(f"i256.const {n}"), "entry": {"conditioning_values": [str(n)] * 20}}
        for n in (0, -12, 98765)
    ]
    singleton = validation_loss(encoder, decoder, rows, "cpu", limit=None)
    batched = validation_loss(encoder, decoder, rows, "cpu", limit=None, batch_size=3)
    assert batched == pytest.approx(singleton, abs=1e-5)


def test_host_checkpoint_diagnostic_packages_separate_evidence(tmp_path, monkeypatch):
    from tests.unit.test_fresh_workflow import run_worker
    from oeis_learn.fresh.checkpoint_diagnostic import diagnose
    from oeis_learn.fresh.common import inventory
    from tests.unit.test_fresh_workflow import FakeClock

    monkeypatch.setattr(host, "clock", FakeClock())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_root = workspace / "run-state"
    source_job = state_root / digest("runs/source")[7:]
    source_job.mkdir(parents=True)
    data = examples()
    cfg = settings(data, 1)
    with patch("oeis_learn.experiments.profiles.construct_qualified_models", small_models):
        previous, result, _ = run_worker(tmp_path / "source-worker", 1)
        shutil.copytree(previous / "checkpoints", source_job / "roles/train/checkpoints")
        write(source_job / "run.json", dict(identity=digest("old-run"), settings={"learner": cfg}))
        write(
            source_job / "status.json",
            {"checkpoint": "roles/train/checkpoints/" + result["checkpoint"]},
        )
        write(source_job / "budget.json", {"charged_ns": 12345})
        write(source_job / "inputs/examples.json", data)
        before = inventory(source_job)

        class Coordinator:
            def __init__(self, *args):
                pass

            def run(self, operation, role, inputs, output, seconds):
                assert operation == "diagnose" and role == "learner" and seconds <= 3600
                assert not (inputs / "truth.json").exists()
                output.mkdir(parents=True)
                cfg = read(inputs / "settings.json")
                cfg["learning"].update(
                    diagnostic_seconds=10, diagnostic_tokens=8, diagnostic_rows=2
                )
                write(inputs / "settings.json", cfg)
                value = diagnose(inputs, output)
                write(output / "result.json", value)
                return value

            def close(self):
                pass

        monkeypatch.setattr(
            "oeis_learn.tracking.role_coordinator.QueueRunner", lambda _: ReferenceFixtureRunner()
        )
        policy = dict(workspace=str(workspace), state_root=str(state_root))
        host.execute(
            policy,
            "new-image",
            ["diagnose", "runs/source", "tmp-analysis/diagnosis"],
            coordinator_factory=Coordinator,
        )
        assert inventory(source_job) == before
        assert (workspace / "tmp-analysis/diagnosis.zip").is_file()
        assert (
            read(workspace / "tmp-analysis/diagnosis/source.json")["checkpoint_sha256"]
            == result["checkpoint_sha256"]
        )


def test_bootstrap_pilot_leaves_final_pending_until_explicit_evaluation(tmp_path):
    from tests.integration.test_fresh_lifecycle import run_environment, fixture_coordinator
    from tests.unit.test_fresh_workflow import FakeClock
    from oeis_learn.experiments.bootstrap_models import bootstrap_profile

    workspace, policy = run_environment(tmp_path)
    path = workspace / "inputs/prepared.json"
    manifest = read(path)
    manifest["config"]["run"].update(
        model=bootstrap_profile("tri_stream"),
        training_seconds=1800,
        learning=dict(learning_defaults(), max_updates=4, validation_every_updates=2),
    )
    manifest["identity"] = digest({k: v for k, v in manifest.items() if k != "identity"})
    write(path, manifest)
    clock = FakeClock()
    base = fixture_coordinator(clock)

    class Coordinator(base):
        def run(self, operation, role, inputs, output, seconds):
            result = super().run(operation, role, inputs, output, seconds)
            if operation == "train":
                result.update(
                    validation_loss=1.0,
                    diagnostic=dict(
                        stage=0,
                        cache_parity={"passed": True},
                        splits={
                            s: dict(denominator=8, exact=0, shuffled_exact=0)
                            for s in ("training", "development")
                        },
                    ),
                )
                write(output / "result.json", result)
            return result

    with (
        patch.object(host, "clock", clock),
        patch.object(host, "process_identity", lambda pid: {"pid": pid}),
    ):
        host.execute(
            policy,
            "image",
            ["run", "inputs", "runs/pilot", "fresh-pilot-1h"],
            coordinator_factory=Coordinator,
        )
        status = read(workspace / "runs/pilot/status.json")
        assert status["status"] == "complete" and status["completed_updates"] == 4
        assert status["final_score"]["status"] == "not_evaluated"
        assert not status["diagnostic_final_cohort_consumed"]
        assert not (workspace / "runs/pilot/roles/final-score").exists()
        host.execute(
            policy,
            "image",
            ["evaluate", "runs/pilot", "final", "latest"],
            coordinator_factory=Coordinator,
        )
        status = read(workspace / "runs/pilot/status.json")
        assert status["diagnostic_final_cohort_consumed"]
        assert status["final_score"]["denominator"] == 1
        assert len(status["final_score"]["solved_groups"]) == 1


def test_parameter_view_reserves_familiar_structures_without_training_leakage():
    from oeis_learn.fresh.bootstrap_views import parameter_view

    data = examples()
    rows = [
        dict(
            data["training"][0],
            program_id=f"{s}-{p}",
            structure=f"s{s}",
            values=[str(100 * s + p)] * 20,
            stage=s % 4,
        )
        for s in range(8)
        for p in range(8)
    ]
    source = dict(training=rows, development=data["development"])
    view = parameter_view(source)
    assert len(source["training"]) == 64  # Sealed source is unchanged.
    assert len(view["parameter_development"]) == 8
    assert len(view["training"]) == 56
    assert view["development"] == source["development"]
    assert (
        parameter_view(dict(source, training=list(reversed(rows))))["parameter_development"]
        == view["parameter_development"]
    )
    for held in view["parameter_development"]:
        assert all(r["values"] != held["values"] for r in view["training"])
        assert sum(r["structure"] == held["structure"] for r in view["training"]) >= 2
    assert parameter_view(data)["parameter_development"] == []


def test_full_development_diagnostic_reports_strata_and_actual_generation_parity(monkeypatch):
    encoder, decoder, _ = small_models(None, device="cpu")
    data = examples()
    data["development"] = [
        dict(
            data["training"][0],
            program_id=f"d{i}",
            values=[str(i)] * 20,
            tokens=encode_body(f"i256.const {i}"),
        )
        for i in range(40)
    ]
    calls = []

    def propose(self, visible, seed, ledger, greedy):
        calls.append(self.use_cache)
        source = f"i256.const {visible[0]}"
        return source, dict(
            method="greedy",
            tokens=encode_body(source),
            complete=True,
            failure=None,
            encoder_sha256=digest("e"),
            decoder_sha256=digest("d"),
            eos_trace=[dict(raw_probability=0.5, grammar_probability=1, allowed=True)],
        )

    monkeypatch.setattr("oeis_learn.evaluation.search.ModelProposer.stochastic", propose)
    result = evaluate(
        encoder,
        decoder,
        data,
        dict(learning_defaults(), diagnostic_rows=2, diagnostic_seconds=10),
        ReferenceFixtureRunner(),
        full_development=True,
        compare_generation=True,
    )
    dev = result["splits"]["development"]
    assert dev["denominator"] == dev["evaluated"] == dev["exact"] == 40
    assert dev["coverage_complete"]
    assert dev["prefix_exact"] == dev["continuation_exact"] == dev["legal"] == 40
    assert dev["by_length"]["1-64"]["denominator"] == 40
    assert dev["by_complexity"]["expression"]["exact"] == 40
    assert dev["output_diversity"] == dict(executed=40, distinct=40)
    assert False in calls and True in calls
    comparisons = [r["generation_parity"] for r in dev["attempts"] if "generation_parity" in r]
    assert len(comparisons) == 3 and all(c["passed"] for c in comparisons)
    assert all(r["eos_trace"] for r in dev["attempts"])


def test_production_sampler_records_eos_without_changing_greedy_tokens():
    from oeis_learn.evaluation.search import ModelProposer, SearchLedger

    encoder, decoder, _ = small_models(None, device="cpu")
    observed = []
    for cache in (True, False):
        proposer = ModelProposer(
            encoder, decoder, context_limit=16, use_cache=cache, diagnostics=True
        )
        _, evidence = proposer.stochastic(
            list(range(20)),
            9,
            SearchLedger(max_attempts=1, max_tokens=16, max_expansions=16, budget_ns=10**10),
            greedy=True,
        )
        assert evidence["eos_trace"]
        for step in evidence["eos_trace"]:
            assert 0 <= step["raw_probability"] <= 1
            assert 0 <= step["grammar_probability"] <= 1
            if not step["allowed"]:
                assert step["grammar_probability"] == 0
        observed.append(evidence["tokens"])
    assert observed[0] == observed[1]


def test_duplicate_development_prefixes_are_not_counted_as_shuffled_controls(monkeypatch):
    encoder, decoder, _ = small_models(None, device="cpu")
    data = examples()
    data["development"] = [dict(data["training"][0], program_id=f"d{i}") for i in range(3)]

    def propose(self, visible, seed, ledger, greedy):
        source = f"i256.const {visible[0]}"
        return source, dict(
            method="greedy",
            tokens=encode_body(source),
            complete=True,
            failure=None,
            encoder_sha256=digest("e"),
            decoder_sha256=digest("d"),
        )

    monkeypatch.setattr("oeis_learn.evaluation.search.ModelProposer.stochastic", propose)
    result = evaluate(
        encoder,
        decoder,
        data,
        dict(learning_defaults(), diagnostic_rows=3, diagnostic_seconds=10),
        ReferenceFixtureRunner(),
    )
    dev = result["splits"]["development"]
    assert dev["denominator"] == dev["exact"] == 3
    assert dev["shuffled_denominator"] == 0
    assert all(row["shuffle_unavailable"] for row in dev["attempts"])
