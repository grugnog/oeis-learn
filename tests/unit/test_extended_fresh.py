"""Measured batch selection, reproducible grouping and bounded long diagnostics."""

import json
from unittest.mock import patch

import pytest
import torch

from oeis_learn.fresh.common import digest, inventory, read, write
from oeis_learn.fresh.tuning import choose_trial, permutation
from tests.integration.test_fresh_lifecycle import fixture_coordinator, run_environment
from tests.unit.test_foundation_local import local
from tests.unit.test_fresh_cli import cli, config
from tests.unit.test_fresh_workflow import FakeClock, host, run_worker, small_models


def test_length_grouping_preserves_all_examples_rng_and_reduces_padding():
    rows = [{"tokens": [1] * (1 + n % 100)} for n in range(517)]
    g = torch.Generator().manual_seed(15)
    state = g.get_state()
    plain = permutation(rows, g, 16)
    g.set_state(state)
    grouped = permutation(rows, g, 16, True)
    assert sorted(plain) == sorted(grouped) == list(range(len(rows)))
    g.set_state(state)
    assert permutation(rows, g, 16, True) == grouped
    after = g.get_state()
    expected = permutation(rows, g, 16, True)
    g.set_state(after)
    assert permutation(rows, g, 16, True) == expected

    def padded(order):
        return sum(
            len(order[i : i + 16]) * max(len(rows[n]["tokens"]) for n in order[i : i + 16])
            for i in range(0, len(order), 16)
        )

    assert padded(grouped) < padded(plain) * 0.7


def test_measured_selection_prefers_small_near_optimal_safe_batch():
    def trial(size, rate, peak=2 << 30, steps=20, status="PASS"):
        return {
            "batch_size": size,
            "length_buckets": True,
            "tokens_per_second": rate,
            "peak_allocated_bytes": peak,
            "steps": steps,
            "status": status,
        }

    trials = [
        trial(4, 100),
        trial(8, 4800),
        trial(16, 5000),
        trial(32, 10000, peak=49 << 30),
        trial(64, 9000, status="OOM"),
        trial(4, float("nan")),
        trial(4, 10000, steps=2),
    ]
    assert choose_trial(trials)["batch_size"] == 8
    with pytest.raises(ValueError, match="no benchmark"):
        choose_trial([trial(4, 5000, status="OOM")])


def test_grouped_learning_state_resume_matches_uninterrupted(tmp_path):
    from tests.unit import test_fresh_workflow as fixture

    original = fixture.settings

    def grouped(data, cap):
        return dict(original(data, cap), length_buckets=True)

    with (
        patch.object(fixture, "settings", grouped),
        patch("oeis_learn.experiments.profiles.construct_qualified_models", small_models),
    ):
        _, _, expected = run_worker(tmp_path / "whole", 3)
        first, _, _ = run_worker(tmp_path / "first", 1)
        _, report, actual = run_worker(tmp_path / "second", 3, first)
    assert report["recovery"]["completed"] == 1
    for model in ("encoder", "decoder"):
        for key in expected[model]:
            torch.testing.assert_close(expected[model][key], actual[model][key], rtol=0, atol=0)
    assert expected["sampler"]["permutation"] == actual["sampler"]["permutation"]
    assert expected["sampler"]["cursor"] == actual["sampler"]["cursor"]


@pytest.mark.parametrize("invalid", [None, "duration", "batch", "bucket"])
def test_extended_config_stays_closed_and_bounded(tmp_path, invalid):
    cfg = config()
    cfg["run"].update(
        profile="fresh-extended-12h", training_seconds=36000, batch_size=16, length_buckets=True
    )
    if invalid == "duration":
        cfg["run"]["training_seconds"] = 36001
    if invalid == "batch":
        cfg["run"]["batch_size"] = 128
    if invalid == "bucket":
        cfg["run"]["length_buckets"] = "yes"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    if invalid:
        with pytest.raises(ValueError):
            cli.configuration(path)
    else:
        loaded = cli.configuration(path)
        assert loaded["run"]["training_seconds"] == 36000
        assert loaded["run"]["model"]["attention_backend"] == "eager"


def test_settings_from_completed_benchmark_need_no_sudo_or_host_policy(tmp_path):
    root = tmp_path / "runs/benchmark"
    cfg = config()
    cfg["run"]["model"] = {"remove": "constructor field"}
    write(root / "run.json", {"settings": {"profile": "fresh-benchmark"}})
    write(root / "status.json", {"status": "complete"})
    write(root / "input-manifest.json", {"config": cfg})
    write(
        root / "roles/benchmark/result.json",
        {"recommended": {"batch_size": 16, "length_buckets": True, "tokens_per_second": 20000}},
    )
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "load_policy", side_effect=AssertionError("sudo")),
    ):
        assert (
            local.main(
                ["extended-settings", "--benchmark", "runs/benchmark", "--output", "tmp-plan.json"]
            )
            == 0
        )
        assert (
            local.main(
                ["extended-settings", "--benchmark", "runs/benchmark", "--output", "tmp-plan.json"]
            )
            == 1
        )
    output = cli.configuration(tmp_path / "tmp-plan.json")
    assert output["archive"]["training_prompts"] == 2048
    assert output["archive"]["sampling_shards"] == 64
    assert output["run"]["evaluation"]["tokens"] == 1023
    assert output["run"]["batch_size"] == 16


def expanded_environment(tmp_path):
    workspace, policy = run_environment(tmp_path)
    policy["foundationTimeout"] = 44100
    root = workspace / "inputs"
    examples = read(root / "examples.json")
    examples["training"] = [
        {"program_id": digest(n), "values": [str(n)] * 20, "tokens": [1]} for n in range(2048)
    ]
    write(root / "examples.json", examples)
    manifest = read(root / "prepared.json")
    manifest["config"]["run"].update(
        profile="fresh-extended-12h", training_seconds=36000, batch_size=16, length_buckets=True
    )
    manifest["files"] = inventory(root, exclude=("prepared.json",))
    manifest["files"].pop("prepared.json", None)
    manifest["identity"] = digest({k: v for k, v in manifest.items() if k != "identity"})
    write(root / "prepared.json", manifest)
    return workspace, policy


def test_extended_stops_on_validation_decline_and_selects_best_validation(tmp_path):
    workspace, policy = expanded_environment(tmp_path)
    clock = FakeClock()
    base = fixture_coordinator(clock)

    class Validation(base):
        count = 0

        def run(self, operation, role, inputs, output, seconds):
            result = super().run(operation, role, inputs, output, seconds)
            if operation == "train" and read(inputs / "settings.json")["update_cap"]:
                self.count += 1
                result.update(
                    completed_updates=self.count * 100,
                    validation_loss=[1.0, 1.2, 1.4][self.count - 1],
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
            ["run", "inputs", "runs/long", "fresh-extended-12h"],
            coordinator_factory=Validation,
        )
    root = workspace / "runs/long"
    assert read(root / "status.json")["status"] == "complete"
    assert read(root / "status.json")["completed_updates"] == 300
    assert read(root / "status.json")["budget"]["cap_ns"] == 43200 * 10**9
    assert read(root / "early-stop.json")["best_validation_loss"] == 1.0
    assert "train-000001" in read(root / "finalization.json")["checkpoint"]


def test_benchmark_has_only_public_training_view_and_can_reuse_completed_result(tmp_path):
    workspace, policy = run_environment(tmp_path)
    clock = FakeClock()
    base = fixture_coordinator(clock)

    class Benchmark(base):
        def run(self, operation, role, inputs, output, seconds):
            assert operation == "benchmark" and role == "learner" and seconds <= 500
            assert set(inventory(inputs)) == {"settings.json", "examples.json"}
            result = {"recommended": {"batch_size": 16, "length_buckets": True}}
            write(output / "result.json", result)
            clock.n += 10**9
            return result

    with (
        patch.object(host, "clock", clock),
        patch.object(host, "process_identity", lambda pid: {"pid": pid}),
    ):
        assert (
            host.execute(
                policy,
                "image",
                ["run", "inputs", "runs/bench", "fresh-benchmark"],
                coordinator_factory=Benchmark,
            )
            == 0
        )
        assert (
            host.execute(
                policy,
                "image",
                ["run", "inputs", "runs/bench", "fresh-benchmark"],
                coordinator_factory=Benchmark,
            )
            == 0
        )
    assert read(workspace / "runs/bench/status.json")["status"] == "complete"


def test_extended_requires_deployed_watchdog_before_issuance(tmp_path):
    workspace, policy = expanded_environment(tmp_path)
    policy["foundationTimeout"] = 4200
    with pytest.raises(ValueError, match="deployed"):
        host.execute(policy, "image", ["run", "inputs", "runs/long", "fresh-extended-12h"])
    assert not (workspace / "runs/long/run.json").exists()


def test_benchmark_step_performs_real_supervised_update_and_tracks_padding():
    from oeis_learn.fresh.benchmark import training_step
    from tests.unit.test_fresh_workflow import examples

    encoder, decoder, _ = small_models({}, device="cpu")
    optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=0.0003)
    before = decoder.lm_head.weight.detach().clone()
    rows = examples()["training"]
    result, tokens, padded, order, cursor = training_step(
        encoder,
        decoder,
        optimizer,
        rows,
        list(range(8)),
        0,
        4,
        torch.Generator().manual_seed(7),
        False,
        device="cpu",
    )
    assert not torch.equal(before, decoder.lm_head.weight)
    assert result["loss"] > 0 and cursor == 4
    assert tokens == sum(len(row["tokens"]) for row in rows[:4]) and padded >= tokens


def test_benchmark_rejects_cpu_without_falling_back(tmp_path):
    from oeis_learn.fresh.benchmark import benchmark
    from tests.unit.test_fresh_workflow import examples, settings

    data = examples()
    write(tmp_path / "settings.json", settings(data, 0))
    write(tmp_path / "examples.json", data)
    with pytest.raises(ValueError, match="HIP benchmark"):
        benchmark(tmp_path, tmp_path / "output")
