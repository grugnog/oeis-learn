import importlib.util
import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from oeis_learn.decoder.program_codec import (
    FOUNDATION_VOCAB_SIZE,
    PAD_ID,
    codec_digest,
    encode_body,
)
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.encoder.integer_bytes import IntegerByteEncoder
from oeis_learn.experiments.profiles import qualified_model_profile
from oeis_learn.fresh import worker
from oeis_learn.fresh.bundle import pack
from oeis_learn.fresh.common import digest, file_hash, read, write
from oeis_learn.tracking.training_checkpoint import QualifiedCheckpointStore

spec = importlib.util.spec_from_file_location(
    "fresh_host", Path(__file__).parents[2] / "scripts/foundation_host.py"
)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


def small_models(profile, *, device):
    e = IntegerByteEncoder(d_model=16, n_heads=2, n_encoder_layers=1, d_ff=32, dropout=0)
    d = WatTransformerDecoder(
        vocab_size=FOUNDATION_VOCAB_SIZE,
        d_model=16,
        n_heads=2,
        n_decoder_layers=1,
        d_ff=32,
        dropout=0,
        pad_idx=PAD_ID,
        codec_sha256=codec_digest(),
        attention_backend="eager",
    )
    e.to(device)
    d.to(device)
    return e, d, {"total": sum(p.numel() for m in (e, d) for p in m.parameters())}


def settings(examples, cap):
    return {
        "device": "cpu",
        "model": qualified_model_profile(attention_backend="eager"),
        "seed": 7,
        "contract_sha256": digest("fixture"),
        "examples_sha256": digest(examples),
        "learning_rate": 0.0003,
        "batch_size": 4,
        "free_disk_floor_bytes": 0,
        "seconds": 30,
        "checkpoint_reserve_seconds": 2,
        "checkpoint_every_seconds": 60,
        "update_cap": cap,
        "charged_budget_ns": 0,
        "ledger_sequence": 0,
    }


def examples():
    return {
        "training": [
            {
                "program_id": digest(n),
                "values": [str(n)] * 20,
                "tokens": encode_body(f"i256.const {n}"),
            }
            for n in range(8)
        ],
        "development": [
            {
                "program_id": digest("dev"),
                "values": ["9"] * 20,
                "tokens": encode_body("i256.const 9"),
            }
        ],
    }


def advance_test_ledger(cfg, checkpoints):
    """Model the durable host ledger moving past every retained checkpoint.

    This is a test fixture, not production budget issuance. Each worker adds its
    elapsed time to the starting charge, so a constant charge fails on a second
    resume. The host instead records actual elapsed time between attempts.
    """
    bases = [read(p) for p in Path(checkpoints).glob("qualified-*-base.json")]
    if not bases:
        raise ValueError("resume fixture requires checkpoint budget evidence")
    cfg["charged_budget_ns"] = max(b["charged_budget_ns"] for b in bases) + 10**9
    cfg["ledger_sequence"] = max(b["ledger_sequence"] for b in bases) + 1


def run_worker(tmp, cap, previous=None):
    inputs, output = tmp / "input", tmp / "output"
    inputs.mkdir(parents=True)
    output.mkdir()
    data = examples()
    cfg = settings(data, cap)
    if previous:
        shutil.copytree(previous / "checkpoints", inputs / "checkpoints")
        advance_test_ledger(cfg, inputs / "checkpoints")
    write(inputs / "examples.json", data)
    write(inputs / "settings.json", cfg)
    result = worker.train(inputs, output)
    store = QualifiedCheckpointStore(output / "checkpoints", free_floor_bytes=0)
    _, _, state = store.load_learning(
        output / "checkpoints" / result["checkpoint"], **worker.identities(cfg)
    )
    return output, result, state


def test_actual_updates_and_complete_resume_match_uninterrupted(tmp_path):
    with patch("oeis_learn.experiments.profiles.construct_qualified_models", small_models):
        _uninterrupted, report, expected = run_worker(tmp_path / "full", 3)
        initial, _, _initial = run_worker(tmp_path / "initial", 0)
        paused, _, _paused = run_worker(tmp_path / "first", 1, initial)
        _resumed, result, actual = run_worker(tmp_path / "second", 3, paused)
    assert report["completed_updates"] == result["completed_updates"] == 3
    assert result["recovery"]["completed"] == 1
    for name in ("encoder", "decoder"):
        for key in expected[name]:
            torch.testing.assert_close(actual[name][key], expected[name][key], rtol=0, atol=0)
    assert actual["sampler"]["cursor"] == expected["sampler"]["cursor"]
    assert actual["sampler"]["epoch"] == expected["sampler"]["epoch"]
    assert actual["sampler"]["next_sample_ids"] == expected["sampler"]["next_sample_ids"]
    assert actual["scaler"] is actual["scheduler"] is None
    assert not result["qualified"]


def test_initial_checkpoint_generation_and_profile_drift_rejection(tmp_path):
    with patch("oeis_learn.experiments.profiles.construct_qualified_models", small_models):
        output, report, state = run_worker(tmp_path / "initial", 0)
        assert report["completed_updates"] == 0 and state["optimizer"]["state"] == {}
        inputs = tmp_path / "generation"
        inputs.mkdir()
        cfg = dict(
            settings(examples(), 0),
            checkpoint=report["checkpoint"],
            attempts=2,
            generation_tokens=8,
            per_group_seconds=1,
            protocol_sha256=digest("protocol"),
        )
        write(inputs / "settings.json", cfg)
        write(inputs / "visible.json", [{"group_id": digest("group"), "values": ["0"] * 20}])
        shutil.copytree(output / "checkpoints", inputs / "checkpoints")
        generated = tmp_path / "generated"
        generated.mkdir()
        seal = worker.generate(inputs, generated)
        assert (
            len(seal["records"]) == 1 and seal["checkpoint_sha256"] == report["checkpoint_sha256"]
        )
        cfg["contract_sha256"] = digest("foreign")
        write(inputs / "settings.json", cfg)
        with pytest.raises(ValueError, match="identity"):
            worker.generate(inputs, generated)


def test_corrupt_newest_checkpoint_recovers_and_never_reuses_generation(tmp_path):
    with patch("oeis_learn.experiments.profiles.construct_qualified_models", small_models):
        output, report, _state = run_worker(tmp_path / "initial", 1)
        corrupt = output / "checkpoints" / Path(report["checkpoint"]).with_suffix(".pt")
        corrupt.write_bytes(b"broken blob")
        resumed, result, _actual = run_worker(tmp_path / "resumed", 2, output)
    assert result["recovery"]["completed"] == 0
    assert result["completed_updates"] == 2
    assert int(Path(result["checkpoint"]).stem.split("-")[1]) >= 2
    assert list((resumed / "checkpoints/quarantine").rglob("*.pt"))


class FakeClock:
    def __init__(self):
        self.n = 0
        self.boot = "test"

    def __call__(self):
        return {"boot_id": self.boot, "monotonic_ns": self.n, "utc_ns": self.n + 10**12}


def test_durable_budget_pause_crash_recovery_and_no_refund(tmp_path):
    clock = FakeClock()
    ledger = host.Ledger(tmp_path, "run", 100, now=clock)
    ledger.record("training")
    clock.n += 10**9
    ledger.record("paused")
    clock.n += 50 * 10**9
    recovered = host.Ledger(tmp_path, "run", 100, now=clock)
    assert recovered.remaining() == 99
    recovered.record("training")
    clock.n += 20 * 10**9
    crashed = host.Ledger(tmp_path, "run", 100, now=clock)
    assert crashed.remaining() == 79
    with pytest.raises(ValueError, match="identity/cap"):
        host.Ledger(tmp_path, "run", 200, now=clock)
    clock.n -= 10**9
    with pytest.raises(ValueError, match="backward"):
        crashed.record("paused")


def test_private_commands_worker_has_no_devices_or_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(
        host.Engine, "device_args", lambda self, node: ["--device", "/dev/kfd", "--device", node]
    )
    policy = {
        "engine": "podman",
        "engine_executable": "podman",
        "project": "fixture",
        "render_node": "/dev/dri/renderD128",
    }
    coordinator = host.Coordinator(tmp_path / "controller", policy, "sha256:" + "a" * 64, None)
    assert coordinator.root.is_dir()
    coordinator.close()
    for role in ("learner", "generation", "evaluator", "worker"):
        (tmp_path / "in").mkdir(exist_ok=True)
        (tmp_path / "out").mkdir(exist_ok=True)
        argv = host.container_command(
            policy,
            "sha256:" + "a" * 64,
            "role",
            role,
            tmp_path / "in",
            tmp_path / "out",
            ["python", "-m", "fixture"],
        )
        assert not any("target=/workspace" in a for a in argv)
        assert not any("docker.sock" in a for a in argv)
        assert "--network=none" in argv and "--read-only" in argv
        assert any(a.endswith("dst=/work/input,readonly") for a in argv)
        assert any(a.startswith("--device") for a in argv) == (role in {"learner", "generation"})
        assert "--memory=" + str((1 if role == "worker" else 16) << 30) in argv


def test_snapshot_hash_and_symlink_rejection(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "file"
    path.write_text("data")
    ref = {"bytes": 4, "sha256": file_hash(path)}
    host.snapshot_file(root, "file", tmp_path / "copy", ref)
    (root / "link").symlink_to(path)
    with pytest.raises(OSError):
        host.snapshot_file(root, "link", tmp_path / "bad", ref)
    with pytest.raises(ValueError):
        host.snapshot_file(root, "../outside", tmp_path / "bad", ref)
    path.write_text("edit")
    with pytest.raises(ValueError, match="changed"):
        host.snapshot_file(root, "file", tmp_path / "changed", ref)


def test_bundle_on_failure_excludes_truth_inputs_and_weights(tmp_path):
    root = tmp_path / "run"
    write(root / "run.json", {"identity": digest("run")})
    write(root / "status.json", {"status": "interrupted", "error": "example"})
    write(root / "roles/train-1-input/truth.json", {"secret": "DO NOT EXPORT"})
    write(root / "truth.json", {"secret": "DO NOT EXPORT"})
    write(
        root / "roles/train-1/metrics.jsonl",
        b'{"loss":2,"nonpadding_tokens":8,"phase_timings_ns":{"update":1000000000}}\n',
    )
    write(root / "roles/train-1/state.pt", b"weights")
    output = tmp_path / "analysis.zip"
    report = pack(root, output)
    with zipfile.ZipFile(output) as archive:
        assert not any(
            "truth" in n or "-input" in n or n.endswith(".pt") for n in archive.namelist()
        )
        summary = json.loads(archive.read("analysis.json"))
        assert summary["tokens_per_training_phase_second"] == 8
        assert summary["status"]["status"] == "interrupted" and not summary["qualified"]
    assert report["sha256"] == file_hash(output)


def test_bundle_log_and_metrics_sampling_is_explicit(tmp_path):
    root = tmp_path / "run"
    write(root / "run.json", {"identity": digest("run")})
    write(root / "status.json", {"status": "paused"})
    write(root / "roles/train-1/stdout.log", b"x" * (2 << 20))
    output = tmp_path / "analysis.zip"
    pack(root, output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("bundle.json"))
        assert manifest["omissions"][0]["reason"] == "log tail only"
        assert len(archive.read("roles/train-1/stdout.log")) == 1 << 20


def test_publication_preserves_user_permissions_and_rejects_links(tmp_path):
    job, output = tmp_path / "private", tmp_path / "public"
    write(job / "status.json", {"status": "interrupted"})
    write(job / "roles/train-1/stdout.log", b"readable log")
    (output / "roles/train-1").mkdir(parents=True)
    (output / "roles/train-1").chmod(0o770)
    host.publish(job, output)
    assert (output / "roles/train-1").stat().st_mode & 0o777 == 0o770
    assert (output / "roles/train-1/stdout.log").stat().st_mode & 0o004
    linked = tmp_path / "linked"
    linked.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        host.publish(job, linked)


def test_resume_rejects_charge_behind_checkpoint(tmp_path):
    with patch("oeis_learn.experiments.profiles.construct_qualified_models", small_models):
        previous, report, _ = run_worker(tmp_path / "first", 1)
        inputs, output = tmp_path / "resume-input", tmp_path / "resume-output"
        inputs.mkdir()
        output.mkdir()
        shutil.copytree(previous / "checkpoints", inputs / "checkpoints")
        manifest = read(inputs / "checkpoints" / report["checkpoint"])
        base = read(inputs / "checkpoints" / manifest["base"])
        cfg = settings(examples(), 2)
        cfg["charged_budget_ns"] = base["charged_budget_ns"] - 1
        assert cfg["charged_budget_ns"] >= 0
        write(inputs / "settings.json", cfg)
        write(inputs / "examples.json", examples())
        with pytest.raises(ValueError, match="checkpoint budget exceeds durable host ledger"):
            worker.train(inputs, output)


def test_resume_fixture_advances_cumulative_charge_and_sequence(tmp_path):
    write(
        tmp_path / "qualified-000001-base.json", dict(charged_budget_ns=10**12, ledger_sequence=5)
    )
    write(
        tmp_path / "qualified-000002-base.json",
        dict(charged_budget_ns=10**12 + 3 * 10**9, ledger_sequence=6),
    )
    cfg = {}
    advance_test_ledger(cfg, tmp_path)
    assert cfg == dict(charged_budget_ns=10**12 + 4 * 10**9, ledger_sequence=7)
