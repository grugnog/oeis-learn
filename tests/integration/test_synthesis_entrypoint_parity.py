"""Integration tests for synthesis CLI and direct service semantic parity."""

from __future__ import annotations

import json
import os
import tempfile
import pytest
import torch
from oeis_learn.cli.main import cli
from oeis_learn.data.benchmark import load_benchmark_manifest
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.evaluation.checkpoint import save_checkpoint_v2
from oeis_learn.evaluation.protocol import EvaluationProtocol
from oeis_learn.evaluation.synthesis import evaluate_cohort_synthesis


@pytest.fixture(scope="module")
def checkpoint_and_manifest():
    enc_cfg = {
        "d_model": 64,
        "n_heads": 2,
        "n_encoder_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
        "primes": [3, 5, 7, 11],
        "max_valuation": 16,
        "use_film": True,
    }
    dec_cfg = {
        "d_model": 64,
        "n_heads": 2,
        "n_decoder_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
    }
    torch.manual_seed(42)
    encoder = TriStreamEncoder(**enc_cfg)
    decoder = WatTransformerDecoder(**dec_cfg)

    with tempfile.NamedTemporaryFile(suffix=".v2.pt", delete=False) as f:
        ckpt_path = f.name

    prov = save_checkpoint_v2(
        checkpoint_path=ckpt_path,
        encoder=encoder,
        decoder=decoder,
        encoder_config=enc_cfg,
        decoder_config=dec_cfg,
        epoch=1,
        producer_version="test-v1",
    )

    manifest_path = "data/benchmarks/trustworthy_synthesis_v1.json"
    yield ckpt_path, manifest_path, encoder, decoder, prov

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def test_cli_and_service_parity(checkpoint_and_manifest):
    ckpt_path, manifest_path, encoder, decoder, prov = checkpoint_and_manifest
    manifest = load_benchmark_manifest(manifest_path)
    target = next(t for t in manifest.targets if t.oeis_id == "A000290")

    seed = 12345
    budget = 1

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_out:
        out_json = f_out.name

    try:
        # 1. Run through CLI
        exit_code = cli([
            "synthesize",
            "--checkpoint", ckpt_path,
            "--benchmark-manifest", manifest_path,
            "--oeis-id", "A000290",
            "--candidate-budget", str(budget),
            "--seed", str(seed),
            "--output-json", out_json,
        ])
        assert exit_code in (0, 1)

        with open(out_json, "r") as f:
            cli_res = json.load(f)

        # 2. Run directly through service
        protocol = EvaluationProtocol.from_dict({
            "schema_version": "1.0",
            "checkpoint_sha256": prov.checkpoint_sha256,
            "benchmark_manifest_sha256": manifest.manifest_sha256,
            "observed_horizon": 20,
            "unseen_horizon": 100,
            "candidate_budget": budget,
            "base_seed": seed,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 128,
            "constant_resolution": True,
            "solver_timeout_ms": 250,
            "max_placeholders": 4,
            "fuel_per_invocation": 10000,
            "memory_limit_mib": 16,
            "mdl_ratio_max": 1.2,
            "native_evaluator_required": True,
            "code_revision": "trustworthy-v1",
            "environment_fingerprint": "sha256:" + "0" * 64,
        })

        service_res = evaluate_cohort_synthesis(
            encoder=encoder,
            decoder=decoder,
            checkpoint=prov,
            target=target,
            protocol=protocol,
        ).to_dict()

        # Both candidate WAT programs must match exactly
        assert len(cli_res["candidates"]) == len(service_res["candidates"])
        for c_cli, c_srv in zip(cli_res["candidates"], service_res["candidates"]):
            assert c_cli["raw_wat"] == c_srv["raw_wat"]
            assert c_cli["classification"] == c_srv["classification"]
            assert c_cli["primary_failure_stage"] == c_srv["primary_failure_stage"]
    finally:
        if os.path.exists(out_json):
            os.remove(out_json)


def test_cli_rejects_missing_target(checkpoint_and_manifest):
    ckpt_path, manifest_path, _, _, _ = checkpoint_and_manifest
    exit_code = cli([
        "synthesize",
        "--checkpoint", ckpt_path,
        "--benchmark-manifest", manifest_path,
        "--oeis-id", "A999999",  # Nonexistent
        "--seed", "42",
    ])
    assert exit_code == 2


def test_cli_rejects_corrupted_checkpoint(checkpoint_and_manifest):
    _, manifest_path, _, _, _ = checkpoint_and_manifest
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(b"not a valid checkpoint")
        corrupt_path = f.name

    try:
        exit_code = cli([
            "synthesize",
            "--checkpoint", corrupt_path,
            "--benchmark-manifest", manifest_path,
            "--oeis-id", "A000290",
            "--seed", "42",
        ])
        assert exit_code == 2
    finally:
        if os.path.exists(corrupt_path):
            os.remove(corrupt_path)
