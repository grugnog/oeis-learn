"""Contract tests verifying CLI argument parsing against cli-interface.contract.json."""

import pytest
from oeis_learn.cli.main import build_parser, cli


def test_cli_parser_ingest_arguments():
    parser = build_parser()
    args = parser.parse_args(["ingest", "--db-path", "data/test.duckdb", "--subset-stage", "1"])
    assert args.command == "ingest"
    assert args.db_path == "data/test.duckdb"
    assert args.subset_stage == 1


def test_cli_parser_train_arguments():
    parser = build_parser()
    args = parser.parse_args([
        "train",
        "--config", "configs/train_tier1.yaml",
        "--tier", "1",
        "--curriculum-stage", "2",
        "--beta-sft", "0.25",
        "--beta-kl", "0.06",
        "--alpha-ent", "0.02",
        "--enable-pbrs",
        "--enable-lexicase",
        "--num-cpu-threads", "8",
    ])
    assert args.command == "train"
    assert args.tier == 1
    assert args.curriculum_stage == 2
    assert args.beta_sft == 0.25
    assert args.beta_kl == 0.06
    assert args.alpha_ent == 0.02
    assert args.enable_pbrs is True
    assert args.enable_lexicase is True
    assert args.num_cpu_threads == 8


def test_cli_parser_synthesize_arguments():
    parser = build_parser()
    args = parser.parse_args([
        "synthesize",
        "--oeis-id", "A000045",
        "--checkpoint", "ckpt.pt",
        "--fuel-budget", "5000",
        "--extrapolate", "50",
        "--mdl-max", "1.15",
    ])
    assert args.command == "synthesize"
    assert args.oeis_id == "A000045"
    assert args.checkpoint == "ckpt.pt"
    assert args.fuel_budget == 5000
    assert args.extrapolate == 50
    assert args.mdl_max == 1.15


def test_cli_parser_discover_arguments():
    parser = build_parser()
    args = parser.parse_args(["discover", "--checkpoint", "vicreg.pt", "--max-candidates", "50", "--precision-digits", "100", "--output-proofs", "reports/proofs.md"])
    assert args.command == "discover"
    assert args.checkpoint == "vicreg.pt"
    assert args.max_candidates == 50
    assert args.precision_digits == 100
    assert args.output_proofs == "reports/proofs.md"


def test_cli_parser_phase2_arguments():
    parser = build_parser()

    # generate-sft
    args = parser.parse_args(["generate-sft", "--output-path", "data/sft.json", "--num-samples", "100", "--seed", "42"])
    assert args.command == "generate-sft"
    assert args.output_path == "data/sft.json"
    assert args.num_samples == 100
    assert args.seed == 42

    # warmup-sft
    args = parser.parse_args(["warmup-sft", "--dataset-path", "data/sft.json", "--output-checkpoint", "checkpoints/sft.pt", "--epochs", "3", "--lr", "0.001"])
    assert args.command == "warmup-sft"
    assert args.dataset_path == "data/sft.json"
    assert args.output_checkpoint == "checkpoints/sft.pt"
    assert args.epochs == 3
    assert args.lr == 0.001

    # test-progressive
    args = parser.parse_args(["test-progressive", "--max-tier", "2", "--output-report", "reports/report.json"])
    assert args.command == "test-progressive"
    assert args.max_tier == 2
    assert args.output_report == "reports/report.json"


def test_cli_execution_commands_run_without_error(tmp_path):
    import torch
    from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
    from oeis_learn.evaluation.checkpoint import save_checkpoint_v2

    enc_cfg = {"d_model": 64, "n_heads": 2, "n_encoder_layers": 2, "d_ff": 128, "dropout": 0.0, "primes": [3, 5], "max_valuation": 16, "use_film": True}
    dec_cfg = {"d_model": 64, "n_heads": 2, "n_decoder_layers": 2, "d_ff": 128, "dropout": 0.0}
    enc = TriStreamEncoder(**enc_cfg)
    dec = WatTransformerDecoder(**dec_cfg)
    ckpt_file = str(tmp_path / "test_model.v2.pt")
    save_checkpoint_v2(ckpt_file, enc, dec, enc_cfg, dec_cfg, epoch=1, producer_version="test")

    # Test ingest
    db_file = str(tmp_path / "test.duckdb")
    ret = cli(["ingest", "--db-path", db_file])
    assert ret == 0

    # Test synthesize
    ret = cli(["synthesize", "--oeis-id", "A000217", "--checkpoint", ckpt_file, "--candidate-budget", "1"])
    assert ret in (0, 1)

    # Test discover
    proof_file = str(tmp_path / "test_proofs.md")
    ret = cli(["discover", "--checkpoint", ckpt_file, "--output-proofs", proof_file, "--precision-digits", "50"])
    assert ret == 0
