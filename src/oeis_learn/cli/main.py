"""OEIS Learn Command-Line Interface (CLI)."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import yaml
from oeis_learn.cli.reporting import export_discovered_proofs_markdown
from oeis_learn.curriculum.extrapolation import ExtrapolationVerifier
from oeis_learn.curriculum.mdl_verifier import MdlVerifier
from oeis_learn.curriculum.sampler import DynamicMixtureSampler
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.data.dataset import OeisSequenceDataset
from oeis_learn.data.ingest import OeisIngestionPipeline
from oeis_learn.data.models import LatentDiscoveryCandidate, SequenceRecord
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.discovery.manifold import cluster_latent_manifold, reduce_manifold_2d
from oeis_learn.discovery.pslq_solver import PslqRelationSolver
from oeis_learn.discovery.symbolic_prover import SymbolicProver
from oeis_learn.discovery.vector_search import VectorRelationSearcher
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.trainer import EgcaGrpoTrainer
from oeis_learn.sandbox.runner import WasmRunner


def build_parser() -> argparse.ArgumentParser:
    """Builds argument parser adhering strictly to cli-interface.contract.json."""
    parser = argparse.ArgumentParser(
        prog="oeis-learn",
        description="Neuro-Symbolic Synthesis, Continuous Representations, and Mathematical Discovery for OEIS",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command 1: ingest
    ingest_p = subparsers.add_parser(
        "ingest",
        help="Ingest and index OEIS database records into local DuckDB / SQLite storage.",
    )
    ingest_p.add_argument("--data-path", type=str, required=False, help="Path to local OEIS data files")
    ingest_p.add_argument(
        "--db-path", type=str, default="data/oeis_learn.duckdb", help="Target database path"
    )
    ingest_p.add_argument(
        "--subset-stage", type=int, choices=[1, 2, 3, 4, 5], required=False, help="Extract stage subset"
    )

    # Command 2: train
    train_p = subparsers.add_parser(
        "train",
        help="Train the Tri-Stream Encoder and Transformer policy using S-GRPO / EGCA-GRPO.",
    )
    train_p.add_argument(
        "--config", type=str, default="configs/train_tier1.yaml", help="Path to YAML training configuration"
    )
    train_p.add_argument(
        "--sft-checkpoint", type=str, required=False, help="Optional warmed-up SFT checkpoint to initialize policy"
    )
    train_p.add_argument(
        "--tier", type=int, choices=[1, 2], default=1, help="Hardware tier (1 = workstation, 2 = cluster)"
    )
    train_p.add_argument(
        "--curriculum-stage", type=int, default=1, help="Initial curriculum stage"
    )
    train_p.add_argument(
        "--enable-cgi", action="store_true", default=True, help="Enable Conditional Ground-Truth Trajectory Injection"
    )
    train_p.add_argument(
        "--beta-sft", type=float, default=0.20, help="Co-training demonstration SFT loss weight"
    )
    train_p.add_argument(
        "--beta-kl", type=float, default=0.05, help="Schulman reference model KL divergence penalty weight"
    )
    train_p.add_argument(
        "--alpha-ent", type=float, default=0.01, help="Policy token entropy regularization bonus"
    )
    train_p.add_argument(
        "--enable-pbrs", action="store_true", default=True, help="Enable Potential-Based Reward Shaping"
    )
    train_p.add_argument(
        "--enable-lexicase", action="store_true", default=True, help="Enable down-sampled lexicase selection"
    )
    train_p.add_argument(
        "--num-cpu-threads", type=int, default=8, help="Rayon CPU worker threads for WASM execution"
    )
    train_p.add_argument(
        "--enable-symple", action="store_true", default=True, help="Enable SYMPLE multi-task bandit and EDB replay"
    )
    train_p.add_argument(
        "--enable-solver", action="store_true", default=True, help="Enable decoupled Diophantine/SMT constant solver"
    )
    train_p.add_argument(
        "--enable-dce", action="store_true", default=True, help="Enable compiler dead code elimination (DCE) pass"
    )
    train_p.add_argument(
        "--run-name", type=str, default="006_phase4_decoupled_symple", help="Run directory name"
    )

    # Command 3: generate-sft
    gen_sft_p = subparsers.add_parser(
        "generate-sft",
        help="Generate synthetic forward-execution demonstration dataset of sequence-program pairs.",
    )
    gen_sft_p.add_argument(
        "--output-path", type=str, default="data/sft_demonstrations.json", help="Output path for the generated dataset"
    )
    gen_sft_p.add_argument(
        "--num-samples", type=int, default=5000, help="Number of synthetic program-sequence pairs to generate"
    )
    gen_sft_p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible generation"
    )
    gen_sft_p.add_argument(
        "--enable-affine-sweeps", action="store_true", default=True, help="Apply randomized affine scaling sweeps"
    )
    gen_sft_p.add_argument(
        "--scale-min-pow", type=float, default=0.0, help="Minimum magnitude exponent for affine scaling"
    )
    gen_sft_p.add_argument(
        "--scale-max-pow", type=float, default=5.0, help="Maximum magnitude exponent for affine scaling"
    )

    # Command: solve-constants
    solve_p = subparsers.add_parser(
        "solve-constants",
        help="Ground abstract WAT skeletons containing i64.const_? placeholders via Diophantine/SMT solvers.",
    )
    solve_p.add_argument("--wat-file", type=str, required=True, help="Path to input WAT skeleton file")
    solve_p.add_argument("--terms", type=str, required=True, help="Comma-separated 20 integer sequence terms")
    solve_p.add_argument("--timeout-ms", type=int, default=250, help="Z3 SMT solver timeout in milliseconds")
    solve_p.add_argument("--output-wat", type=str, required=False, help="Path to write grounded WAT file")

    # Command 4: warmup-sft
    warmup_sft_p = subparsers.add_parser(
        "warmup-sft",
        help="Execute Supervised Fine-Tuning (SFT) pretraining on synthetic demonstrations.",
    )
    warmup_sft_p.add_argument(
        "--dataset-path", type=str, default="data/sft_demonstrations.json", help="Path to synthetic demonstration JSON file"
    )
    warmup_sft_p.add_argument(
        "--output-checkpoint", type=str, default="checkpoints/sft_warmup_best.pt", help="Destination path for checkpoint"
    )
    warmup_sft_p.add_argument(
        "--epochs", type=int, default=5, help="Number of SFT training epochs"
    )
    warmup_sft_p.add_argument(
        "--lr", type=float, default=0.0005, help="Initial learning rate"
    )

    # Command 5: test-progressive
    prog_p = subparsers.add_parser(
        "test-progressive",
        help="Run the 5-tier progressive validation hierarchy (Tiers 0 through 3) before full training.",
    )
    prog_p.add_argument(
        "--max-tier", type=int, choices=[0, 1, 2, 3], default=3, help="Highest tier to execute in pre-flight suite"
    )
    prog_p.add_argument(
        "--policy", type=str, default="configs/readiness_tier1_v1.json", help="Path to versioned readiness policy JSON"
    )
    prog_p.add_argument(
        "--output-report", type=str, default="reports/progressive_validation_report.json", help="Path to write JSON report"
    )
    prog_p.add_argument(
        "--output-markdown", type=str, default=None, help="Path to write Markdown projection"
    )
    prog_p.add_argument(
        "--diagnostic-override", action="store_true", default=False, help="Authorize diagnostic override for unqualified run"
    )
    prog_p.add_argument("--override-operator", type=str, default=None, help="Operator authorizing diagnostic override")
    prog_p.add_argument("--override-reason", type=str, default=None, help="Technical reason for diagnostic override")
    prog_p.add_argument("--override-intent", type=str, default=None, help="Diagnostic intent for override")

    # Command 6: synthesize
    synth_p = subparsers.add_parser(
        "synthesize",
        help="Synthesize a WebAssembly algorithm for a target sequence.",
    )
    synth_p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    synth_p.add_argument(
        "--benchmark-manifest",
        type=str,
        default="data/benchmarks/trustworthy_synthesis_v1.json",
        help="Path to frozen benchmark manifest",
    )
    synth_p.add_argument("--oeis-id", type=str, required=False, help="Target OEIS sequence ID (e.g., A000045)")
    synth_p.add_argument("--terms", type=str, required=False, help="Comma-separated integer terms")
    synth_p.add_argument(
        "--candidate-budget", type=int, choices=[1, 8, 16], default=8, help="Candidate budget (1, 8, or 16)"
    )
    synth_p.add_argument("--seed", type=int, default=42, help="Deterministic evaluation random seed")
    synth_p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    synth_p.add_argument("--top-p", type=float, default=0.95, help="Nucleus top-p parameter")
    synth_p.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens")
    synth_p.add_argument(
        "--constant-resolution",
        dest="constant_resolution",
        action="store_true",
        default=True,
        help="Enable constant resolution",
    )
    synth_p.add_argument(
        "--no-constant-resolution",
        dest="constant_resolution",
        action="store_false",
        help="Disable constant resolution",
    )
    synth_p.add_argument(
        "--solver-timeout-ms", type=int, default=250, help="SMT solver timeout in milliseconds"
    )
    synth_p.add_argument(
        "--fuel-per-invocation", type=int, default=10000, help="Maximum execution fuel per compute(n) call"
    )
    synth_p.add_argument("--fuel-budget", type=int, default=10000, help="Deprecated alias for fuel-per-invocation")
    synth_p.add_argument(
        "--memory-limit-mib", type=int, default=16, help="Linear memory ceiling in MiB"
    )
    synth_p.add_argument(
        "--mdl-max", type=float, default=1.20, help="Maximum allowed Minimum Description Length ratio"
    )
    synth_p.add_argument("--device", type=str, default="cpu", help="Torch device")
    synth_p.add_argument("--output-json", type=str, default=None, help="Destination file for evaluation JSON report")
    synth_p.add_argument(
        "--output-markdown", type=str, default=None, help="Destination file for Markdown projection"
    )
    synth_p.add_argument(
        "--diagnostic", action="store_true", default=False, help="Explicit diagnostic mode for unverified targets"
    )
    synth_p.add_argument(
        "--extrapolate", type=int, default=100, help="Deprecated alias for unseen horizon"
    )

    # Command 7: convert-checkpoint
    conv_p = subparsers.add_parser(
        "convert-checkpoint",
        help="Convert legacy checkpoint to Checkpoint v2 format.",
    )
    conv_p.add_argument("--input-checkpoint", type=str, required=True, help="Path to input legacy checkpoint")
    conv_p.add_argument("--config", type=str, required=True, help="Path to config YAML")
    conv_p.add_argument("--output-checkpoint", type=str, required=True, help="Path to output v2 checkpoint")

    # Command 8: run-ablations
    abl_p = subparsers.add_parser(
        "run-ablations",
        help="Executes a predeclared paired experiment manifest.",
    )
    abl_p.add_argument("--manifest", type=str, required=True, help="Path to experiment manifest JSON")
    abl_p.add_argument(
        "--output-directory",
        type=str,
        default="reports/experiments",
        help="Destination folder for experiment artifacts",
    )
    abl_p.add_argument(
        "--resume", action="store_true", default=False, help="Resume incomplete cells without modifying manifest"
    )

    # Command 9: discover
    disc_p = subparsers.add_parser(
        "discover",
        help="Run the self-supervised manifold discovery and PSLQ theorem verification pipeline.",
    )
    disc_p.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    disc_p.add_argument(
        "--benchmark-manifest",
        type=str,
        default="data/benchmarks/trustworthy_synthesis_v1.json",
        help="Path to frozen benchmark manifest",
    )
    disc_p.add_argument(
        "--protocol",
        type=str,
        default="configs/discovery_protocol_v1.json",
        help="Path to discovery protocol JSON",
    )
    disc_p.add_argument(
        "--definitions",
        type=str,
        default="data/benchmarks/symbolic_definitions_v1.json",
        help="Path to reviewed symbolic definitions JSON",
    )
    disc_p.add_argument("--seed", type=int, default=42, help="Random seed for discovery")
    disc_p.add_argument(
        "--normalize-l2", action="store_true", default=True, help="Apply L2 normalization to latent vectors"
    )
    disc_p.add_argument(
        "--distance-threshold", type=float, default=0.8, help="Manifold distance threshold epsilon for candidate triples"
    )
    disc_p.add_argument(
        "--max-candidates", type=int, default=50, help="Maximum number of vector relation candidates to verify"
    )
    disc_p.add_argument(
        "--precision-digits", type=int, default=500, help="Decimal precision digits for mpmath sampling"
    )
    disc_p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Destination JSON report for discovery findings",
    )
    disc_p.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="Destination Markdown report for discovery findings",
    )
    disc_p.add_argument(
        "--output-proofs",
        type=str,
        default="reports/discovered_proofs.md",
        help="Destination file for verified symbolic proofs (legacy alias)",
    )

    # Command 8: list-runs
    subparsers.add_parser(
        "list-runs",
        help="List all tracked experiment runs, configurations, and summary metrics.",
    )

    return parser


def handle_ingest(args: argparse.Namespace) -> int:
    """Handles the `ingest` subcommand."""
    print(f"Initializing database at: {args.db_path}")
    pipeline = OeisIngestionPipeline(db_path=args.db_path)

    if args.data_path and os.path.exists(args.data_path):
        stripped = os.path.join(args.data_path, "stripped")
        names = os.path.join(args.data_path, "names")
        count = pipeline.ingest_from_files(
            stripped_path=stripped if os.path.exists(stripped) else args.data_path,
            names_path=names if os.path.exists(names) else None,
            stage_filter=args.subset_stage,
        )
    else:
        # Generate representative synthetic curriculum dataset
        print("Generating synthetic multi-stage curriculum dataset...")
        count = pipeline.generate_synthetic_curriculum_dataset(num_per_stage=50)

    pipeline.close()
    print(f"Successfully ingested and indexed {count} sequence records.")
    return 0


def handle_train(args: argparse.Namespace) -> int:
    """Handles the `train` subcommand."""
    print(f"Loading configuration from: {args.config}")
    cfg: Dict[str, Any] = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    d_model = cfg.get("model", {}).get("d_model", 256)
    n_heads = cfg.get("model", {}).get("n_heads", 4)
    db_path = cfg.get("data", {}).get("db_path", "data/oeis_learn.duckdb")

    encoder = TriStreamEncoder(d_model=d_model, n_heads=n_heads, n_encoder_layers=2)
    decoder = WatTransformerDecoder(d_model=d_model, n_heads=n_heads, n_decoder_layers=2)
    scheduler = CurriculumScheduler(initial_stage=args.curriculum_stage)

    # Initialize synthetic sequences if db empty
    if not os.path.exists(db_path):
        ingest_p = OeisIngestionPipeline(db_path=db_path)
        ingest_p.generate_synthetic_curriculum_dataset(num_per_stage=20)
        ingest_p.close()

    dataset = OeisSequenceDataset(db_path=db_path)
    sampler = DynamicMixtureSampler(records=dataset.records, scheduler=scheduler)
    runner = WasmRunner(fuel_budget=10000)

    trainer = EgcaGrpoTrainer(
        encoder=encoder,
        decoder=decoder,
        scheduler=scheduler,
        sampler=sampler,
        wasm_runner=runner,
        rollout_group_size=4,
    )

    print(f"Starting EGCA-GRPO training on Stage {args.curriculum_stage} ({len(dataset)} sequences)...")
    for epoch in range(1, 3):
        batch_records = sampler.sample_batch(batch_size=4)
        for rec in batch_records:
            metrics = trainer.train_step_for_prompt(rec)
            print(f"Epoch {epoch} | Prompt: {rec.oeis_id} | Loss: {metrics['loss']:.4f} | Pass Rate: {metrics['pass_rate']:.2f}")

    print("Training step completed successfully.")
    return 0


def handle_generate_sft(args: argparse.Namespace) -> int:
    """Handles the `generate-sft` subcommand."""
    from oeis_learn.data.synthetic_generator import SyntheticDemonstrationGenerator

    print(f"Generating {args.num_samples} synthetic demonstrations (seed={args.seed})...")
    gen = SyntheticDemonstrationGenerator(seed=args.seed)
    dataset = gen.generate_dataset(num_samples=args.num_samples)
    gen.save_dataset(dataset, args.output_path)
    print(f"Generated {len(dataset.samples)} synthetic demonstration pairs -> {args.output_path}")
    return 0


def handle_warmup_sft(args: argparse.Namespace) -> int:
    """Handles the `warmup-sft` subcommand."""
    from oeis_learn.rl.sft_trainer import SftTrainer

    print(f"Starting SFT Warmup from {args.dataset_path} for {args.epochs} epochs...")
    trainer = SftTrainer(
        dataset_path=args.dataset_path,
        output_checkpoint=args.output_checkpoint,
        epochs=args.epochs,
        lr=args.lr,
    )
    res = trainer.train()
    print(f"SFT Warmup completed: final loss = {res['final_loss']:.4f}, checkpoint = {args.output_checkpoint}")
    return 0


def handle_test_progressive(args: argparse.Namespace) -> int:
    """Handles the `test-progressive` subcommand."""
    from scripts.run_progressive_validation import run_progressive_with_policy

    if args.diagnostic_override:
        if not (args.override_operator and args.override_reason and args.override_intent):
            print("Validation error: --diagnostic-override requires all of --override-operator, --override-reason, and --override-intent")
            return 2

    print(f"Running progressive validation hierarchy up to Tier {args.max_tier} with policy {args.policy}...")
    return run_progressive_with_policy(
        max_tier=args.max_tier,
        policy_path=args.policy,
        output_report_path=args.output_report,
        output_markdown_path=args.output_markdown,
        diagnostic_override=args.diagnostic_override,
        override_operator=args.override_operator,
        override_reason=args.override_reason,
        override_intent=args.override_intent,
    )


def handle_convert_checkpoint(args: argparse.Namespace) -> int:
    """Handles the `convert-checkpoint` subcommand."""
    from oeis_learn.evaluation.checkpoint import convert_legacy_checkpoint, load_checkpoint_v2
    import yaml

    if not os.path.exists(args.input_checkpoint):
        print(f"Validation error: input checkpoint not found: {args.input_checkpoint}")
        return 2
    if not os.path.exists(args.config):
        print(f"Validation error: config YAML not found: {args.config}")
        return 2

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        model_cfg = cfg.get("model", {})
        enc_cfg = {
            "d_model": model_cfg.get("d_model", 256),
            "n_heads": model_cfg.get("n_heads", 4),
            "n_encoder_layers": model_cfg.get("n_encoder_layers", 4),
            "d_ff": model_cfg.get("d_ff", 1024),
            "dropout": model_cfg.get("dropout", 0.1),
            "primes": model_cfg.get("primes", [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]),
            "max_valuation": model_cfg.get("max_valuation", 32),
            "use_film": True,
        }
        dec_cfg = {
            "d_model": model_cfg.get("d_model", 256),
            "n_heads": model_cfg.get("n_heads", 4),
            "n_decoder_layers": model_cfg.get("n_decoder_layers", 4),
            "d_ff": model_cfg.get("d_ff", 1024),
            "dropout": model_cfg.get("dropout", 0.1),
        }

        prov = convert_legacy_checkpoint(
            legacy_checkpoint_path=args.input_checkpoint,
            output_v2_path=args.output_checkpoint,
            encoder_config=enc_cfg,
            decoder_config=dec_cfg,
            producer_version="oeis-learn-0.1.0",
        )
        print(f"Converted legacy checkpoint to v2 format: {args.output_checkpoint}")
        print(f"Checksum: {prov.checkpoint_sha256} (source: {prov.source_checkpoint_sha256})")

        # Verify load
        load_checkpoint_v2(args.output_checkpoint)
        return 0
    except Exception as e:
        print(f"Execution error converting checkpoint: {e}")
        return 3


def handle_synthesize(args: argparse.Namespace) -> int:
    """Handles the `synthesize` subcommand using the shared evaluation workflow."""
    from oeis_learn.data.benchmark import BenchmarkTarget, compute_term_fingerprint, load_benchmark_manifest
    from oeis_learn.evaluation.checkpoint import load_checkpoint_v2
    from oeis_learn.evaluation.protocol import EvaluationProtocol
    from oeis_learn.evaluation.synthesis import evaluate_cohort_synthesis
    from oeis_learn.cli.reporting import project_synthesis_markdown, save_authoritative_json
    import torch

    device = torch.device(args.device)

    # 1. Load Checkpoint
    try:
        encoder, decoder, checkpoint_prov = load_checkpoint_v2(args.checkpoint, device=device)
    except Exception as e:
        print(f"Validation error: could not load checkpoint {args.checkpoint}: {e}")
        return 2

    # 2. Load Manifest & Target
    target = None
    manifest_sha = "sha256:" + "0" * 64
    if args.benchmark_manifest and os.path.exists(args.benchmark_manifest):
        try:
            manifest = load_benchmark_manifest(args.benchmark_manifest)
            manifest_sha = manifest.manifest_sha256
            if args.oeis_id:
                target = next((t for t in manifest.targets if t.oeis_id == args.oeis_id), None)
        except Exception as e:
            print(f"Validation error: could not load benchmark manifest {args.benchmark_manifest}: {e}")
            return 2

    if target is None:
        if args.diagnostic and args.terms:
            raw_terms = [int(x.strip()) for x in args.terms.split(",") if x.strip()]
            if len(raw_terms) < 120:
                print(f"Validation error: diagnostic evaluation requires exactly 120 terms, got {len(raw_terms)}")
                return 2
            obs_s = [str(x) for x in raw_terms[:20]]
            uns_s = [str(x) for x in raw_terms[20:120]]
            target = BenchmarkTarget(
                oeis_id=args.oeis_id or "A000000",
                name=f"Diagnostic Sequence {args.oeis_id or 'Custom'}",
                offset=0,
                family="DIAGNOSTIC",
                curriculum_stage=1,
                observed_terms=obs_s,
                unseen_terms=uns_s,
                result_profile="i64_scalar_v1",
                terms_sha256=compute_term_fingerprint(raw_terms),
                term_fingerprint=compute_term_fingerprint(raw_terms),
                tags=["diagnostic"],
            )
        else:
            print(f"Validation error: target '{args.oeis_id}' not found in manifest and no valid diagnostic terms provided.")
            return 2

    # 3. Build Protocol
    try:
        protocol = EvaluationProtocol.from_dict({
            "schema_version": "1.0",
            "checkpoint_sha256": checkpoint_prov.checkpoint_sha256,
            "benchmark_manifest_sha256": manifest_sha,
            "observed_horizon": 20,
            "unseen_horizon": 100,
            "candidate_budget": args.candidate_budget,
            "base_seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "constant_resolution": args.constant_resolution,
            "solver_timeout_ms": args.solver_timeout_ms,
            "max_placeholders": 4,
            "fuel_per_invocation": min(10000, args.fuel_per_invocation),
            "memory_limit_mib": min(16, args.memory_limit_mib),
            "mdl_ratio_max": min(1.20, args.mdl_max),
            "native_evaluator_required": True,
            "code_revision": "trustworthy-v1",
            "environment_fingerprint": "sha256:" + "0" * 64,
        })
    except Exception as e:
        print(f"Protocol validation error: {e}")
        return 2

    # 4. Run Evaluation
    try:
        res = evaluate_cohort_synthesis(
            encoder=encoder,
            decoder=decoder,
            checkpoint=checkpoint_prov,
            target=target,
            protocol=protocol,
            device=device,
        )
    except Exception as e:
        print(f"Execution error during synthesis: {e}")
        return 3

    res_dict = res.to_dict()

    # 5. Output JSON & Markdown
    if args.output_json:
        save_authoritative_json(res_dict, args.output_json, schema_name="synthesis-evaluation")
    if args.output_markdown:
        project_synthesis_markdown(res_dict, output_path=args.output_markdown)

    # Console summary
    print(project_synthesis_markdown(res_dict))

    if res.status == "QUALIFIED_SUCCESS":
        return 0
    return 1


def handle_discover(args: argparse.Namespace) -> int:
    """Handles the `discover` subcommand using the shared discovery pipeline."""
    from oeis_learn.cli.reporting import project_discovery_markdown, save_authoritative_json
    from oeis_learn.discovery.pipeline import run_discovery_pipeline

    if not os.path.exists(args.checkpoint):
        print(f"Validation error: checkpoint not found: {args.checkpoint}")
        return 2
    if not os.path.exists(args.benchmark_manifest):
        print(f"Validation error: benchmark manifest not found: {args.benchmark_manifest}")
        return 2
    if not os.path.exists(args.protocol):
        print(f"Validation error: discovery protocol not found: {args.protocol}")
        return 2

    print(f"Running latent manifold discovery and PSLQ theorem prover (seed: {args.seed})...")

    try:
        report_data = run_discovery_pipeline(
            checkpoint_path=args.checkpoint,
            manifest_path=args.benchmark_manifest,
            protocol_path=args.protocol,
            definitions_path=args.definitions,
            seed=args.seed,
        )
    except Exception as e:
        print(f"Execution error during discovery: {e}")
        return 3

    out_json = args.output_json or "reports/discovery_report.json"
    save_authoritative_json(report_data, out_json, schema_name="discovery-report")

    out_md = args.output_markdown or args.output_proofs
    md_content = project_discovery_markdown(report_data, output_path=out_md)
    print(md_content)

    return 0


def handle_solve_constants(args: argparse.Namespace) -> int:
    """Handles the `solve-constants` subcommand."""
    from oeis_learn.decoder.constant_solver import (
        parse_ast_placeholders,
        solve_linear_diophantine,
        solve_smt_constants,
        splice_constants_into_wat,
    )

    if not os.path.exists(args.wat_file):
        print(f"Error: WAT file not found: {args.wat_file}")
        return 1

    with open(args.wat_file, "r") as f:
        wat_skeleton = f.read()

    terms = [int(x.strip()) for x in args.terms.split(",") if x.strip()]
    if len(terms) < 20:
        print(f"Error: expected at least 20 integer terms, got {len(terms)}")
        return 1

    skeleton = parse_ast_placeholders(wat_code=wat_skeleton)
    print(f"Parsed skeleton: {skeleton.placeholder_count} placeholders, linear={skeleton.is_linear}")

    runner = WasmRunner(fuel_budget=10000)
    result = solve_linear_diophantine(skeleton=skeleton, terms=terms, runner=runner)
    if not result.is_sat and not skeleton.is_linear:
        print("Linear solver UNSAT. Trying Z3 SMT solver fallback...")
        result = solve_smt_constants(skeleton=skeleton, terms=terms, timeout_ms=args.timeout_ms)

    if result.is_sat and result.constants:
        print(f"Solution Found ({result.solver_type}) in {result.solve_duration_ms:.2f}ms: {result.constants}")
        grounded_wat = splice_constants_into_wat(skeleton, result.constants)
        if args.output_wat:
            with open(args.output_wat, "w") as f:
                f.write(grounded_wat)
            print(f"Grounded WAT written to {args.output_wat}")
        else:
            print("\nGrounded WAT Code:")
            print(grounded_wat)
        return 0
    else:
        print(f"Failed to solve constants: {result.error_message}")
        return 1


def handle_run_ablations(args: argparse.Namespace) -> int:
    """Handles the `run-ablations` subcommand."""
    from scripts.run_trustworthy_ablations import run_ablation_manifest

    if not os.path.exists(args.manifest):
        print(f"Validation error: manifest not found: {args.manifest}")
        return 2

    return run_ablation_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_directory,
        resume=args.resume,
    )


def handle_list_runs(args: argparse.Namespace) -> int:
    """Handles the `list-runs` subcommand."""
    from oeis_learn.tracking.run_manager import RunManager

    manager = RunManager()
    runs = manager.list_runs()
    if not runs:
        print("No tracked experiment runs found in runs/")
        return 0

    print("\n--- Tracked Experiment Runs ---")
    print(f"{'Run ID':<8} {'Name':<28} {'Status':<12} {'Created At':<26}")
    print("-" * 80)
    for r in runs:
        run_id = r.get("run_id", "N/A")
        name = r.get("name", "N/A")
        status = r.get("status", "UNKNOWN")
        created = r.get("created_at", "N/A")[:19]
        print(f"{run_id:<8} {name:<28} {status:<12} {created:<26}")
        if "summary_metrics" in r and r["summary_metrics"]:
            metrics_str = ", ".join(f"{k}={v}" for k, v in r["summary_metrics"].items())
            print(f"   ↳ Metrics: {metrics_str}")

    print("-" * 80)
    print(f"Total Runs: {len(runs)} | Next Run ID: {manager.get_next_run_id()}\n")
    return 0


def cli(args: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    if parsed_args.command == "ingest":
        return handle_ingest(parsed_args)
    elif parsed_args.command == "train":
        return handle_train(parsed_args)
    elif parsed_args.command == "generate-sft":
        return handle_generate_sft(parsed_args)
    elif parsed_args.command == "warmup-sft":
        return handle_warmup_sft(parsed_args)
    elif parsed_args.command == "test-progressive":
        return handle_test_progressive(parsed_args)
    elif parsed_args.command == "convert-checkpoint":
        return handle_convert_checkpoint(parsed_args)
    elif parsed_args.command == "run-ablations":
        return handle_run_ablations(parsed_args)
    elif parsed_args.command == "synthesize":
        return handle_synthesize(parsed_args)
    elif parsed_args.command == "solve-constants":
        return handle_solve_constants(parsed_args)
    elif parsed_args.command == "discover":
        return handle_discover(parsed_args)
    elif parsed_args.command == "list-runs":
        return handle_list_runs(parsed_args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(cli())
