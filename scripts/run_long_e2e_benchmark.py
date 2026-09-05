#!/usr/bin/env python3
"""Autonomous Long-Running End-to-End Real-Data Training and Discovery Benchmark.

Engineered for Tier 1 Local Workstation Baseline:
- 4 CPU Cores / 8 Threads
- 64 GB RAM
- 4 GB VRAM (FP32 Strict Precision, micro-batching)
- Duration: 2 to 12 hours of training & exploration
- Uses structured experiment tracking in runs/<RUN_ID>_<NAME>/
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
import time
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
from oeis_learn.data.models import (
    EliteReplayBufferEntry,
    LatentDiscoveryCandidate,
    SequenceRecord,
)
from oeis_learn.data.real_data_loader import RealOeisDataLoader
from oeis_learn.data.synthetic_generator import SyntheticDemonstrationGenerator
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.discovery.manifold import cluster_latent_manifold, reduce_manifold_2d
from oeis_learn.discovery.pslq_solver import PslqRelationSolver
from oeis_learn.discovery.symbolic_prover import SymbolicProver
from oeis_learn.discovery.vector_search import VectorRelationSearcher
from oeis_learn.discovery.vicreg_loss import VicRegLoss
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer
from oeis_learn.rl.sft_trainer import SftTrainer
from oeis_learn.rl.trainer import EgcaGrpoTrainer
from oeis_learn.sandbox.runner import WasmRunner
from oeis_learn.tracking.run_manager import RunManager


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("oeis_long_run")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def run_benchmark(
    run_id: Optional[str] = None,
    name: str = "phase2_bootstrapping",
    num_epochs: int = 60,
    steps_per_epoch: int = 100,
    config_path: str = "configs/train_tier1.yaml",
    skip_preflight: bool = False,
    sft_samples: int = 2500,
    sft_epochs: int = 5,
):
    start_time = time.time()

    # 1. Initialize Run Tracking Context
    run_manager = RunManager()
    cfg: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    cfg["benchmark"] = {
        "num_epochs": num_epochs,
        "steps_per_epoch": steps_per_epoch,
        "sft_samples": sft_samples,
        "sft_epochs": sft_epochs,
        "skip_preflight": skip_preflight,
    }

    ctx = run_manager.create_run(run_id=run_id, name=name, config=cfg)
    logger = setup_logger(str(ctx.log_file))
    ctx.set_status("RUNNING")

    logger.info("=" * 80)
    logger.info(f"Starting Autonomous OEIS-Learn Training & Discovery: Run {ctx.metadata.run_id} ({ctx.metadata.name})")
    logger.info(f"Run Directory: {ctx.run_dir}")
    logger.info(f"Target Duration: ~10-12 hours ({num_epochs} RL epochs x {steps_per_epoch} steps)")
    logger.info(f"PID: {os.getpid()}")
    logger.info("=" * 80)

    # 2. Pre-Flight Progressive Validation Suite (Tiers 0 through 3)
    if not skip_preflight:
        logger.info("Executing Pre-Flight Progressive Validation Suite (Tiers 0 through 3)...")
        from oeis_learn.rl.progressive import run_progressive_suite
        preflight_report = run_progressive_suite(
            max_tier=3, output_report_path=str(ctx.preflight_report_path)
        )
        if not preflight_report.overall_passed:
            logger.error("Pre-Flight Progressive Validation failed! Halting run.")
            ctx.set_status("FAILED")
            ctx.record_summary_metrics({"error": "Pre-flight validation failed"})
            return 1
        logger.info("Pre-Flight Progressive Validation PASSED! Authorizing training.")
    else:
        logger.info("Skipping pre-flight validation as requested.")

    # 3. Real Data Ingestion & Dataset Construction
    logger.info("Stage 1: Ingesting curated real OEIS sequence catalogs...")
    db_path = str(ctx.run_dir / "oeis_learn.duckdb")
    loader = RealOeisDataLoader()
    records = loader.load_local_benchmark_records()
    logger.info(f"Loaded {len(records)} real sequences from local catalog.")

    elite_buffer = EliteSeedDemonstrationBuffer()
    pipeline = OeisIngestionPipeline(db_path=db_path)
    pipeline.insert_records(records)
    synth_count = pipeline.generate_synthetic_curriculum_dataset(num_per_stage=100, elite_buffer=elite_buffer)
    logger.info(f"Generated {synth_count} synthetic multi-stage curriculum sequences.")
    pipeline.close()

    dataset = OeisSequenceDataset(db_path=db_path)
    total_seqs = len(dataset)
    logger.info(f"Total active sequence dataset size: {total_seqs} sequences.")

    # 4. SFT Demonstration Generation & Warmup
    logger.info("Stage 1.5: Generating synthetic demonstration dataset and running SFT warmup...")
    sft_data_path = str(ctx.run_dir / "sft_demonstrations.json")
    gen = SyntheticDemonstrationGenerator(seed=42)
    sft_dataset = gen.generate_dataset(num_samples=sft_samples)
    gen.save_dataset(sft_dataset, sft_data_path)
    logger.info(f"Generated {len(sft_dataset.samples)} synthetic demonstration pairs.")

    for s in sft_dataset.samples:
        elite_buffer.add_entry(
            EliteReplayBufferEntry(
                oeis_id=s.sample_id,
                terms=s.terms,
                wat_code=s.wat_code,
                byte_size=s.byte_size,
                extrapolation_passed=True,
                mdl_ratio=0.90,
                source="SYNTHETIC",
            )
        )

    # 5. Model Initialization (Tier 1 Workstation Baseline)
    d_model = cfg.get("model", {}).get("d_model", 256)
    n_heads = cfg.get("model", {}).get("n_heads", 4)
    n_layers = cfg.get("model", {}).get("n_encoder_layers", 4)
    d_ff = cfg.get("model", {}).get("d_ff", 1024)
    primes = cfg.get("model", {}).get("primes", None)
    max_val = cfg.get("model", {}).get("max_valuation", 16)
    moduli_count = cfg.get("model", {}).get("moduli_count", 100)

    logger.info(f"Initializing TriStreamEncoder & WatDecoder (d_model={d_model}, layers={n_layers}, strict FP32)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TriStreamEncoder(
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_layers,
        d_ff=d_ff,
        primes=primes,
        max_valuation=max_val,
        moduli_count=moduli_count,
    )
    decoder = WatTransformerDecoder(d_model=d_model, n_heads=n_heads, n_decoder_layers=n_layers, d_ff=d_ff)

    # Run SFT Pretraining Warmup
    sft_ckpt_path = ctx.get_checkpoint_path("sft_warmup_best.pt")
    sft_trainer = SftTrainer(
        dataset_path=sft_data_path,
        output_checkpoint=sft_ckpt_path,
        encoder=encoder,
        decoder=decoder,
        epochs=sft_epochs,
        lr=5.0e-4,
        batch_size=16,
        device=device,
    )
    sft_res = sft_trainer.train()
    logger.info(f"SFT Warmup Complete: final loss = {sft_res['final_loss']:.4f}")

    # Freeze reference decoder for Schulman KL divergence penalty
    ref_decoder = WatTransformerDecoder(d_model=d_model, n_heads=n_heads, n_decoder_layers=n_layers, d_ff=d_ff)
    ref_decoder.load_state_dict(decoder.state_dict())
    ref_decoder.to(device)
    ref_decoder.eval()

    scheduler = CurriculumScheduler(initial_stage=1, window_size=20)
    sampler = DynamicMixtureSampler(records=dataset.records, scheduler=scheduler)
    wasm_runner = WasmRunner(fuel_budget=10000)

    train_cfg = cfg.get("training", {})
    rl_cfg = cfg.get("rl", {})
    trainer = EgcaGrpoTrainer(
        encoder=encoder,
        decoder=decoder,
        scheduler=scheduler,
        sampler=sampler,
        wasm_runner=wasm_runner,
        elite_buffer=elite_buffer,
        ref_decoder=ref_decoder,
        lr=train_cfg.get("learning_rate", 5e-4),
        rollout_group_size=train_cfg.get("rollout_group_size", 8),
        asymmetric_penalty_weight=rl_cfg.get("asymmetric_penalty_weight", 1.33),
        enable_cgi=True,
        use_composite_rewards=True,
        beta_sft=rl_cfg.get("beta_sft", 0.20),
        beta_kl=rl_cfg.get("beta_kl", 0.05),
        alpha_ent=rl_cfg.get("alpha_ent", 0.01),
        enable_pbrs=rl_cfg.get("pbrs", {}).get("enabled", True),
        enable_lexicase=rl_cfg.get("lexicase", {}).get("enabled", True),
        sampling_temperature=train_cfg.get("sampling_temperature_initial", 0.4),
        device=device,
    )

    # 6. Main EGCA-GRPO Curriculum RL Training Loop
    logger.info(f"Stage 2: Beginning {num_epochs} Epochs of S-GRPO Curriculum Training...")
    epoch_losses = []
    epoch_pass_rates = []

    for epoch in range(1, num_epochs + 1):
        losses = []
        pass_rates = []

        for step in range(steps_per_epoch):
            batch_records = sampler.sample_batch(batch_size=4)
            for rec in batch_records:
                metrics = trainer.train_step_for_prompt(rec, epoch=epoch)
                losses.append(metrics["loss"])
                pass_rates.append(metrics["pass_rate"])

        mean_loss = float(np.mean(losses))
        mean_pass_rate = float(np.mean(pass_rates))
        epoch_losses.append(mean_loss)
        epoch_pass_rates.append(mean_pass_rate)

        # Check curriculum stage graduation
        graduated, new_stage = scheduler.check_and_update_graduation()
        c_k, min_cov = scheduler.compute_stage_competence()
        elapsed_min = (time.time() - start_time) / 60.0

        logger.info(
            f"Epoch {epoch:03d}/{num_epochs:03d} | "
            f"Loss: {mean_loss:.4f} | "
            f"Pass Rate: {mean_pass_rate*100:.1f}% | "
            f"Active Stage: {scheduler.active_stage} | "
            f"Competence C(S_k): {c_k:.3f} | "
            f"Min Coverage: {min_cov:.3f} | "
            f"ACR: {trainer.telemetry.current_acr:.2f} | "
            f"Elapsed: {elapsed_min:.1f}m"
        )

        if graduated:
            logger.info(f"🎉 CURRICULUM GRADUATION: Advanced to Stage {new_stage}!")

        # Periodic checkpointing
        if epoch % 10 == 0 or epoch == num_epochs:
            ckpt_path = ctx.get_checkpoint_path(f"model_epoch_{epoch:03d}.pt")
            trainer.save_checkpoint(
                ckpt_path,
                epoch=epoch,
                metadata={
                    "loss": mean_loss,
                    "pass_rate": mean_pass_rate,
                    "stage": scheduler.active_stage,
                },
            )
            logger.info(f"Saved model checkpoint to {ckpt_path}")

    # Save telemetry history
    trainer.telemetry.save_json(str(ctx.telemetry_file))

    # 7. Program Synthesis & Extrapolation Verification
    logger.info("Stage 3: Running Comprehensive Program Synthesis & Anti-Memorization Benchmarks...")
    from oeis_learn.data.benchmark import load_benchmark_manifest
    from oeis_learn.evaluation.protocol import EvaluationProtocol
    from oeis_learn.evaluation.synthesis import evaluate_cohort_synthesis
    from oeis_learn.cli.reporting import save_authoritative_json

    manifest_path = "data/benchmarks/trustworthy_synthesis_v1.json"
    benchmark_manifest = load_benchmark_manifest(manifest_path) if os.path.exists(manifest_path) else None

    # Save final Checkpoint v2 for evaluation
    final_v2_ckpt = ctx.get_checkpoint_path("model_final.v2.pt")
    from oeis_learn.evaluation.checkpoint import save_checkpoint_v2
    checkpoint_prov = save_checkpoint_v2(
        checkpoint_path=final_v2_ckpt,
        encoder=encoder,
        decoder=decoder,
        encoder_config={
            "d_model": d_model,
            "n_heads": n_heads,
            "n_encoder_layers": n_layers,
            "d_ff": d_ff,
            "dropout": 0.1,
            "primes": primes,
            "max_valuation": max_val,
            "use_film": True,
        },
        decoder_config={
            "d_model": d_model,
            "n_heads": n_heads,
            "n_decoder_layers": n_layers,
            "d_ff": d_ff,
            "dropout": 0.1,
        },
        epoch=num_epochs,
        producer_version="oeis-learn-0.1.0",
    )

    eval_sequences = ["A000217", "A000079", "A000045", "A000032", "A000129", "A000290"]
    synthesis_results = []
    cohort_eval_results = []

    for oeis_id in eval_sequences:
        target = None
        if benchmark_manifest:
            target = next((t for t in benchmark_manifest.targets if t.oeis_id == oeis_id), None)

        if not target:
            rec = next((r for r in dataset.records if r.oeis_id == oeis_id), None)
            if not rec:
                continue
            from oeis_learn.data.benchmark import BenchmarkTarget, compute_term_fingerprint
            target = BenchmarkTarget(
                oeis_id=oeis_id,
                name=rec.name,
                offset=0,
                family="BENCHMARK",
                curriculum_stage=rec.curriculum_stage,
                observed_terms=[str(x) for x in rec.terms[:20]],
                unseen_terms=[str(x) for x in rec.terms[20:120]] if len(rec.terms) >= 120 else [str(x) for x in rec.terms[20:]],
                result_profile="i64_scalar_v1",
                terms_sha256=compute_term_fingerprint(rec.terms),
                term_fingerprint=compute_term_fingerprint(rec.terms),
                tags=rec.tags,
            )

        eval_proto = EvaluationProtocol.from_dict({
            "schema_version": "1.0",
            "checkpoint_sha256": checkpoint_prov.checkpoint_sha256,
            "benchmark_manifest_sha256": benchmark_manifest.manifest_sha256 if benchmark_manifest else ("sha256:" + "0" * 64),
            "observed_horizon": 20,
            "unseen_horizon": 100,
            "candidate_budget": 8,
            "base_seed": 42,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 128,
            "constant_resolution": True,
            "solver_timeout_ms": 250,
            "max_placeholders": 4,
            "fuel_per_invocation": 10000,
            "memory_limit_mib": 16,
            "mdl_ratio_max": 1.20,
            "native_evaluator_required": True,
            "code_revision": "trustworthy-v1",
            "environment_fingerprint": "sha256:" + "0" * 64,
        })

        eval_res = evaluate_cohort_synthesis(
            encoder=encoder,
            decoder=decoder,
            checkpoint=checkpoint_prov,
            target=target,
            protocol=eval_proto,
            device=device,
        )
        cohort_eval_results.append(eval_res.to_dict())

        # Extract best candidate summary for legacy reporting compatibility
        best_cand = next((c for c in eval_res.candidates if c.classification == "EXTRAPOLATING_SUCCESS"), eval_res.candidates[0])
        synthesis_results.append({
            "oeis_id": oeis_id,
            "name": target.name,
            "status": "SUCCESS" if best_cand.classification == "EXTRAPOLATING_SUCCESS" else "FAILED",
            "fuel": best_cand.max_fuel or 0,
            "extrap_passed": best_cand.classification == "EXTRAPOLATING_SUCCESS",
            "mdl_ratio": best_cand.mdl_ratio or 0.0,
            "unique_candidates": eval_res.unique_candidate_count,
        })
        logger.info(
            f"Synthesize {oeis_id} | Status: {synthesis_results[-1]['status']} | "
            f"Extrap (100 terms): {synthesis_results[-1]['extrap_passed']} | "
            f"Unique: {eval_res.unique_candidate_count}/8"
        )

    save_authoritative_json(synthesis_results, str(ctx.synthesis_results_path))
    detailed_eval_path = str(ctx.reports_dir / "synthesis_evaluations_v1.json")
    save_authoritative_json({"evaluations": cohort_eval_results}, detailed_eval_path)

    # 8. Latent Space Manifold Discovery & PSLQ Theorem Proving
    logger.info("Stage 4: Extracting Continuous Manifold & Running PSLQ Automated Theorem Discovery...")
    from oeis_learn.discovery.pipeline import run_discovery_pipeline
    from oeis_learn.cli.reporting import project_discovery_markdown

    proto_cfg = "configs/discovery_protocol_v1.json"
    defs_cfg = "data/benchmarks/symbolic_definitions_v1.json"

    discovery_report = run_discovery_pipeline(
        checkpoint_path=final_v2_ckpt,
        manifest_path=manifest_path if os.path.exists(manifest_path) else "data/benchmarks/trustworthy_synthesis_v1.json",
        protocol_path=proto_cfg if os.path.exists(proto_cfg) else "configs/discovery_protocol_v1.json",
        definitions_path=defs_cfg if os.path.exists(defs_cfg) else "data/benchmarks/symbolic_definitions_v1.json",
        device=device,
    )
    discovery_report_path = str(ctx.reports_dir / "discovery_report_v1.json")
    save_authoritative_json(discovery_report, discovery_report_path, schema_name="discovery-report")
    project_discovery_markdown(discovery_report, output_path=str(ctx.theorems_path))

    proven_theorems_count = discovery_report["summary"]["symbolically_proven"]
    n_clusters = discovery_report["summary"]["unique_claims"]
    logger.info(f"Discovery complete: {proven_theorems_count} proven, {discovery_report['summary']['numerical_conjectures']} conjectures.")

    total_duration_hours = (time.time() - start_time) / 3600.0
    with open(ctx.summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Run {ctx.metadata.run_id} ({ctx.metadata.name}) Execution Summary\n\n")
        f.write(f"- **Execution Date**: {datetime.datetime.now().isoformat()}\n")
        f.write(f"- **Total Duration**: {total_duration_hours:.2f} hours\n")
        f.write(f"- **Final Stage Competence**: {c_k:.3f}\n")
        f.write(f"- **Final Average Loss**: {epoch_losses[-1]:.4f}\n")
        f.write(f"- **Final Average Pass Rate**: {epoch_pass_rates[-1]*100:.1f}%\n")
        f.write(f"- **Verified Theorems**: {proven_theorems_count}\n")
        f.write(f"- **Discovered Families**: {n_clusters}\n\n")
        f.write("## Synthesis Evaluation Results\n\n")
        f.write("| OEIS ID | Status | Fuel | Extrap Passed | MDL Ratio |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for s in synthesis_results:
            f.write(f"| `{s['oeis_id']}` | `{s['status']}` | `{s['fuel']}` | `{s['extrap_passed']}` | `{s['mdl_ratio']:.2f}` |\n")

    # Update metadata
    ctx.record_summary_metrics({
        "total_duration_hours": round(total_duration_hours, 2),
        "final_stage_competence": round(c_k, 3),
        "final_pass_rate": round(epoch_pass_rates[-1], 4),
        "final_loss": round(epoch_losses[-1], 4),
        "verified_theorems": proven_theorems_count,
        "discovered_families": n_clusters,
    })
    ctx.set_status("COMPLETED")

    logger.info("=" * 80)
    logger.info(f"Run {ctx.metadata.run_id} Complete in {total_duration_hours:.2f} hours.")
    logger.info(f"Artifacts saved to {ctx.run_dir}")
    logger.info("=" * 80)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run Long OEIS Learn Training Benchmark")
    parser.add_argument("--run-id", type=str, required=False, help="Explicit run ID (e.g., 003)")
    parser.add_argument("--name", type=str, default="phase3_inductive_generalization", help="Descriptive run name")
    parser.add_argument("--epochs", type=int, default=60, help="Number of RL training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=100, help="Number of steps per epoch")
    parser.add_argument("--config", type=str, default="configs/train_tier1.yaml", help="Path to config file")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip pre-flight progressive suite")
    parser.add_argument("--sft-samples", type=int, default=2500, help="Number of synthetic SFT demonstration pairs")
    parser.add_argument("--sft-epochs", type=int, default=5, help="Number of SFT pretraining epochs")

    args = parser.parse_args()
    ret = run_benchmark(
        run_id=args.run_id,
        name=args.name,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        config_path=args.config,
        skip_preflight=args.skip_preflight,
        sft_samples=args.sft_samples,
        sft_epochs=args.sft_epochs,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
