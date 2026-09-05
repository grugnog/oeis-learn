#!/usr/bin/env python3
"""Automated Hyperparameter Optimization (HPO) and Encoder Constant Ablation Runner.

Evaluates:
1. Encoder Mathematical Constants:
   - PRIMES: [2, 3, 5, 7, 11, 13] vs [2, 3, 5, 7, 11, 13, 17, 19] vs 10 primes
   - MAX_VALUATION: 16 vs 32
   - MODULI_COUNT: 50 vs 100 vs 150
2. RL Regularization & Training Hyperparameters:
   - beta_sft (SFT co-training loss weight)
   - beta_kl (Schulman reference model penalty)
   - alpha_ent (Policy entropy regularization bonus)
   - gamma_pbrs (Potential discount factor)
   - k_lexicase (Down-sampled test cases)
   - learning_rate (AdamW policy learning rate)

Generates:
- reports/hpo_sweep_results.json
- reports/hpo_sweep_results.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from oeis_learn.curriculum.sampler import DynamicMixtureSampler
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.data.models import EliteReplayBufferEntry, SequenceRecord
from oeis_learn.data.synthetic_generator import SyntheticDemonstrationGenerator
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import BOS_ID, EOS_ID, PAD_ID, encode_wat
from oeis_learn.discovery.vicreg_loss import compute_rank_dispersion_ratio
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer
from oeis_learn.rl.trainer import EgcaGrpoTrainer
from oeis_learn.sandbox.runner import WasmRunner

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("oeis_hpo")


def evaluate_tier1_fitting(
    encoder: TriStreamEncoder,
    decoder: WatTransformerDecoder,
    device: torch.device,
    lr: float = 0.03,
    steps: int = 20,
) -> Dict[str, float]:
    """Evaluates Tier 1 oracle reference solution fitting and perplexity."""
    terms = [42 for _ in range(20)]
    wat_code = '(module (func (export "compute") (param $n i32) (result i64) (i64.const 42)))'
    encoded_tokens = [BOS_ID] + encode_wat(wat_code) + [EOS_ID]
    tgt_tensor = torch.tensor([encoded_tokens], dtype=torch.long, device=device)
    dec_input = tgt_tensor[:, :-1]
    dec_target = tgt_tensor[:, 1:]

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    ppl = 999.0
    steps_to_converge = steps

    for step in range(1, steps + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        memory = encoder.forward_from_sequences([terms], device=device)
        logits = decoder(dec_input, memory)
        loss = criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
        loss.backward()
        optimizer.step()

        current_ppl = float(np.exp(min(loss.item(), 20.0)))
        if current_ppl < 1.25 and steps_to_converge == steps:
            steps_to_converge = step
        ppl = current_ppl

    return {
        "final_oracle_ppl": ppl,
        "final_loss": float(loss.item()),
        "steps_to_converge": float(steps_to_converge),
    }


def evaluate_tier2_convergence(
    encoder: TriStreamEncoder,
    decoder: WatTransformerDecoder,
    runner: WasmRunner,
    device: torch.device,
    rl_params: Dict[str, Any],
    steps: int = 15,
) -> Dict[str, float]:
    """Evaluates Tier 2 single-prompt RL convergence and pass rate."""
    scheduler = CurriculumScheduler(initial_stage=1)
    elite_buffer = EliteSeedDemonstrationBuffer()

    trainer = EgcaGrpoTrainer(
        encoder=encoder,
        decoder=decoder,
        scheduler=scheduler,
        wasm_runner=runner,
        elite_buffer=elite_buffer,
        rollout_group_size=4,
        enable_cgi=True,
        lr=rl_params.get("learning_rate", 0.005),
        beta_sft=rl_params.get("beta_sft", 0.20),
        beta_kl=rl_params.get("beta_kl", 0.05),
        alpha_ent=rl_params.get("alpha_ent", 0.01),
        enable_pbrs=rl_params.get("enable_pbrs", True),
        device=device,
    )

    record = SequenceRecord(
        oeis_id="A000027",
        name="Positive integers: 1, 2, 3, 4, 5...",
        terms=[n for n in range(20)],
        curriculum_stage=1,
    )

    max_pass_rate = 0.0
    converged_step = steps
    losses = []

    for step in range(1, steps + 1):
        step_metrics = trainer.train_step_for_prompt(record, epoch=1)
        pr = step_metrics["pass_rate"]
        losses.append(step_metrics["loss"])
        if pr > max_pass_rate:
            max_pass_rate = pr
        if pr >= 0.20 and converged_step == steps:
            converged_step = step

    return {
        "tier2_max_pass_rate": max_pass_rate,
        "tier2_final_pass_rate": step_metrics["pass_rate"],
        "tier2_converged_step": float(converged_step),
        "tier2_final_acr": trainer.telemetry.current_acr,
        "tier2_mean_loss": float(np.mean(losses)),
    }


def evaluate_tier3_micro_cohort(
    encoder: TriStreamEncoder,
    decoder: WatTransformerDecoder,
    runner: WasmRunner,
    device: torch.device,
    rl_params: Dict[str, Any],
    num_tasks: int = 8,
    epochs: int = 2,
) -> Dict[str, float]:
    """Evaluates Tier 3 micro-cohort rolling competence and stability."""
    gen = SyntheticDemonstrationGenerator(seed=101)
    synth_dataset = gen.generate_dataset(num_samples=num_tasks)

    records: List[SequenceRecord] = []
    elite_buffer = EliteSeedDemonstrationBuffer()

    for idx, s in enumerate(synth_dataset.samples):
        oeis_id = f"A{idx:06d}"
        records.append(
            SequenceRecord(
                oeis_id=oeis_id,
                name=f"Synthetic {s.family}",
                terms=s.terms,
                curriculum_stage=1,
            )
        )
        elite_buffer.add_entry(
            EliteReplayBufferEntry(
                oeis_id=oeis_id,
                terms=s.terms,
                wat_code=s.wat_code,
                byte_size=s.byte_size,
                extrapolation_passed=True,
                mdl_ratio=0.90,
                source="SYNTHETIC",
            )
        )

    scheduler = CurriculumScheduler(initial_stage=1, window_size=10)
    sampler = DynamicMixtureSampler(records=records, scheduler=scheduler)

    trainer = EgcaGrpoTrainer(
        encoder=encoder,
        decoder=decoder,
        scheduler=scheduler,
        sampler=sampler,
        wasm_runner=runner,
        elite_buffer=elite_buffer,
        rollout_group_size=4,
        enable_cgi=True,
        lr=rl_params.get("learning_rate", 5e-4),
        beta_sft=rl_params.get("beta_sft", 0.20),
        beta_kl=rl_params.get("beta_kl", 0.05),
        alpha_ent=rl_params.get("alpha_ent", 0.01),
        enable_pbrs=rl_params.get("enable_pbrs", True),
        device=device,
    )

    for ep in range(1, epochs + 1):
        batch = sampler.sample_batch(batch_size=4)
        for rec in batch:
            trainer.train_step_for_prompt(rec, epoch=ep)

    competence = scheduler.get_competence_score(1)
    acr = trainer.telemetry.current_acr

    return {
        "tier3_competence": float(competence),
        "tier3_acr": float(acr),
    }


def compute_hpo_compound_score(metrics: Dict[str, float]) -> float:
    """Computes compound ranking score:

    Score = 0.40 * (1/PPL_ref) + 0.30 * Tier2_PassRate + 0.30 * Tier3_Competence - 0.50 * ACR
    """
    ppl = metrics.get("final_oracle_ppl", 10.0)
    inv_ppl = 1.0 / max(ppl, 1.0)
    tier2_pr = metrics.get("tier2_max_pass_rate", 0.0)
    tier3_comp = metrics.get("tier3_competence", 0.0)
    acr = metrics.get("tier3_acr", metrics.get("tier2_final_acr", 0.0))

    score = (0.40 * inv_ppl) + (0.30 * tier2_pr) + (0.30 * tier3_comp) - (0.50 * acr)
    return float(score)


def run_encoder_constant_ablation(
    search_space: Dict[str, Any],
    device: torch.device,
    d_model: int = 64,
) -> List[Dict[str, Any]]:
    """Runs systematic factorial grid ablation across encoder mathematical constants."""
    logger.info("=" * 80)
    logger.info("Starting Phase 1: Encoder Mathematical Constants Ablation Sweep")
    logger.info("=" * 80)

    ablation_cfg = search_space.get("encoder_constants_ablation", {})
    primes_list = ablation_cfg.get("primes_candidates", [
        {"name": "primes_6", "primes": [2, 3, 5, 7, 11, 13]},
        {"name": "primes_8", "primes": [2, 3, 5, 7, 11, 13, 17, 19]},
        {"name": "primes_10", "primes": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]},
    ])
    max_vals = ablation_cfg.get("max_valuation_candidates", [16, 32])
    moduli_counts = ablation_cfg.get("moduli_count_candidates", [50, 100, 150])

    runner = WasmRunner(fuel_budget=10000)
    results = []
    trial_idx = 1
    total_trials = len(primes_list) * len(max_vals) * len(moduli_counts)

    for p_cand in primes_list:
        p_name = p_cand["name"]
        primes = p_cand["primes"]
        for max_val in max_vals:
            for m_count in moduli_counts:
                trial_start = time.perf_counter()
                torch.manual_seed(42)

                encoder = TriStreamEncoder(
                    d_model=d_model,
                    n_heads=2,
                    n_encoder_layers=2,
                    d_ff=128,
                    primes=primes,
                    max_valuation=max_val,
                    moduli_count=m_count,
                ).to(device)
                decoder = WatTransformerDecoder(
                    d_model=d_model, n_heads=2, n_decoder_layers=2, d_ff=128
                ).to(device)

                # 1. Measure Rank Dispersion Ratio (RDR)
                sample_seqs = [[n**2 + 2*n + 1 for n in range(20)], [2**n for n in range(20)], [n * (n+1)//2 for n in range(20)], [3*n + 5 for n in range(20)]]
                with torch.no_grad():
                    z_mat = encoder.forward_from_sequences(sample_seqs, device=device)  # (4, 20, d_model)
                    rdr = compute_rank_dispersion_ratio(z_mat.reshape(-1, d_model))

                # 2. Tier 1 Oracle Fitting
                t1_metrics = evaluate_tier1_fitting(encoder, decoder, device=device, lr=0.03, steps=20)

                # 3. Tier 2 RL Convergence
                rl_params = {"learning_rate": 0.005, "beta_sft": 0.20, "beta_kl": 0.05, "alpha_ent": 0.01}
                t2_metrics = evaluate_tier2_convergence(encoder, decoder, runner, device=device, rl_params=rl_params, steps=15)

                metrics = {**t1_metrics, **t2_metrics, "rank_dispersion_ratio": float(rdr)}
                compound_score = compute_hpo_compound_score(metrics)
                latency = time.perf_counter() - trial_start

                trial_record = {
                    "trial_id": trial_idx,
                    "type": "encoder_ablation",
                    "primes_name": p_name,
                    "num_primes": len(primes),
                    "primes": primes,
                    "max_valuation": max_val,
                    "moduli_count": m_count,
                    "metrics": metrics,
                    "compound_score": compound_score,
                    "latency_seconds": round(latency, 2),
                }
                results.append(trial_record)

                logger.info(
                    f"Trial {trial_idx:02d}/{total_trials:02d} | "
                    f"Primes: {p_name} ({len(primes)}) | MaxVal: {max_val:2d} | Moduli: {m_count:3d} | "
                    f"PPL: {metrics['final_oracle_ppl']:.3f} | T2 PassRate: {metrics['tier2_max_pass_rate']*100:.0f}% | "
                    f"RDR: {rdr:.2f} | Score: {compound_score:.4f} | Latency: {latency:.2f}s"
                )
                trial_idx += 1

    # Sort results by compound score descending
    results.sort(key=lambda x: x["compound_score"], reverse=True)
    return results


def run_rl_hyperparameter_sweep(
    search_space: Dict[str, Any],
    best_encoder_params: Dict[str, Any],
    device: torch.device,
    n_trials: int = 15,
    d_model: int = 64,
) -> List[Dict[str, Any]]:
    """Runs randomized / Bayesian hyperparameter optimization over RL regularization parameters."""
    logger.info("=" * 80)
    logger.info(f"Starting Phase 2: RL Regularization Hyperparameter Sweep ({n_trials} Trials)")
    logger.info("=" * 80)

    rl_space = search_space.get("rl_hyperparameter_search_space", {})
    runner = WasmRunner(fuel_budget=10000)
    results = []

    # Grid / Sample candidates
    learning_rates = rl_space.get("learning_rate", {}).get("choices", [1.0e-4, 3.0e-4, 5.0e-4])
    gamma_pbrs_choices = rl_space.get("gamma_pbrs", {}).get("choices", [0.95, 0.99])
    k_lex_choices = rl_space.get("k_lexicase", {}).get("choices", [3, 5, 8])

    rng = random.Random(42)

    for trial_idx in range(1, n_trials + 1):
        trial_start = time.perf_counter()
        torch.manual_seed(42 + trial_idx)

        # Sample hyperparameters
        lr = rng.choice(learning_rates)
        beta_sft = round(rng.uniform(0.10, 0.35), 3)
        beta_kl = round(rng.uniform(0.01, 0.10), 3)
        alpha_ent = round(rng.uniform(0.005, 0.03), 3)
        gamma_pbrs = rng.choice(gamma_pbrs_choices)
        k_lex = rng.choice(k_lex_choices)
        asym_weight = round(rng.uniform(1.2, 1.8), 2)

        rl_params = {
            "learning_rate": lr,
            "beta_sft": beta_sft,
            "beta_kl": beta_kl,
            "alpha_ent": alpha_ent,
            "gamma_pbrs": gamma_pbrs,
            "k_lexicase": k_lex,
            "asymmetric_penalty_weight": asym_weight,
            "enable_pbrs": True,
        }

        encoder = TriStreamEncoder(
            d_model=d_model,
            n_heads=2,
            n_encoder_layers=2,
            d_ff=128,
            primes=best_encoder_params.get("primes", [2, 3, 5, 7, 11, 13]),
            max_valuation=best_encoder_params.get("max_valuation", 16),
            moduli_count=best_encoder_params.get("moduli_count", 100),
        ).to(device)
        decoder = WatTransformerDecoder(
            d_model=d_model, n_heads=2, n_decoder_layers=2, d_ff=128
        ).to(device)

        # 1. Tier 1 Oracle Fitting
        t1_metrics = evaluate_tier1_fitting(encoder, decoder, device=device, lr=lr * 5, steps=20)

        # 2. Tier 2 RL Convergence
        t2_metrics = evaluate_tier2_convergence(encoder, decoder, runner, device=device, rl_params=rl_params, steps=15)

        # 3. Tier 3 Micro-Cohort
        t3_metrics = evaluate_tier3_micro_cohort(encoder, decoder, runner, device=device, rl_params=rl_params, num_tasks=8, epochs=2)

        metrics = {**t1_metrics, **t2_metrics, **t3_metrics}
        compound_score = compute_hpo_compound_score(metrics)
        latency = time.perf_counter() - trial_start

        trial_record = {
            "trial_id": trial_idx,
            "type": "rl_hyperparameter_sweep",
            "hyperparameters": rl_params,
            "metrics": metrics,
            "compound_score": compound_score,
            "latency_seconds": round(latency, 2),
        }
        results.append(trial_record)

        logger.info(
            f"RL Trial {trial_idx:02d}/{n_trials:02d} | "
            f"LR: {lr:.1e} | b_sft: {beta_sft:.2f} | b_kl: {beta_kl:.3f} | a_ent: {alpha_ent:.3f} | "
            f"PPL: {metrics['final_oracle_ppl']:.3f} | T2 PR: {metrics['tier2_max_pass_rate']*100:.0f}% | "
            f"ACR: {metrics['tier3_acr']:.2f} | Score: {compound_score:.4f} | Latency: {latency:.2f}s"
        )

    results.sort(key=lambda x: x["compound_score"], reverse=True)
    return results


def export_hpo_reports(
    encoder_results: List[Dict[str, Any]],
    rl_results: List[Dict[str, Any]],
    output_json: str = "reports/hpo_sweep_results.json",
    output_md: str = "reports/hpo_sweep_results.md",
) -> None:
    """Exports structured JSON and Markdown reports summarizing HPO sweep results."""
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)

    best_encoder = encoder_results[0] if encoder_results else {}
    best_rl = rl_results[0] if rl_results else {}

    report_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "best_encoder_constants": {
            "primes": best_encoder.get("primes", [2, 3, 5, 7, 11, 13]),
            "num_primes": best_encoder.get("num_primes", 6),
            "max_valuation": best_encoder.get("max_valuation", 16),
            "moduli_count": best_encoder.get("moduli_count", 100),
            "compound_score": best_encoder.get("compound_score", 0.0),
        },
        "best_rl_hyperparameters": best_rl.get("hyperparameters", {}),
        "best_rl_score": best_rl.get("compound_score", 0.0),
        "encoder_ablation_trials": encoder_results,
        "rl_sweep_trials": rl_results,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Hyperparameter Optimization & Mathematical Constant Ablation Report\n\n")
        f.write(f"**Date**: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  \n")
        f.write(f"**Total Encoder Trials**: {len(encoder_results)}  \n")
        f.write(f"**Total RL Sweep Trials**: {len(rl_results)}  \n\n")

        f.write("## 🏆 Top Recommended Configuration\n\n")
        f.write("### 1. Optimal Encoder Mathematical Constants\n")
        f.write(f"- **PRIMES**: `{best_encoder.get('primes', [2, 3, 5, 7, 11, 13])}` ({best_encoder.get('num_primes', 6)} primes)\n")
        f.write(f"- **MAX_VALUATION**: `{best_encoder.get('max_valuation', 16)}`\n")
        f.write(f"- **MODULI_COUNT**: `{best_encoder.get('moduli_count', 100)}`\n")
        f.write(f"- **Ablation Score**: `{best_encoder.get('compound_score', 0.0):.4f}`\n\n")

        if best_rl:
            hp = best_rl.get("hyperparameters", {})
            f.write("### 2. Optimal RL Regularization Hyperparameters\n")
            f.write(f"- **Learning Rate**: `{hp.get('learning_rate', 3e-4)}`\n")
            f.write(f"- **beta_sft (Co-Training Weight)**: `{hp.get('beta_sft', 0.20)}`\n")
            f.write(f"- **beta_kl (Schulman Penalty)**: `{hp.get('beta_kl', 0.05)}`\n")
            f.write(f"- **alpha_ent (Entropy Bonus)**: `{hp.get('alpha_ent', 0.01)}`\n")
            f.write(f"- **gamma_pbrs (Potential Discount)**: `{hp.get('gamma_pbrs', 0.99)}`\n")
            f.write(f"- **k_lexicase (Test Subsample)**: `{hp.get('k_lexicase', 5)}`\n")
            f.write(f"- **Asymmetric Penalty Weight**: `{hp.get('asymmetric_penalty_weight', 1.5)}`\n")
            f.write(f"- **Compound RL Score**: `{best_rl.get('compound_score', 0.0):.4f}`\n\n")

        f.write("## 📊 Encoder Mathematical Constants Ablation Results\n\n")
        f.write("| Rank | Primes | Max Val | Moduli | PPL (Ref) | Tier 2 PassRate | RDR | Score | Latency |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for rank, r in enumerate(encoder_results, 1):
            m = r["metrics"]
            f.write(
                f"| {rank} | {r['primes_name']} ({r['num_primes']}) | {r['max_valuation']} | {r['moduli_count']} | "
                f"{m['final_oracle_ppl']:.3f} | {m['tier2_max_pass_rate']*100:.0f}% | {m['rank_dispersion_ratio']:.2f} | "
                f"**{r['compound_score']:.4f}** | {r['latency_seconds']:.2f}s |\n"
            )

        if rl_results:
            f.write("\n## 🎯 RL Hyperparameter Sweep Results\n\n")
            f.write("| Rank | LR | beta_sft | beta_kl | alpha_ent | gamma_pbrs | Tier 2 PR | ACR | Score |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for rank, r in enumerate(rl_results, 1):
                hp = r["hyperparameters"]
                m = r["metrics"]
                f.write(
                    f"| {rank} | {hp['learning_rate']:.1e} | {hp['beta_sft']:.2f} | {hp['beta_kl']:.3f} | "
                    f"{hp['alpha_ent']:.3f} | {hp['gamma_pbrs']} | {m.get('tier2_max_pass_rate', 0.0)*100:.0f}% | "
                    f"{m.get('tier3_acr', 0.0):.2f} | **{r['compound_score']:.4f}** |\n"
                )

    logger.info(f"Exported HPO reports -> {output_json} and {output_md}")


def main():
    parser = argparse.ArgumentParser(description="Run Automated HPO and Encoder Constant Ablation")
    parser.add_argument("--config", type=str, default="configs/hpo_search_space.yaml", help="Path to HPO search space config")
    parser.add_argument("--mode", type=str, choices=["all", "encoder", "rl"], default="all", help="Sweep mode")
    parser.add_argument("--n-rl-trials", type=int, default=15, help="Number of RL hyperparameter trials")
    parser.add_argument("--output-json", type=str, default="reports/hpo_sweep_results.json", help="Path to output JSON report")
    parser.add_argument("--output-md", type=str, default="reports/hpo_sweep_results.md", help="Path to output Markdown report")
    parser.add_argument("--apply-best", action="store_true", help="Apply best parameters to configs/train_tier1.yaml")

    args = parser.parse_args()

    search_space: Dict[str, Any] = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            search_space = yaml.safe_load(f) or {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Executing HPO on device: {device}")

    encoder_results = []
    best_encoder_params = {"primes": [2, 3, 5, 7, 11, 13], "max_valuation": 16, "moduli_count": 100}

    if args.mode in ("all", "encoder"):
        encoder_results = run_encoder_constant_ablation(search_space, device=device)
        if encoder_results:
            best_encoder_params = {
                "primes": encoder_results[0]["primes"],
                "max_valuation": encoder_results[0]["max_valuation"],
                "moduli_count": encoder_results[0]["moduli_count"],
            }

    rl_results = []
    if args.mode in ("all", "rl"):
        rl_results = run_rl_hyperparameter_sweep(search_space, best_encoder_params, device=device, n_trials=args.n_rl_trials)

    export_hpo_reports(encoder_results, rl_results, output_json=args.output_json, output_md=args.output_md)

    if args.apply_best and rl_results:
        best_rl = rl_results[0]["hyperparameters"]
        train_cfg_path = "configs/train_tier1.yaml"
        if os.path.exists(train_cfg_path):
            with open(train_cfg_path, "r", encoding="utf-8") as f:
                train_cfg = yaml.safe_load(f) or {}

            # Update encoder & RL parameters
            train_cfg.setdefault("model", {})["primes"] = best_encoder_params["primes"]
            train_cfg["model"]["max_valuation"] = best_encoder_params["max_valuation"]
            train_cfg["model"]["moduli_count"] = best_encoder_params["moduli_count"]

            train_cfg.setdefault("training", {})["learning_rate"] = best_rl["learning_rate"]
            train_cfg.setdefault("rl", {})["beta_sft"] = best_rl["beta_sft"]
            train_cfg["rl"]["beta_kl"] = best_rl["beta_kl"]
            train_cfg["rl"]["alpha_ent"] = best_rl["alpha_ent"]
            train_cfg["rl"]["asymmetric_penalty_weight"] = best_rl["asymmetric_penalty_weight"]
            train_cfg["rl"].setdefault("pbrs", {})["gamma"] = best_rl["gamma_pbrs"]
            train_cfg["rl"].setdefault("lexicase", {})["subsample_size"] = best_rl["k_lexicase"]

            with open(train_cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(train_cfg, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Successfully applied optimal parameters to {train_cfg_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
