"""End-to-end S-GRPO / EGCA-GRPO RL Training Engine with Trajectory Injection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from oeis_learn.curriculum.sampler import DynamicMixtureSampler
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.curriculum.symple_bandit import AdaGGroupAllocator, Exp3SBanditScheduler
from oeis_learn.data.models import SequenceRecord
from oeis_learn.decoder.constant_solver import (
    parse_ast_placeholders,
    solve_linear_diophantine,
    solve_smt_constants,
    splice_constants_into_wat,
)
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import BOS_ID, EOS_ID, PAD_ID, encode_wat
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.egca_grpo import (
    compute_egca_grpo_loss,
    compute_partitioned_semantic_entropy,
    get_dynamic_sampling_temperature,
    inject_virtual_sample_if_needed,
)
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer
from oeis_learn.rl.prompt_weighting import compute_sgrpo_advantages
from oeis_learn.rl.reward import (
    compute_binary_reward,
    compute_composite_reward,
    compute_covariant_parsimony_penalty,
    compute_dense_log_distance_reward,
    compute_lexicographic_advantages,
    compute_validity_reward,
)
from oeis_learn.rl.telemetry import DiagnosticTelemetryTracker
from oeis_learn.sandbox.runner import WasmRunner
from oeis_learn.sandbox.tracer import build_fine_grained_attribution, locate_divergence_token_span

logger = logging.getLogger(__name__)


class EgcaGrpoTrainer:
    """Trainer orchestrating TriStreamEncoder + WatTransformerDecoder via S-GRPO / EGCA-GRPO curriculum learning

    with SFT demonstration co-training, Schulman KL regularization, and potential-based reward shaping.
    """

    def __init__(
        self,
        encoder: TriStreamEncoder,
        decoder: WatTransformerDecoder,
        scheduler: CurriculumScheduler,
        sampler: Optional[DynamicMixtureSampler] = None,
        wasm_runner: Optional[WasmRunner] = None,
        elite_buffer: Optional[EliteSeedDemonstrationBuffer] = None,
        telemetry_tracker: Optional[DiagnosticTelemetryTracker] = None,
        ref_decoder: Optional[WatTransformerDecoder] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        rollout_group_size: int = 8,
        asymmetric_penalty_weight: float = 1.5,
        enable_cgi: bool = True,
        use_composite_rewards: bool = True,
        beta_sft: float = 0.20,
        beta_kl: float = 0.05,
        alpha_ent: float = 0.01,
        enable_pbrs: bool = True,
        enable_lexicase: bool = True,
        sampling_temperature: float = 0.4,
        max_program_length: int = 128,
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.scheduler = scheduler
        self.sampler = sampler
        self.wasm_runner = wasm_runner or WasmRunner(fuel_budget=10000)
        self.elite_buffer = elite_buffer or EliteSeedDemonstrationBuffer()
        self.telemetry = telemetry_tracker or DiagnosticTelemetryTracker()
        self.ref_decoder = ref_decoder
        self.rollout_group_size = rollout_group_size
        self.asymmetric_penalty_weight = asymmetric_penalty_weight
        self.enable_cgi = enable_cgi
        self.use_composite_rewards = use_composite_rewards
        self.beta_sft = beta_sft
        self.beta_kl = beta_kl
        self.alpha_ent = alpha_ent
        self.enable_pbrs = enable_pbrs
        self.enable_lexicase = enable_lexicase
        self.sampling_temperature = sampling_temperature
        self.max_program_length = max_program_length
        self.device = device or torch.device("cpu")
        self.current_epoch = 1

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        if self.ref_decoder is not None:
            self.ref_decoder.to(self.device)
            self.ref_decoder.eval()

        # Joint AdamW optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.program_sampler = WatProgramSampler(
            decoder=self.decoder,
            max_length=self.max_program_length,
            temperature=sampling_temperature,
        )

    def train_step_for_prompt(
        self,
        record: SequenceRecord,
        epoch: Optional[int] = None,
        group_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs a single S-GRPO / EGCA-GRPO training step on a prompt record."""
        if epoch is not None:
            self.current_epoch = epoch

        self.encoder.train()
        self.decoder.train()

        active_g = group_size if group_size is not None else self.rollout_group_size

        # 1. Encode prompt representation Z
        seq_input = [record.terms[:20]]
        z = self.encoder.forward_from_sequences(seq_input, device=self.device)  # (1, 20, d_model)
        z_expanded = z.expand(active_g, -1, -1)  # (G, 20, d_model)

        # 2. Sample G candidate programs (with recurrence accumulator prefix if Stage >= 2)
        is_rec = (
            record.curriculum_stage >= 2
            or any(tag in ("recurrence", "fibonacci", "lucas", "geometric") for tag in record.tags)
        )
        rec_prefix = '(module (func (export "compute") (param $n i32) (result i64) (local $a i64) (local $b i64) (local $temp i64) (local $i i32)'
        prefix_wat = rec_prefix if is_rec else None

        wat_programs, token_ids = self.program_sampler.sample(
            z_expanded,
            temperature=self.sampling_temperature,
            use_grammar_mask=True,
            prefix_wat=prefix_wat,
            max_length=self.max_program_length,
        )

        # 2b. Phase 4 Decoupled Constant Solver Dispatch
        grounded_wat_programs = []
        for wat in wat_programs:
            if "i64.const_?" in wat:
                skeleton = parse_ast_placeholders(wat)
                solver_res = solve_linear_diophantine(skeleton, record.terms[:20], self.wasm_runner)
                if not solver_res.is_sat and not skeleton.is_linear:
                    solver_res = solve_smt_constants(skeleton, record.terms[:20], timeout_ms=250, runner=self.wasm_runner)
                if solver_res.is_sat and solver_res.grounded_wat:
                    grounded_wat_programs.append(solver_res.grounded_wat)
                    # Ingest grounded program into EDB
                    self.elite_buffer.add_canonical_entry(
                        oeis_id=record.oeis_id,
                        wat_code=solver_res.grounded_wat,
                        terms=record.terms[:20],
                        step=self.current_epoch,
                    )
                else:
                    grounded_wat_programs.append(wat)
            else:
                grounded_wat_programs.append(wat)

        # 3. Batch evaluate across CPU worker threads with optional DCE optimization
        opt_artifacts = self.wasm_runner.run_optimized_batch(
            grounded_wat_programs, fuel_budget=10000, terms_to_generate=len(record.terms[:20])
        )
        exec_results = [r[0] for r in opt_artifacts]
        artifacts = [r[1] for r in opt_artifacts]

        # 4. Compute rewards and execution divergence spans
        rewards = []
        compiler_traps = 0
        prefix_lengths = []
        token_masks = torch.ones_like(token_ids, dtype=torch.float32, device=self.device)

        for i, res in enumerate(exec_results):
            if res.status != "SUCCESS":
                compiler_traps += 1

            # Phase 4 Dense Log-Distance Return + Hard Waste Threshold
            r_dense = compute_dense_log_distance_reward(res.output, record.terms[:20])
            r_val = compute_validity_reward(artifacts[i].waste_ratio, threshold=0.30)
            if res.status == "SUCCESS" and res.output == record.terms[: len(res.output)]:
                rew = 1.0
                div_step = None
                prefix_lengths.append(len(record.terms[:20]))
            else:
                rew = r_dense * (0.8 if r_val > 0.0 else 0.0) - 0.2
                div_step = len(res.output)
                prefix_lengths.append(0.0)

            rewards.append(rew)
            self.scheduler.record_outcome(record.oeis_id, success=(res.status == "SUCCESS" and div_step is None))

            # Apply fine-grained credit assignment and downstream zero-masking
            attr = build_fine_grained_attribution(
                wat_code=wat_programs[i],
                exec_result=res,
                target_terms=record.terms[:20],
                total_advantage=1.0,
                total_tokens=token_ids.size(1),
            )
            mask_vec = torch.tensor(attr.token_advantage_mask, dtype=torch.float32, device=self.device)
            cov_vec = torch.tensor(attr.executed_token_mask, dtype=torch.float32, device=self.device)
            if attr.failure_mode == "LOGIC" and attr.causal_token_end > attr.causal_token_start:
                token_masks[i] = mask_vec * cov_vec * token_ids.size(1)
            else:
                token_masks[i] = cov_vec

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        pass_count = sum(1 for res in exec_results if res.status == "SUCCESS" and res.output == record.terms[: len(res.output)])
        pass_rate = pass_count / len(rewards)

        # 5. Conditional Ground-Truth Trajectory Injection (CGI) if entire group fails
        ref_injected = False
        all_failed = (pass_count == 0)

        if all_failed and self.enable_cgi:
            elite_entry = self.elite_buffer.get_entry(record.oeis_id)
            if elite_entry is not None:
                # Inject reference solution
                ref_encoded = [BOS_ID] + encode_wat(elite_entry.wat_code) + [EOS_ID]
                max_len = max(token_ids.size(1), len(ref_encoded))

                if token_ids.size(1) < max_len:
                    pad_w = max_len - token_ids.size(1)
                    token_ids = F.pad(token_ids, (0, pad_w), value=PAD_ID)
                    token_masks = F.pad(token_masks, (0, pad_w), value=0.0)

                ref_tensor = torch.full((1, max_len), PAD_ID, dtype=torch.long, device=self.device)
                ref_tensor[0, : len(ref_encoded)] = torch.tensor(ref_encoded, dtype=torch.long, device=self.device)

                token_ids = torch.cat([ref_tensor, token_ids], dim=0)
                z_expanded = torch.cat([z, z_expanded], dim=0)
                rewards_tensor = torch.cat([torch.tensor([1.0], device=self.device), rewards_tensor])
                token_masks = torch.cat([torch.ones((1, max_len), device=self.device), token_masks], dim=0)
                ref_injected = True

        # 6. Compute S-GRPO advantages
        advantages = compute_sgrpo_advantages(
            rewards=rewards_tensor,
            ref_injected=ref_injected,
            asymmetric_penalty_weight=self.asymmetric_penalty_weight,
            use_avspo_anchor=True,
        )

        # 7. SFT Demonstration Co-Training Auxiliary Loss
        sft_loss = None
        if self.beta_sft > 0.0:
            demo_entry = self.elite_buffer.get_entry(record.oeis_id)
            if demo_entry is None:
                # Stochastic diverse sampling to prevent collapsing to a single template
                demo_entry = self.elite_buffer.sample_demonstration()
            if demo_entry is not None:
                demo_encoded = [BOS_ID] + encode_wat(demo_entry.wat_code) + [EOS_ID]
                demo_t = torch.tensor([demo_encoded], dtype=torch.long, device=self.device)
                demo_in = demo_t[:, :-1]
                demo_out = demo_t[:, 1:]
                # Encode demonstration's own sequence terms to prevent cross-conditioning mismatch
                z_demo = self.encoder.forward_from_sequences([demo_entry.terms[:20]], device=self.device)
                demo_logits = self.decoder(demo_in, z_demo, tgt_key_padding_mask=(demo_in == PAD_ID))
                sft_loss = F.cross_entropy(demo_logits.reshape(-1, demo_logits.size(-1)), demo_out.reshape(-1), ignore_index=PAD_ID)

        # 8. Forward pass through decoder to compute logits & KL divergence
        self.optimizer.zero_grad()
        tgt_in = token_ids[:, :-1]
        tgt_out = token_ids[:, 1:]
        pad_mask = (tgt_in == PAD_ID)

        logits = self.decoder(tgt_in, z_expanded, tgt_key_padding_mask=pad_mask)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()

        old_log_probs = F.log_softmax(logits.detach(), dim=-1).gather(
            dim=-1, index=tgt_out.unsqueeze(-1)
        ).squeeze(-1)

        ref_log_probs = None
        if self.ref_decoder is not None and self.beta_kl > 0.0:
            with torch.no_grad():
                ref_logits = self.ref_decoder(tgt_in, z_expanded, tgt_key_padding_mask=pad_mask)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1).gather(
                    dim=-1, index=tgt_out.unsqueeze(-1)
                ).squeeze(-1)

        loss = compute_egca_grpo_loss(
            logits=logits,
            old_log_probs=old_log_probs,
            token_ids=tgt_out,
            advantages=advantages,
            token_masks=token_masks[:, 1:],
            beta_kl=self.beta_kl,
            ref_log_probs=ref_log_probs,
            alpha_ent=self.alpha_ent,
            sft_loss=sft_loss,
            beta_sft=self.beta_sft,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0
        )
        self.optimizer.step()

        # 9. Telemetry tracking
        reward_var = float(rewards_tensor.var().item()) if len(rewards_tensor) > 1 else 0.0
        self.telemetry.record_step(
            epoch=self.current_epoch,
            step=len(self.telemetry.records) + 1,
            policy_entropy=entropy,
            reward_variance=reward_var,
            compiler_trapped=(compiler_traps > (active_g // 2)),
            prefix_length=float(np.mean(prefix_lengths)) if prefix_lengths else 0.0,
            active_stage=self.scheduler.active_stage,
        )

        return {
            "loss": float(loss.item()),
            "pass_rate": pass_rate,
            "pass_count": pass_count,
            "group_size": active_g,
            "mean_reward": float(rewards_tensor.mean().item()),
            "entropy": entropy,
            "acr": self.telemetry.current_acr,
            "oeis_id": record.oeis_id,
            "compiler_traps": compiler_traps,
            "sft_loss": float(sft_loss.item()) if sft_loss is not None else None,
        }

    def save_checkpoint(self, checkpoint_path: str, epoch: int = 0, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Saves encoder and decoder weights with training metadata."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        if self.encoder_config and self.decoder_config:
            from oeis_learn.evaluation.checkpoint import save_checkpoint_v2
            save_checkpoint_v2(
                checkpoint_path=checkpoint_path,
                encoder=self.encoder,
                decoder=self.decoder,
                encoder_config=self.encoder_config,
                decoder_config=self.decoder_config,
                epoch=epoch,
                producer_version="oeis-learn-0.1.0",
            )
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": self.encoder.state_dict(),
                    "decoder_state_dict": self.decoder.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metadata": metadata or {},
                },
                checkpoint_path,
            )

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Loads encoder and decoder weights from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
