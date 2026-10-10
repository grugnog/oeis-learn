"""Corrected original policy trainer; whole-trajectory reward and true behavior logs."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from oeis_learn.curriculum.sampler import DynamicMixtureSampler
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.decoder.constant_solver import (
    parse_ast_placeholders,
    solve_constants,
)
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import EOS_ID, PAD_ID
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer
from oeis_learn.rl.telemetry import DiagnosticTelemetryTracker
from oeis_learn.sandbox.runner import WasmRunner

logger = logging.getLogger(__name__)


class EgcaGrpoTrainer:
    """Trainer orchestrating TriStreamEncoder + WatTransformerDecoder via S-GRPO / EGCA-GRPO curriculum learning

    with explicit masked objectives and a frozen encoder/decoder reference.
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
        asymmetric_penalty_weight: float = 1.0,
        enable_cgi: bool = False,
        use_composite_rewards: bool = False,
        beta_sft: float = 0.0,
        beta_kl: float = 0.01,
        alpha_ent: float = 0.0,
        enable_pbrs: bool = False,
        enable_lexicase: bool = False,
        sampling_temperature: float = 0.8,
        max_program_length: int = 256,
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        result_profile: str = "i64_scalar_v1",
        ref_encoder=None,
        objective="grpo",
        seed=20260913,
    ):
        if (
            enable_cgi
            or use_composite_rewards
            or enable_pbrs
            or enable_lexicase
            or beta_sft
            or asymmetric_penalty_weight != 1.0
        ):
            raise ValueError(
                "unsupported legacy shaping/injection/blending; use whole-trajectory policy and archive SFT replay"
            )
        if (ref_encoder is None) != (ref_decoder is None):
            raise ValueError("both reference encoder and decoder must be supplied together")
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.scheduler = scheduler
        self.sampler = sampler
        self.strict_codec = getattr(decoder, "codec_sha256", None) is not None
        if self.strict_codec and (wasm_runner is None or not hasattr(wasm_runner, "run_single")):
            raise ValueError("body-codec training requires an explicit FoundationRunner")
        self.wasm_runner = wasm_runner or WasmRunner(fuel_budget=10000)
        self.elite_buffer = elite_buffer or EliteSeedDemonstrationBuffer()
        self.telemetry = telemetry_tracker or DiagnosticTelemetryTracker()
        from oeis_learn.rl.policy import FrozenReference

        self.reference = (
            FrozenReference(ref_encoder or encoder, ref_decoder or decoder) if beta_kl else None
        )
        self.ref_decoder = self.reference.decoder if self.reference else None
        self.objective, self.seed, self.rollout_cursor = objective, seed, 0
        self.replay_examples_used = 0
        if objective not in ("grpo", "reinforce"):
            raise ValueError("explicit grpo or reinforce objective required")
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
        self.result_profile = result_profile
        self.device = device or torch.device("cpu")
        self.current_epoch = 1

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        if self.ref_decoder is not None:
            self.reference.encoder.to(self.device)
            self.ref_decoder.to(self.device)
            self.ref_decoder.eval()

        # Joint AdamW optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.program_sampler = WatProgramSampler(
            decoder=self.decoder,
            max_length=self.max_program_length,
            temperature=sampling_temperature,
            top_p=1.0,
            use_cache=True,
            codec_profile="wat_body_decimal_v1"
            if getattr(self.decoder, "codec_sha256", None)
            else None,
        )

    def train_step_for_prompt(self, record, epoch=None, group_size=None):
        from oeis_learn.rl.policy import scoring_mode

        with scoring_mode(self.encoder, self.decoder):
            return self._policy_step(record, epoch, group_size)

    def _policy_step(self, record, epoch, group_size):
        import time
        from oeis_learn.rl.policy import (
            group_advantages,
            masked_distribution,
            model_digest,
            policy_loss,
            trajectory_mean,
        )
        from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

        started = time.monotonic()
        if epoch is not None:
            self.current_epoch = epoch
        count = self.rollout_group_size if group_size is None else group_size
        if type(count) is not int or not 1 <= count <= 16:
            raise ValueError("bounded actual rollout group required")
        terms = list(record.terms[:20])
        if len(terms) != 20 or any(type(v) is not int for v in terms):
            raise ValueError("exact twenty-term visible prompt required")
        encoder_id, decoder_id = model_digest(self.encoder), model_digest(self.decoder)
        with torch.no_grad():
            behavior_memory = self.encoder.forward_from_sequences([terms], device=self.device)
        seeds = [
            int(digest([self.seed, self.rollout_cursor + i, terms])[7:23], 16) for i in range(count)
        ]
        trajectories = self.program_sampler.sample_trajectories(
            behavior_memory.expand(count, -1, -1), seeds, policy=True
        )
        self.rollout_cursor += count
        candidates, self.last_grounding_results = [], []
        for trajectory in trajectories:
            grounded = (
                solve_constants(parse_ast_placeholders(trajectory.source), terms)
                if trajectory.source
                else None
            )
            self.last_grounding_results.append(grounded)
            candidates.append(
                grounded.grounded_wat
                if grounded is not None and grounded.is_sat
                else trajectory.source
            )
        if self.strict_codec:
            from types import SimpleNamespace

            results = []
            for i, source in enumerate(candidates):
                if not source:
                    execution = SimpleNamespace(status="INCOMPLETE", output=[], total_fuel=0)
                else:
                    evidence = self.wasm_runner.run_single(
                        source, list(range(20)), request_id=f"policy-{self.rollout_cursor}-{i}"
                    ).to_dict()
                    execution = SimpleNamespace(
                        status="SUCCESS"
                        if evidence["outcome"] is None
                        and evidence["outputs"] == evidence["reference_outputs"]
                        else "EXECUTION_FAILURE",
                        output=[int(v) for v in evidence["outputs"]],
                        total_fuel=evidence["fuel"] or 0,
                        evidence=evidence,
                    )
                results.append((execution, None))
        else:
            results = self.wasm_runner.run_optimized_batch(
                candidates,
                fuel_budget=self.wasm_runner.fuel_budget,
                terms_to_generate=20,
                result_profile=self.result_profile,
            )
        rewards, successes = [], []
        for trajectory, (execution, _) in zip(trajectories, results):
            matched = (
                trajectory.complete and execution.status == "SUCCESS" and execution.output == terms
            )
            successes.append(matched)
            rewards.append(1.0 if matched else -1.0)
        reward = torch.tensor(rewards, device=self.device)
        advantages = group_advantages(reward) if self.objective == "grpo" else reward
        length = max(map(lambda t: len(t.tokens), trajectories))
        pad_id = self.program_sampler.codec.PAD_ID if self.strict_codec else PAD_ID
        eos_id = self.program_sampler.codec.EOS_ID if self.strict_codec else EOS_ID
        tokens = torch.full((count, length), pad_id, dtype=torch.long, device=self.device)
        supports = torch.zeros(
            (count, length - 1, self.decoder.vocab_size), dtype=torch.bool, device=self.device
        )
        supports[..., pad_id] = True  # inactive padded rows need a harmless distribution
        old = torch.zeros((count, length - 1), device=self.device)
        action_mask = torch.zeros_like(old, dtype=torch.bool)
        for row, trajectory in enumerate(trajectories):
            tokens[row, : len(trajectory.tokens)] = torch.tensor(
                trajectory.tokens, device=self.device
            )
            for position in range(1, len(trajectory.tokens)):
                supports[row, position - 1].zero_()
                supports[row, position - 1, list(trajectory.supports[position])] = True
                if not trajectory.forced[position]:
                    old[row, position - 1] = trajectory.log_probabilities[position]
                    action_mask[row, position - 1] = True
        self.optimizer.zero_grad(set_to_none=True)
        memory = self.encoder.forward_from_sequences([terms], device=self.device).expand(
            count, -1, -1
        )
        logits = self.decoder(
            tokens[:, :-1], memory, tgt_key_padding_mask=tokens[:, :-1].eq(pad_id)
        )
        reference_logits = None
        if self.reference is not None:
            reference_logits = self.reference.logits(
                [terms] * count, tokens[:, :-1], self.device, pad_id
            )
        loss = policy_loss(
            logits,
            tokens[:, 1:],
            old,
            advantages,
            supports=supports,
            temperatures=self.sampling_temperature,
            action_mask=action_mask,
            objective=self.objective,
            reference_logits=reference_logits,
            kl_coefficient=self.beta_kl,
            entropy_coefficient=self.alpha_ent,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            1.0,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.scheduler.record_visit(record.oeis_id, sum(successes), count)
        _, _, entropies = masked_distribution(logits.detach(), supports, self.sampling_temperature)
        entropy = float(trajectory_mean(entropies, action_mask))
        self.last_rollouts = []
        for trajectory, (execution, _), value in zip(trajectories, results, rewards):
            self.last_rollouts.append(
                trajectory.record(
                    encoder_sha256=encoder_id,
                    decoder_sha256=decoder_id,
                    prompt_sha256=digest([str(v) for v in terms]),
                    execution_sha256=digest(execution.__dict__),
                    reward_components={
                        "fit": value,
                        "validity": float(execution.status == "SUCCESS"),
                        "cost": float(execution.total_fuel),
                    },
                    eos_id=eos_id,
                )
            )
        variance = float(reward.var(unbiased=False))
        self.telemetry.record_step(
            epoch=self.current_epoch,
            step=len(self.telemetry.records) + 1,
            policy_entropy=entropy,
            reward_variance=variance,
            compiler_trapped=sum(e.status != "SUCCESS" for e, _ in results) > count // 2,
            prefix_length=20 * sum(successes) / count,
            active_stage=self.scheduler.active_stage,
        )
        return {
            "loss": float(loss.detach()),
            "pass_rate": sum(successes) / count,
            "pass_count": sum(successes),
            "group_size": count,
            "curriculum_stage": record.curriculum_stage,
            "mean_reward": float(reward.mean()),
            "entropy": entropy,
            "acr": self.telemetry.current_acr,
            "oeis_id": record.oeis_id,
            "compiler_traps": sum(e.status != "SUCCESS" for e, _ in results),
            "sft_loss": None,
            "reward_variance": variance,
            "zero_variance": variance == 0,
            "elapsed_s": time.monotonic() - started,
        }

    def replay_step(self, entries):
        raise ValueError(
            "legacy elite replay has no strict provenance; use archive-backed FrozenReplay and replay_update"
        )

    def save_checkpoint(self, checkpoint_path, epoch=0, metadata=None):
        """Atomic complete original-trainer continuation, including full reference.

        Qualified runs use QualifiedCheckpointStore and the append-only ledger;
        this explicit diagnostic format must not masquerade as that contract.
        """
        import os
        from pathlib import Path
        from oeis_learn.tracking.training_checkpoint import capture_learning_state

        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = capture_learning_state(
            self.encoder,
            self.decoder,
            self.optimizer,
            sampler={"seed": self.seed, "cursor": self.rollout_cursor},
            prefetch={"next": []},
            reference=self.reference,
        )
        payload = {
            "version": "original-policy-state/v1",
            "state": state,
            "epoch": epoch,
            "metadata": metadata or {},
            "objective": self.objective,
            "temperature": self.sampling_temperature,
            "kl": self.beta_kl,
            "entropy": self.alpha_ent,
        }
        temporary = path.with_name(path.name + ".pending")
        with temporary.open("xb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        from oeis_learn.tracking.foundation_metrics import fsync_directory

        fsync_directory(path.parent)

    def load_checkpoint(self, checkpoint_path):
        from oeis_learn.tracking.training_checkpoint import restore_learning_state

        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if payload.get("version") != "original-policy-state/v1":
            raise ValueError(
                "weights-only legacy checkpoint is inspection-only; cannot continue the policy"
            )
        if any(
            payload[name] != value
            for name, value in [
                ("objective", self.objective),
                ("temperature", self.sampling_temperature),
                ("kl", self.beta_kl),
                ("entropy", self.alpha_ent),
            ]
        ):
            raise ValueError("policy profile differs on resume")
        sampler, _ = restore_learning_state(
            payload["state"], self.encoder, self.decoder, self.optimizer, reference=self.reference
        )
        self.seed, self.rollout_cursor = sampler["seed"], sampler["cursor"]
        self.current_epoch = payload["epoch"]
        return payload
