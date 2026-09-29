"""Real-Time Diagnostic Telemetry and Early-Warning Monitor for Reinforcement Learning."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict
from typing import Callable, Deque, Dict, List, Optional
import numpy as np
import torch
from oeis_learn.data.models import DiagnosticTelemetryRecord

logger = logging.getLogger("oeis_learn.telemetry")


class DivergenceHaltError(Exception):
    """Raised when training diagnostic metrics cross critical divergence thresholds."""

    def __init__(self, message: str, record: Optional[DiagnosticTelemetryRecord] = None):
        super().__init__(message)
        self.record = record


class DiagnosticTelemetryTracker:
    """Tracks training dynamics and early warning indicators (ACR, entropy decay, reward variance, trap rate)."""

    def __init__(
        self,
        window_size: int = 20,
        acr_threshold: float = 0.30,
        entropy_min: float = 0.20,
        trap_rate_max: float = 0.15,
        halt_on_divergence: bool = False,
        on_warning_callback: Optional[Callable[[str, DiagnosticTelemetryRecord], None]] = None,
    ):
        self.window_size = window_size
        self.acr_threshold = acr_threshold
        self.entropy_min = entropy_min
        self.trap_rate_max = trap_rate_max
        self.halt_on_divergence = halt_on_divergence
        self.on_warning = on_warning_callback

        self.records: List[DiagnosticTelemetryRecord] = []
        self.variance_window: Deque[float] = deque(maxlen=window_size)
        self.zero_variance_flags: Deque[bool] = deque(maxlen=window_size)
        self.trap_window: Deque[bool] = deque(maxlen=window_size)
        self.prefix_window: Deque[float] = deque(maxlen=window_size)
        self.entropy_history: List[float] = []

        # Granular per-task training & candidate telemetry
        self.task_telemetry: Dict[str, Dict[str, Any]] = {}
        self.curriculum_events: List[Dict[str, Any]] = []

    def record_task_candidate(
        self,
        oeis_id: str,
        is_unique: bool = True,
        solver_outcome: Optional[str] = None,
        solver_duration_ms: float = 0.0,
        trap_reason: Optional[str] = None,
        observed_match: bool = False,
        unseen_match: bool = False,
        mdl_passed: bool = False,
    ) -> None:
        """Records detailed execution statistics for a single generated candidate on a task."""
        if oeis_id not in self.task_telemetry:
            self.task_telemetry[oeis_id] = {
                "generated_count": 0,
                "unique_count": 0,
                "solver_attempts": 0,
                "solver_successes": 0,
                "solver_total_ms": 0.0,
                "trap_count": 0,
                "trap_reasons": {},
                "observed_matches": 0,
                "unseen_matches": 0,
                "mdl_passes": 0,
                "selection_probability": 0.0,
                "allocated_rollouts": 0,
                "replay_events": 0,
            }

        stats = self.task_telemetry[oeis_id]
        stats["generated_count"] += 1
        if is_unique:
            stats["unique_count"] += 1
        if solver_outcome is not None:
            stats["solver_attempts"] += 1
            stats["solver_total_ms"] += solver_duration_ms
            if solver_outcome == "PASSED":
                stats["solver_successes"] += 1
        if trap_reason:
            stats["trap_count"] += 1
            stats["trap_reasons"][trap_reason] = stats["trap_reasons"].get(trap_reason, 0) + 1
        if observed_match:
            stats["observed_matches"] += 1
        if unseen_match:
            stats["unseen_matches"] += 1
        if mdl_passed:
            stats["mdl_passes"] += 1

    def record_curriculum_decision(
        self,
        oeis_id: str,
        selection_probability: float,
        allocated_rollouts: int,
        is_replay: bool = False,
    ) -> None:
        """Records curriculum selection and compute allocation decision."""
        if oeis_id not in self.task_telemetry:
            self.record_task_candidate(oeis_id)
        stats = self.task_telemetry[oeis_id]
        stats["selection_probability"] = selection_probability
        stats["allocated_rollouts"] += allocated_rollouts
        if is_replay:
            stats["replay_events"] += 1

    def get_worst_performing_tasks(self, n: int = 5) -> List[Tuple[str, float]]:
        """Returns the n tasks with the lowest observed matching rates."""
        rates = []
        for sid, stats in self.task_telemetry.items():
            gen = stats.get("generated_count", 0)
            matches = stats.get("observed_matches", 0)
            rate = float(matches) / float(gen) if gen > 0 else 0.0
            rates.append((sid, rate))
        rates.sort(key=lambda x: x[1])
        return rates[:n]

    def record_step(
        self,
        epoch: int,
        step: int,
        policy_entropy: float,
        reward_variance: float,
        compiler_trapped: bool,
        prefix_length: float,
        oracle_ppl: Optional[float] = None,
        active_stage: int = 1,
    ) -> DiagnosticTelemetryRecord:
        """Records a single training rollout batch and computes rolling statistics."""
        self.variance_window.append(reward_variance)
        self.zero_variance_flags.append(reward_variance < 1e-6)
        self.trap_window.append(compiler_trapped)
        self.prefix_window.append(prefix_length)
        self.entropy_history.append(policy_entropy)

        acr = float(np.mean(list(self.zero_variance_flags))) if self.zero_variance_flags else 0.0
        trap_rate = float(np.mean(list(self.trap_window))) if self.trap_window else 0.0
        avg_prefix = float(np.mean(list(self.prefix_window))) if self.prefix_window else 0.0

        rec = DiagnosticTelemetryRecord(
            epoch=epoch,
            step=step,
            policy_entropy=policy_entropy,
            reward_variance=reward_variance,
            advantage_collapse_rate=acr,
            compiler_trap_rate=trap_rate,
            avg_prefix_length=avg_prefix,
            oracle_ppl=oracle_ppl,
            active_stage=active_stage,
        )
        self.records.append(rec)
        self._check_warnings(rec)
        return rec

    def _check_warnings(self, rec: DiagnosticTelemetryRecord) -> None:
        """Checks metric thresholds for early warning signals."""
        warnings: List[str] = []

        # 1. Premature Entropy Collapse Warning
        if rec.policy_entropy < self.entropy_min:
            warnings.append(
                f"Premature Entropy Collapse: entropy={rec.policy_entropy:.4f} < {self.entropy_min}"
            )
        elif len(self.entropy_history) >= 5:
            recent_drop = (self.entropy_history[-5] - rec.policy_entropy) / max(1e-4, self.entropy_history[-5])
            if recent_drop > 0.70:
                warnings.append(
                    f"Rapid Entropy Decay: dropped {recent_drop * 100:.1f}% over last 5 steps"
                )

        # 2. Advantage Collapse Rate (ACR) Warning
        if len(self.zero_variance_flags) >= 5 and rec.advantage_collapse_rate >= self.acr_threshold:
            warnings.append(
                f"Advantage Collapse: ACR={rec.advantage_collapse_rate:.2f} >= {self.acr_threshold}"
            )

        # 3. High Compiler Trap Rate
        if len(self.trap_window) >= 10 and rec.compiler_trap_rate > self.trap_rate_max:
            warnings.append(
                f"High Compiler Trap Rate: {rec.compiler_trap_rate * 100:.1f}% > {self.trap_rate_max * 100:.1f}%"
            )

        for w in warnings:
            logger.warning(f"[Telemetry Warning - Step {rec.step}] {w}")
            if self.on_warning is not None:
                self.on_warning(w, rec)
            if self.halt_on_divergence:
                raise DivergenceHaltError(w, record=rec)

    @property
    def current_acr(self) -> float:
        return float(np.mean(list(self.zero_variance_flags))) if self.zero_variance_flags else 0.0

    @property
    def latest_record(self) -> Optional[DiagnosticTelemetryRecord]:
        return self.records[-1] if self.records else None

    def export_summary(self) -> Dict[str, float]:
        """Returns summary statistics across all recorded telemetry steps."""
        if not self.records:
            return {}
        return {
            "total_steps": len(self.records),
            "final_entropy": self.records[-1].policy_entropy,
            "mean_reward_variance": float(np.mean([r.reward_variance for r in self.records])),
            "final_acr": self.records[-1].advantage_collapse_rate,
            "final_trap_rate": self.records[-1].compiler_trap_rate,
            "final_avg_prefix": self.records[-1].avg_prefix_length,
        }

    def save_json(self, file_path: str) -> None:
        """Saves telemetry log to JSON file."""
        data = [r.to_dict() for r in self.records]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"telemetry": data, "summary": self.export_summary()}, f, indent=2)
