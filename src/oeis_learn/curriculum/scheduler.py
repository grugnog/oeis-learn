"""Curriculum Scheduler and Automated Stage Graduation Gating Engine."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple
from oeis_learn.curriculum.taxonomy import compute_prompt_difficulty_weight, get_stage_name
from oeis_learn.data.models import CurriculumProgress


class CurriculumScheduler:
    """Manages 5-stage taxonomic curriculum learning, rolling competence tracking,

    coverage equilibrium, and automated graduation gates.
    """

    def __init__(
        self,
        initial_stage: int = 1,
        competence_threshold: float = 0.85,
        coverage_min_threshold: float = 0.50,
        variance_threshold: float = 0.05,
        window_size: int = 20,
    ):
        self.active_stage = initial_stage
        self.competence_threshold = competence_threshold
        self.coverage_min_threshold = coverage_min_threshold
        self.variance_threshold = variance_threshold
        self.window_size = window_size

        # Prompt rolling pass histories: oeis_id -> deque of binary outcomes (True/False)
        self.prompt_histories: Dict[str, deque[bool]] = {}
        # Prompt metadata (stage, tags, etc.)
        self.prompt_metadata: Dict[str, Dict[str, Any]] = {}
        # Epoch competence history
        self.epoch_competence_history: List[float] = []
        self.graduated_stages: Set[int] = set()

    def register_prompt(self, oeis_id: str, stage: int, tags: Optional[List[str]] = None, term_count: int = 20) -> None:
        """Register a sequence prompt under a specific curriculum stage."""
        if oeis_id not in self.prompt_histories:
            self.prompt_histories[oeis_id] = deque(maxlen=self.window_size)
            self.prompt_metadata[oeis_id] = {
                "stage": stage,
                "tags": tags or [],
                "term_count": term_count,
            }

    def record_outcome(self, oeis_id: str, success: bool) -> None:
        """Record evaluation outcome for a prompt."""
        if oeis_id not in self.prompt_histories:
            self.register_prompt(oeis_id, stage=self.active_stage)
        self.prompt_histories[oeis_id].append(bool(success))

    def get_prompt_pass_rate(self, oeis_id: str) -> float:
        """Get rolling pass-rate rho_hat for a given prompt."""
        history = self.prompt_histories.get(oeis_id)
        if not history:
            return 0.0
        return sum(1.0 for x in history if x) / len(history)

    def compute_stage_competence(self, stage: Optional[int] = None) -> Tuple[float, float]:
        """Computes weighted stage competence C(S_k) and minimum coverage min(rho_hat).

        Returns:
            Tuple of (C(S_k), min_pass_rate)
        """
        target_stage = stage if stage is not None else self.active_stage
        prompts = [
            pid for pid, meta in self.prompt_metadata.items() if meta["stage"] == target_stage
        ]

        if not prompts:
            return 0.0, 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        pass_rates = []

        for pid in prompts:
            meta = self.prompt_metadata[pid]
            w = compute_prompt_difficulty_weight(meta["tags"], target_stage, meta["term_count"])
            rho = self.get_prompt_pass_rate(pid)
            pass_rates.append(rho)
            weighted_sum += w * rho
            total_weight += w

        competence = weighted_sum / total_weight if total_weight > 0 else 0.0
        min_coverage = min(pass_rates) if pass_rates else 0.0

        return float(competence), float(min_coverage)

    def get_competence_score(self, stage: Optional[int] = None) -> float:
        """Convenience method returning weighted stage competence score C(S_k)."""
        return self.compute_stage_competence(stage)[0]

    def compute_epoch_variance(self, window: int = 5) -> float:
        """Computes variance of recent epoch competence scores."""
        if len(self.epoch_competence_history) < 2:
            return 0.0
        recent = self.epoch_competence_history[-window:]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return float(variance)

    def check_and_update_graduation(self) -> Tuple[bool, Optional[int]]:
        """Evaluates whether current active stage qualifies for graduation.

        Graduation Criteria:
        1. C(S_k) >= competence_threshold (0.85)
        2. min(rho_hat) >= coverage_min_threshold (0.50)
        3. Variance[C_e] <= variance_threshold (0.05)

        Returns:
            Tuple of (graduated_bool, new_stage_int_or_none)
        """
        c_k, min_cov = self.compute_stage_competence(self.active_stage)
        self.epoch_competence_history.append(c_k)
        var_e = self.compute_epoch_variance()

        if (
            c_k >= self.competence_threshold
            and min_cov >= self.coverage_min_threshold
            and var_e <= self.variance_threshold
            and self.active_stage < 5
        ):
            self.graduated_stages.add(self.active_stage)
            self.active_stage += 1
            return True, self.active_stage

        return False, None

    def get_progress(self) -> CurriculumProgress:
        """Returns snapshot of current curriculum progress state."""
        c_k, min_cov = self.compute_stage_competence(self.active_stage)
        pass_rates = {
            pid: self.get_prompt_pass_rate(pid)
            for pid, meta in self.prompt_metadata.items()
            if meta["stage"] == self.active_stage
        }
        return CurriculumProgress(
            active_stage=self.active_stage,
            rolling_pass_rates=pass_rates,
            stage_competence=c_k,
            coverage_min=min_cov,
            epoch_variance=self.compute_epoch_variance(),
            graduated_stages=sorted(list(self.graduated_stages)),
        )
