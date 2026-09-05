"""SYMPLE Multi-Task Engine: EXP3.S Non-Stationary Bandit Scheduler & Ada-G Dynamic Group Allocator."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence
import numpy as np


class Exp3SBanditScheduler:
    """EXP3.S adversarial non-stationary multi-armed bandit task scheduler

    with binomial dispersion learning progress feedback in the Zone of Proximal Development.
    """

    def __init__(
        self,
        sequence_ids: Sequence[str],
        gamma: float = 0.15,
        alpha: float = 0.05,
        competence_window: int = 20,
    ):
        self.sequence_ids = list(sequence_ids)
        self.K = max(1, len(self.sequence_ids))
        self.gamma = gamma
        self.alpha = alpha
        self.window_size = competence_window

        # Weights initialized to 1.0
        self.weights: Dict[str, float] = {sid: 1.0 for sid in self.sequence_ids}
        self.histories: Dict[str, List[int]] = {sid: [] for sid in self.sequence_ids}
        self.last_visited: Dict[str, int] = {sid: 0 for sid in self.sequence_ids}

    def get_competence(self, oeis_id: str) -> float:
        """Computes rolling pass competence p_hat in [0.0, 1.0]."""
        hist = self.histories.get(oeis_id, [])
        if not hist:
            return 0.0
        return float(np.mean(hist))

    def get_competence_slope(self, oeis_id: str) -> float:
        """Computes competence velocity Delta C_i across early and late history."""
        hist = self.histories.get(oeis_id, [])
        if len(hist) < 4:
            return 0.0
        mid = len(hist) // 2
        early = hist[:mid]
        late = hist[mid:]
        return float(np.mean(late) - np.mean(early))

    def get_selection_probabilities(self) -> Dict[str, float]:
        """Calculates current probability distribution over the K task arms."""
        total_w = sum(self.weights.values())
        if total_w <= 0.0:
            total_w = float(self.K)
            self.weights = {sid: 1.0 for sid in self.sequence_ids}

        probs = {}
        for sid in self.sequence_ids:
            w_i = self.weights.get(sid, 1.0)
            p_i = (1.0 - self.gamma) * (w_i / total_w) + (self.gamma / self.K)
            probs[sid] = float(p_i)
        return probs

    def sample_active_prompts(self, batch_size: int = 2) -> List[str]:
        """Samples B_active frontier task prompts without replacement."""
        if self.K <= batch_size:
            return list(self.sequence_ids)

        probs_dict = self.get_selection_probabilities()
        sids = list(probs_dict.keys())
        p_vals = np.array([probs_dict[sid] for sid in sids], dtype=np.float64)
        p_vals /= np.sum(p_vals)

        chosen = np.random.choice(sids, size=batch_size, replace=False, p=p_vals)
        return [str(c) for c in chosen]

    def update_feedback(
        self,
        oeis_id: str,
        success_count: int,
        group_size: int,
        current_step: int = 0,
    ) -> None:
        """Updates rolling pass history and EXP3.S weights given task rollout outcomes."""
        if oeis_id not in self.weights:
            self.sequence_ids.append(oeis_id)
            self.K = len(self.sequence_ids)
            self.weights[oeis_id] = 1.0
            self.histories[oeis_id] = []

        self.last_visited[oeis_id] = current_step

        # Update sliding history window
        outcomes = [1] * success_count + [0] * max(0, group_size - success_count)
        self.histories[oeis_id].extend(outcomes)
        if len(self.histories[oeis_id]) > self.window_size:
            self.histories[oeis_id] = self.histories[oeis_id][-self.window_size :]

        # Compute learning progress feedback r_{i,t} in Zone of Proximal Development
        p_hat = self.get_competence(oeis_id)
        delta_c = self.get_competence_slope(oeis_id)

        # Binomial dispersion (peaks at p=0.5) + score velocity + alarm for decay
        binomial_dispersion = p_hat * (1.0 - p_hat) * 4.0  # Normalized to [0, 1]
        decay_alarm = 2.0 * max(0.0, -delta_c)
        r_feedback = binomial_dispersion + abs(delta_c) + decay_alarm

        # Importance weight estimate: r_hat = r / p_i
        probs = self.get_selection_probabilities()
        p_i = max(1e-6, probs.get(oeis_id, 1.0 / self.K))
        r_hat = min(10.0, r_feedback / p_i)

        # EXP3.S weight update
        w_old = self.weights[oeis_id]
        w_new = w_old * math.exp(self.gamma * r_hat / self.K)
        self.weights[oeis_id] = min(1e6, max(1e-4, w_new))

        # Uniform mixing across all arms
        total_w = sum(self.weights.values())
        mixing = (math.e * self.alpha / self.K) * total_w
        for sid in self.sequence_ids:
            self.weights[sid] += mixing


class AdaGGroupAllocator:
    """Adaptive Dynamic Group Sizing (Ada-G) allocating rollout compute G_i in [min_g, max_g]

    to guarantee P(Hit >= 1) >= p_target on frontier tasks.
    """

    def __init__(
        self,
        total_budget: int = 32,
        min_g: int = 8,
        max_g: int = 16,
        p_target: float = 0.50,
        p_floor: float = 0.02,
    ):
        self.total_budget = total_budget
        self.min_g = min_g
        self.max_g = max_g
        self.p_target = p_target
        self.p_floor = p_floor

    def compute_group_sizes(
        self,
        prompts: Sequence[str],
        bandit: Exp3SBanditScheduler,
        total_budget: Optional[int] = None,
    ) -> Dict[str, int]:
        """Computes allocated rollout count G_i for each active prompt, filling the budget."""
        budget = total_budget if total_budget is not None else self.total_budget
        if not prompts:
            return {}

        raw_sizes: Dict[str, int] = {}

        for pid in sorted(prompts):
            p_hat = bandit.get_competence(pid)
            effective_p = max(p_hat, self.p_floor)

            # G_i = ceil( ln(1 - p_target) / ln(1 - max(p_hat, p_floor)) )
            log_num = math.log(1.0 - self.p_target)
            log_den = math.log(1.0 - effective_p)
            g_ideal = math.ceil(log_num / log_den)
            g_clipped = min(self.max_g, max(self.min_g, g_ideal))
            raw_sizes[pid] = g_clipped

        current_total = sum(raw_sizes.values())

        # If below budget, allocate surplus up to max_g starting with lowest competence (deterministic tie-breaking)
        if current_total < budget:
            sorted_by_need = sorted(
                prompts,
                key=lambda p: (bandit.get_competence(p), p),
            )
            while current_total < budget:
                allocated_any = False
                for pid in sorted_by_need:
                    if raw_sizes[pid] < self.max_g and current_total < budget:
                        raw_sizes[pid] += 1
                        current_total += 1
                        allocated_any = True
                if not allocated_any:
                    break

        # If above budget, scale down while respecting min_g
        elif current_total > budget:
            sorted_by_ease = sorted(
                prompts,
                key=lambda p: (-bandit.get_competence(p), p),
            )
            while current_total > budget:
                reduced_any = False
                for pid in sorted_by_ease:
                    if raw_sizes[pid] > self.min_g and current_total > budget:
                        raw_sizes[pid] -= 1
                        current_total -= 1
                        reduced_any = True
                if not reduced_any:
                    break

        return raw_sizes
