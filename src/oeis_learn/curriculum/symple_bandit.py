"""Cost-aware visit scheduler with explicit conditional sampling probabilities.

The old public class name is retained; this is deliberately not an EXP3 claim.
"""

from copy import deepcopy
import math
import random


class Exp3SBanditScheduler:
    def __init__(
        self,
        sequence_ids,
        gamma=0.25,
        alpha=0.05,
        competence_window=20,
        gamma_floor=0.25,
        seed=20260913,
    ):
        if len(set(sequence_ids)) != len(sequence_ids) or not sequence_ids:
            raise ValueError("nonempty unique task identities required")
        if not 0.25 <= max(gamma, gamma_floor) <= 1 or competence_window < 2:
            raise ValueError("at least 25% uniform exploration required")
        self.sequence_ids = list(sequence_ids)
        self.K = len(sequence_ids)
        self.gamma = max(gamma, gamma_floor)
        self.gamma_floor = gamma_floor
        self.alpha, self.window_size = alpha, competence_window
        self.weights = {sid: 1.0 for sid in sequence_ids}
        self.histories = {sid: [] for sid in sequence_ids}
        self.last_visited = {sid: 0 for sid in sequence_ids}
        self.last_selection = {}
        self.rng = random.Random(seed)

    def recalibrate_uniform(self, gamma_floor=0.25):
        if not 0.25 <= gamma_floor <= 1:
            raise ValueError("exploration floor out of bounds")
        self.gamma_floor = gamma_floor
        self.gamma = max(self.gamma, gamma_floor)
        self.weights = {sid: 1.0 for sid in self.sequence_ids}

    def get_competence(self, oeis_id):
        history = self.histories.get(oeis_id, [])
        total = sum(v["attempts"] for v in history)
        return sum(v["successes"] for v in history) / total if total else 0.0

    def get_competence_slope(self, oeis_id):
        history = self.histories.get(oeis_id, [])
        if len(history) < 2:
            return 0.0

        def mean(visits):
            return sum(v["successes"] for v in visits) / sum(v["attempts"] for v in visits)

        mid = len(history) // 2
        return mean(history[mid:]) - mean(history[:mid])

    def get_selection_probabilities(self, remaining=None):
        remaining = self.sequence_ids if remaining is None else list(remaining)
        if not remaining:
            return {}
        total = sum(self.weights[sid] for sid in remaining)
        return {
            sid: (1 - self.gamma) * self.weights[sid] / total + self.gamma / len(remaining)
            for sid in remaining
        }

    def sample_active_prompts(self, batch_size=4):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("positive active prompt count required")
        remaining, chosen = list(self.sequence_ids), []
        self.last_selection = {}
        for _ in range(min(batch_size, self.K)):
            probabilities = self.get_selection_probabilities(remaining)
            pick = self.rng.choices(remaining, [probabilities[sid] for sid in remaining])[0]
            self.last_selection[pick] = {
                "probability": probabilities[pick],
                "draw": len(chosen),
                "remaining": tuple(remaining),
                "rule": "conditional-without-replacement/v1",
            }
            chosen.append(pick)
            remaining.remove(pick)
        return chosen

    def update_feedback(
        self,
        oeis_id,
        success_count,
        group_size,
        current_step=0,
        elapsed_s=None,
        selection_probability=None,
    ):
        if (
            oeis_id not in self.histories
            or type(group_size) is not int
            or group_size <= 0
            or type(success_count) is not int
            or not 0 <= success_count <= group_size
            or elapsed_s is None
            or not math.isfinite(elapsed_s)
            or elapsed_s <= 0
        ):
            raise ValueError("valid aggregate visit and measured positive cost required")
        probability = selection_probability
        if probability is None:
            probability = self.last_selection.get(oeis_id, {}).get(
                "probability", self.get_selection_probabilities()[oeis_id]
            )
        if not 0 < probability <= 1:
            raise ValueError("invalid recorded conditional probability")
        visit = {
            "successes": success_count,
            "attempts": group_size,
            "elapsed_s": elapsed_s,
            "step": current_step,
            "selection_probability": probability,
        }
        self.histories[oeis_id].append(visit)
        self.histories[oeis_id] = self.histories[oeis_id][-self.window_size :]
        self.last_visited[oeis_id] = current_step
        self.weights[oeis_id] = max(1e-4, self.priority(oeis_id))

    def priority(self, sid):
        visits = self.histories[sid]
        attempts = sum(v["attempts"] for v in visits)
        # A planning uncertainty score, not independent-trial confidence.
        uncertainty = 1 / math.sqrt(len(visits) + 1)
        cost = sum(v["elapsed_s"] for v in visits) / max(1, attempts) if visits else 1.0
        competence = self.get_competence(sid)
        return (
            uncertainty + 4 * competence * (1 - competence) + abs(self.get_competence_slope(sid))
        ) / max(cost, 1e-6)

    def state_dict(self):
        return deepcopy(
            {
                "version": "visit-scheduler/v1",
                "sequence_ids": self.sequence_ids,
                "gamma": self.gamma,
                "gamma_floor": self.gamma_floor,
                "alpha": self.alpha,
                "window_size": self.window_size,
                "weights": self.weights,
                "histories": self.histories,
                "last_visited": self.last_visited,
                "last_selection": self.last_selection,
                "rng": self.rng.getstate(),
            }
        )

    def load_state_dict(self, state):
        expected = set(self.state_dict())
        if (
            set(state) != expected
            or state["version"] != "visit-scheduler/v1"
            or state["sequence_ids"] != self.sequence_ids
        ):
            raise ValueError("incompatible curriculum state")
        if (
            not 0.25 <= state["gamma_floor"] <= state["gamma"] <= 1
            or type(state["window_size"]) is not int
            or state["window_size"] < 2
        ):
            raise ValueError("invalid scheduler settings")
        for key in ("weights", "histories", "last_visited"):
            if set(state[key]) != set(self.sequence_ids):
                raise ValueError("missing scheduler task state")
        if any(not math.isfinite(weight) or weight <= 0 for weight in state["weights"].values()):
            raise ValueError("invalid scheduler weights")
        for history in state["histories"].values():
            if len(history) > state["window_size"]:
                raise ValueError("history exceeds window")
            for visit in history:
                if (
                    set(visit)
                    != {"successes", "attempts", "elapsed_s", "step", "selection_probability"}
                    or type(visit["attempts"]) is not int
                    or type(visit["successes"]) is not int
                    or not 0 <= visit["successes"] <= visit["attempts"]
                    or visit["attempts"] <= 0
                    or not math.isfinite(visit["elapsed_s"])
                    or visit["elapsed_s"] <= 0
                    or not 0 < visit["selection_probability"] <= 1
                ):
                    raise ValueError("invalid aggregate visit state")
        rng = random.Random()
        rng.setstate(state["rng"])
        for key in (
            "gamma",
            "gamma_floor",
            "alpha",
            "window_size",
            "weights",
            "histories",
            "last_visited",
            "last_selection",
        ):
            setattr(self, key, deepcopy(state[key]))
        self.rng = rng


class AdaGGroupAllocator:
    def __init__(self, total_budget=32, min_g=2, max_g=16, p_target=0.50, p_floor=0.02):
        if not 1 <= min_g <= max_g <= 16 or total_budget <= 0:
            raise ValueError("invalid allocation bounds")
        self.total_budget, self.min_g, self.max_g = total_budget, min_g, max_g
        self.unallocated = 0

    def compute_group_sizes(self, prompts, bandit, total_budget=None):
        budget = self.total_budget if total_budget is None else total_budget
        if type(budget) is not int or budget < 0 or len(set(prompts)) != len(prompts):
            raise ValueError("exact budget and unique prompts required")
        if budget < len(prompts) * self.min_g:
            raise ValueError("budget cannot satisfy per-prompt minimum")
        sizes = {pid: self.min_g for pid in sorted(prompts)}
        target = min(budget, len(prompts) * self.max_g)
        while sum(sizes.values()) < target:
            candidates = [pid for pid in sizes if sizes[pid] < self.max_g]
            pick = min(candidates, key=lambda pid: (-bandit.priority(pid) / (sizes[pid] + 1), pid))
            sizes[pick] += 1
        self.unallocated = budget - target
        return sizes
