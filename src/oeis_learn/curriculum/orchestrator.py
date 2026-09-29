"""Adaptive curriculum orchestrator coordinating EXP3.S, Ada-G, replay, and training events."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Sequence
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.curriculum.symple_bandit import AdaGGroupAllocator, Exp3SBanditScheduler
from oeis_learn.data.models import SequenceRecord
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer


class CurriculumOrchestrator:
    """Coordinates task selection, dynamic group sizing, trainer execution, feedback, and dormancy replay."""

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        bandit: Optional[Exp3SBanditScheduler] = None,
        allocator: Optional[AdaGGroupAllocator] = None,
        elite_buffer: Optional[EliteSeedDemonstrationBuffer] = None,
        scheduler: Optional[CurriculumScheduler] = None,
        active_batch_size: int = 2,
        rollout_budget: int = 32,
        replay_batch_size: int = 2,
    ):
        self.records = list(records)
        self.records_by_id = {r.oeis_id: r for r in self.records}
        self.sequence_ids = list(self.records_by_id.keys())

        self.bandit = bandit or Exp3SBanditScheduler(sequence_ids=self.sequence_ids)
        self.allocator = allocator or AdaGGroupAllocator(total_budget=rollout_budget)
        self.elite_buffer = elite_buffer or EliteSeedDemonstrationBuffer()
        self.scheduler = scheduler or CurriculumScheduler(initial_stage=1)

        self.active_batch_size = active_batch_size
        self.rollout_budget = rollout_budget
        self.replay_batch_size = replay_batch_size

        self.events: List[Dict[str, Any]] = []

    def _record_event(self, event_type: str, sequence_id: str, step: int, data: Dict[str, Any]) -> None:
        """Appends an immutable curriculum event to the trace."""
        self.events.append({
            "event_type": event_type,
            "sequence_id": sequence_id,
            "step": step,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **data,
        })

    def select_active_prompts(self, batch_size: Optional[int] = None, current_step: int = 0) -> List[str]:
        """Samples active task prompts from the non-stationary bandit distribution."""
        b_size = batch_size if batch_size is not None else self.active_batch_size
        prompts = self.bandit.sample_active_prompts(batch_size=b_size)
        probs = self.bandit.get_selection_probabilities()

        for pid in prompts:
            self._record_event(
                event_type="TASK_SELECTED",
                sequence_id=pid,
                step=current_step,
                data={
                    "selection_probability": probs.get(pid, 0.0),
                    "competence": self.bandit.get_competence(pid),
                },
            )
        return prompts

    def allocate_rollouts(
        self,
        prompts: Sequence[str],
        total_budget: Optional[int] = None,
        current_step: int = 0,
    ) -> Dict[str, int]:
        """Allocates dynamic rollout compute G_i for each active prompt."""
        budget = total_budget if total_budget is not None else self.rollout_budget
        sizes = self.allocator.compute_group_sizes(prompts=prompts, bandit=self.bandit, total_budget=budget)

        for pid, sz in sizes.items():
            self._record_event(
                event_type="ROLLOUT_ALLOCATED",
                sequence_id=pid,
                step=current_step,
                data={"allocated_rollouts": sz},
            )
        return sizes

    def execute_step(
        self,
        trainer: Any,
        current_step: int,
        epoch: int = 1,
    ) -> Dict[str, Any]:
        """Runs one full orchestrated adaptive step: active sampling -> train -> feedback -> replay."""
        prompts = self.select_active_prompts(current_step=current_step)
        allocations = self.allocate_rollouts(prompts, current_step=current_step)

        step_outcomes = []
        for pid in prompts:
            rec = self.records_by_id.get(pid)
            if not rec:
                continue

            g_size = allocations.get(pid, 8)
            # Record active visitation in elite buffer
            self.elite_buffer.record_active_visit(pid, current_step)

            metrics = trainer.train_step_for_prompt(rec, epoch=epoch, group_size=g_size)
            pass_count = metrics.get("pass_count", 0)

            # Update bandit feedback
            self.bandit.update_feedback(
                oeis_id=pid,
                success_count=pass_count,
                group_size=g_size,
                current_step=current_step,
            )

            self._record_event(
                event_type="BANDIT_FEEDBACK_APPLIED",
                sequence_id=pid,
                step=current_step,
                data={
                    "pass_count": pass_count,
                    "group_size": g_size,
                    "updated_competence": self.bandit.get_competence(pid),
                },
            )
            step_outcomes.append(metrics)

        # Sample dormant replay sequences
        replay_items = self.elite_buffer.sample_dormancy_vulnerable_batch(
            batch_size=self.replay_batch_size,
            current_step=current_step,
        )
        for r_id, _ in replay_items:
            self._record_event(
                event_type="REPLAY_SELECTED",
                sequence_id=r_id,
                step=current_step,
                data={"is_dormancy_replay": True},
            )

        return {
            "step": current_step,
            "active_prompts": prompts,
            "allocations": allocations,
            "replay_count": len(replay_items),
            "step_metrics": step_outcomes,
        }
