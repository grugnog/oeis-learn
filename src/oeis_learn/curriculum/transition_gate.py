"""Transition Gate Evaluator for Strategy C Progressive SFT Warmstart Transfer.

Enforces landmark pre-RL qualification metrics:
- Advantage Collapse Rate (ACR) <= 15%
- Grammar Validity >= 98.5%
- Candidate Pass Rate >= 75%
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from oeis_learn.data.models import SyntheticDemonstrationPair
from oeis_learn.sandbox.runner import WasmRunner

logger = logging.getLogger(__name__)


class TransitionGateEvaluator:
    """Evaluates whether an SFT bridge model satisfies criteria to transition to GRPO."""

    def __init__(
        self,
        max_advantage_collapse_rate: float = 0.15,
        min_grammar_validity: float = 0.985,
        min_candidate_pass_rate: float = 0.75,
        runner: Optional[WasmRunner] = None,
    ):
        self.max_acr = max_advantage_collapse_rate
        self.min_grammar = min_grammar_validity
        self.min_pass = min_candidate_pass_rate
        self.runner = runner or WasmRunner(fuel_budget=10000)

    def evaluate_rollouts(
        self,
        rollout_groups: List[List[str]],
        target_sequences: List[List[int]],
    ) -> Dict[str, Any]:
        """Evaluates a collection of generated rollout groups against ground-truth sequences.

        Args:
            rollout_groups: List of groups, each group contains candidate WAT strings
            target_sequences: List of ground-truth term lists corresponding to each prompt group
        """
        total_groups = len(rollout_groups)
        if total_groups == 0:
            return {
                "advantage_collapse_rate": 1.0,
                "grammar_validity": 0.0,
                "candidate_pass_rate": 0.0,
                "gate_passed": False,
                "reason": "Empty rollout groups",
            }

        total_candidates = 0
        valid_grammar_count = 0
        passed_candidate_count = 0
        collapsed_group_count = 0

        for group, target_terms in zip(rollout_groups, target_sequences):
            group_successes = 0
            for cand_wat in group:
                total_candidates += 1
                res = self.runner.run_single(
                    cand_wat,
                    terms_to_generate=min(20, len(target_terms)),
                    result_profile="i256x4_v1",
                )
                if res.status != "PARSE_ERROR" and res.status != "COMPILE_ERROR":
                    valid_grammar_count += 1

                if res.status == "SUCCESS" and res.output == target_terms[: len(res.output)]:
                    passed_candidate_count += 1
                    group_successes += 1

            # In GRPO, if group_successes == 0, every rollout has reward 0 -> zero advantage variance
            if group_successes == 0:
                collapsed_group_count += 1

        acr = collapsed_group_count / total_groups
        grammar_val = valid_grammar_count / max(1, total_candidates)
        pass_rate = passed_candidate_count / max(1, total_candidates)

        gate_passed = (
            acr <= self.max_acr
            and grammar_val >= self.min_grammar
            and pass_rate >= self.min_pass
        )

        return {
            "advantage_collapse_rate": acr,
            "grammar_validity": grammar_val,
            "candidate_pass_rate": pass_rate,
            "gate_passed": gate_passed,
            "total_candidates": total_candidates,
            "total_groups": total_groups,
            "collapsed_groups": collapsed_group_count,
        }
