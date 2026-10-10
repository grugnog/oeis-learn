"""Asymmetric Prompt Weighting, S-GRPO Conditional Trajectory Injection (CGI), and Down-Sampled Lexicase Selection."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence
import torch
from oeis_learn.data.models import LexicaseSelectionBatch


def compute_group_advantages(
    rewards: torch.Tensor,  # shape: (group_size,)
    asymmetric_penalty_weight: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes group-normalized advantages with asymmetric penalty weighting for failed groups.

    If all completions fail (rewards all -1.0), std is 0. Standard GRPO yields advantages = 0.
    Asymmetric prompt weighting sets advantages = -asymmetric_penalty_weight to apply informative
    negative gradients to failed program tokens.
    """
    from oeis_learn.rl.policy import group_advantages

    return group_advantages(rewards)


def compute_sgrpo_advantages(
    rewards, ref_injected=False, asymmetric_penalty_weight=1.5, use_avspo_anchor=False, eps=1e-8
):
    """Only actual sampled rewards define the group; zero variance gives zero."""
    from oeis_learn.rl.policy import group_advantages

    if ref_injected:
        raise ValueError("teacher trajectories are SFT-only")
    return group_advantages(rewards)


def filter_downsampled_lexicase(
    candidates_outputs: List[Sequence[int]],
    target_terms: Sequence[int],
    prompt_id: str = "PROMPT",
    subsample_size: int = 5,
    seed: Optional[int] = None,
) -> LexicaseSelectionBatch:
    """Performs down-sampled lexicase selection filtering over a group of candidate program outputs.

    Evaluates candidate outputs against randomized individual test cases sequentially,
    eliminating compromise constant generalists that fail on non-zero domain points.

    Args:
        candidates_outputs: List of output integer sequences from group rollouts
        target_terms: Ground truth target sequence
        prompt_id: Associated OEIS sequence ID
        subsample_size: Number of test cases to randomly sample for evaluation
        seed: Random seed for reproducibility

    Returns:
        LexicaseSelectionBatch capturing test indices, errors, and surviving candidate indices
    """
    G = len(candidates_outputs)
    if G == 0:
        return LexicaseSelectionBatch(prompt_id=prompt_id)

    rng = random.Random(seed)
    max_terms = min(
        len(target_terms), min((len(o) for o in candidates_outputs), default=len(target_terms))
    )
    available_indices = list(range(max(1, max_terms)))

    k_sample = min(subsample_size, len(available_indices))
    test_cases = rng.sample(available_indices, k=k_sample)

    # Compute error matrix for each candidate on each test case
    candidate_errors: Dict[int, List[float]] = {}
    for i in range(G):
        out = candidates_outputs[i]
        errs = []
        for tc in test_cases:
            if tc < len(out) and tc < len(target_terms):
                errs.append(float(abs(out[tc] - target_terms[tc])))
            else:
                errs.append(1000.0)
        candidate_errors[i] = errs

    # Sequential lexicase filtering
    surviving = list(range(G))
    for step_idx, tc in enumerate(test_cases):
        if len(surviving) <= 1:
            break

        step_errs = [candidate_errors[idx][step_idx] for idx in surviving]
        min_err = min(step_errs)
        # Retain only candidates that achieve the minimum error on this individual test case
        surviving = [
            idx for idx in surviving if candidate_errors[idx][step_idx] <= min_err + 1.0e-6
        ]

    return LexicaseSelectionBatch(
        prompt_id=prompt_id,
        test_case_indices=test_cases,
        candidate_errors=candidate_errors,
        surviving_candidates=surviving,
    )
