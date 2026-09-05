"""Asymmetric Prompt Weighting, S-GRPO Conditional Trajectory Injection (CGI), and Down-Sampled Lexicase Selection."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
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
    mean = rewards.mean()
    std = rewards.std()

    if std < eps:
        if mean < 0.0:
            # All completions in group failed: apply asymmetric penalty
            return torch.full_like(rewards, -asymmetric_penalty_weight)
        elif mean > 0.0:
            # All completions in group succeeded: apply positive advantage
            return torch.full_like(rewards, 1.0)
        else:
            return torch.zeros_like(rewards)

    advantages = (rewards - mean) / (std + eps)
    return advantages


def compute_sgrpo_advantages(
    rewards: torch.Tensor,  # shape: (group_size,) or (group_size + 1,) if reference injected
    ref_injected: bool = False,
    asymmetric_penalty_weight: float = 1.5,
    use_avspo_anchor: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes Supervised Group Relative Policy Optimization (S-GRPO) advantages with CGI & AVSPO.

    Args:
        rewards: Tensor of rollout rewards
        ref_injected: Whether a reference solution (r=+1.0) was injected at index 0
        asymmetric_penalty_weight: Multiplier applied to failed samples
        use_avspo_anchor: Whether to synthesize a virtual positive anchor if all samples fail without reference
    """
    if ref_injected:
        # Group includes injected reference trajectory
        mean = rewards.mean()
        std = rewards.std()
        advantages = (rewards - mean) / (std + eps)
        return advantages

    mean = rewards.mean()
    std = rewards.std()

    if std < eps:
        if mean <= 0.0:
            if use_avspo_anchor:
                # AVSPO: Synthesize virtual positive anchor (virtual r=+1.0)
                virtual_rewards = torch.cat([rewards, torch.tensor([1.0], device=rewards.device)])
                v_mean = virtual_rewards.mean()
                v_std = virtual_rewards.std()
                # Return normalized advantages for the real rollouts
                adv_real = (rewards - v_mean) / (v_std + eps)
                return adv_real * asymmetric_penalty_weight
            else:
                return torch.full_like(rewards, -asymmetric_penalty_weight)
        else:
            return torch.full_like(rewards, 1.0)

    advantages = (rewards - mean) / (std + eps)
    return advantages


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
    max_terms = min(len(target_terms), min((len(o) for o in candidates_outputs), default=len(target_terms)))
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
        surviving = [idx for idx in surviving if candidate_errors[idx][step_idx] <= min_err + 1.0e-6]

    return LexicaseSelectionBatch(
        prompt_id=prompt_id,
        test_case_indices=test_cases,
        candidate_errors=candidate_errors,
        surviving_candidates=surviving,
    )
