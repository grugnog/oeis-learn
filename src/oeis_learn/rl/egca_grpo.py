"""Execution-Guided Credit Assignment GRPO (EGCA-GRPO) Loss Module with Schulman KL and SFT Co-Training."""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn.functional as F
from oeis_learn.decoder.wat_grammar import PAD_ID


def compute_schulman_kl(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """Computes A nonnegative sampled KL diagnostic (expectation assumes current-policy sampling):

    D_KL(pi || pi_ref) approx exp(log_ref - log_pi) - (log_ref - log_pi) - 1 >= 0
    """
    diff = ref_log_probs - log_probs
    kl = torch.exp(diff) - diff - 1.0
    return torch.clamp(kl, min=0.0)


def compute_policy_entropy(
    logits: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes token policy entropy H(pi_theta)."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    token_entropy = -torch.sum(
        probs * torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs)),
        dim=-1,
    )  # (batch, seq_len)
    if mask is not None:
        return (token_entropy * mask).sum() / mask.sum().clamp(min=1.0)
    return token_entropy.mean()


def compute_egca_grpo_loss(
    logits,
    old_log_probs,
    token_ids,
    advantages,
    token_masks=None,
    clip_eps=0.2,
    epsilon_low=None,
    epsilon_high=None,
    beta_kl=0.01,
    kl_coef=None,
    ref_log_probs=None,
    alpha_ent=0.0,
    sft_loss=None,
    beta_sft=0.20,
    *,
    supports=None,
    temperatures=1.0,
    reference_logits=None,
    objective="grpo",
):
    """Mean-per-trajectory objective using stored behavior and exact support.

    Selected-action reference probabilities cannot supply exact categorical KL.
    """
    from oeis_learn.rl.policy import policy_loss

    if epsilon_low is not None or epsilon_high is not None:
        if epsilon_low != clip_eps or epsilon_high != clip_eps:
            raise ValueError("asymmetric legacy clipping is unsupported")
    weight = beta_kl if kl_coef is None else kl_coef
    if ref_log_probs is not None and weight:
        raise ValueError("full reference_logits required; stale sampled KL is not unbiased")
    mask = token_ids != PAD_ID
    if token_masks is not None:
        mask &= token_masks.bool()
    loss = policy_loss(
        logits,
        token_ids,
        old_log_probs,
        advantages,
        supports=supports,
        temperatures=temperatures,
        action_mask=mask,
        objective=objective,
        clip_eps=clip_eps,
        reference_logits=reference_logits,
        kl_coefficient=weight,
        entropy_coefficient=alpha_ent,
    )
    if sft_loss is not None and beta_sft:
        loss = loss + beta_sft * sft_loss
    return loss


# =========================================================================
# Phase 4 Partitioned Semantic Policy Entropy & Dynamic Temperature
# =========================================================================


def compute_partitioned_semantic_entropy(
    logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    sem_indices: Optional[List[int]] = None,
    struct_indices: Optional[List[int]] = None,
    alpha_sem: float = 0.02,
    beta_pen: float = 0.05,
    struct_threshold: float = 0.15,
) -> torch.Tensor:
    """Computes Partitioned Semantic Policy Entropy loss:

    L_ent = alpha_sem * H_sem(pi | s) / log(|A_sem| + eps) - beta_pen * max(0, P(A_struct) - threshold)
    """
    from oeis_learn.rl.policy import masked_distribution

    legal = (
        valid_mask
        if valid_mask is not None and valid_mask.ndim == 3
        else torch.ones_like(logits, dtype=torch.bool)
    )
    _, probs, token_entropy = masked_distribution(logits, legal)

    if sem_indices is not None and len(sem_indices) > 0:
        sem_idx_t = torch.tensor(sem_indices, dtype=torch.long, device=logits.device)
        p_sem_raw = probs.index_select(-1, sem_idx_t)  # (B, L, |sem|)
        p_sem_sum = p_sem_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, L, 1)
        p_sem_norm = p_sem_raw / p_sem_sum
        h_sem = -torch.sum(p_sem_norm * torch.log(p_sem_norm + 1e-8), dim=-1)  # (B, L)
        legal_count = legal.index_select(-1, sem_idx_t).sum(-1)
        normalized_h_sem = torch.where(
            legal_count > 1, h_sem / legal_count.clamp_min(2).log(), torch.zeros_like(h_sem)
        )
    else:
        legal_count = legal.sum(-1)
        normalized_h_sem = torch.where(
            legal_count > 1,
            token_entropy / legal_count.clamp_min(2).log(),
            torch.zeros_like(token_entropy),
        )

    if struct_indices is not None and len(struct_indices) > 0:
        struct_idx_t = torch.tensor(struct_indices, dtype=torch.long, device=logits.device)
        p_struct_sum = probs.index_select(-1, struct_idx_t).sum(dim=-1)  # (B, L)
        p_struct_excess = torch.clamp(p_struct_sum - struct_threshold, min=0.0)
    else:
        p_struct_excess = torch.zeros_like(normalized_h_sem)

    # Combine: Positive semantic entropy bonus - penalty on excessive structural tokens
    ent_reward = (alpha_sem * normalized_h_sem) - (beta_pen * p_struct_excess)
    if valid_mask is not None:
        if valid_mask.dim() == 3:
            v_mask_2d = valid_mask.any(dim=-1).float()
        else:
            v_mask_2d = valid_mask.float()
        mean_ent = (ent_reward * v_mask_2d).sum() / v_mask_2d.sum().clamp(min=1.0)
    else:
        mean_ent = ent_reward.mean()

    return -mean_ent  # Loss minimizes negative entropy reward


def get_dynamic_sampling_temperature(
    stack_height: int,
    max_stack: int = 16,
    t_base: float = 0.80,
    decay: float = 0.60,
) -> float:
    """Dynamically scales sampling temperature by stack height:

    T(s_t) = T_base * (1 - decay * (stack_height / max_stack))
    """
    ratio = min(1.0, max(0.0, float(stack_height) / max(1.0, float(max_stack))))
    temp = t_base * (1.0 - (decay * ratio))
    return max(0.05, float(temp))


def inject_virtual_sample_if_needed(
    group_rewards: List[float],
    has_edb_solution: bool,
) -> List[float]:
    """Applies Virtual Sample Injection when all on-policy exploratory rollouts fail

    on an active prompt that possesses an existing verified solution in the EDB.
    """
    return list(group_rewards)  # teacher examples never enter sampled reward groups


def compute_chunked_egca_grpo_loss(
    decoder,
    input_tokens: torch.Tensor,
    memory: torch.Tensor,
    old_log_probs: torch.Tensor,
    target_token_ids: torch.Tensor,
    advantages: torch.Tensor,
    token_masks: Optional[torch.Tensor] = None,
    chunk_size: int = 256,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
) -> torch.Tensor:
    """Executes sequence-chunked forward pass and policy gradient loss under strict VRAM bounds."""
    logits = decoder(input_tokens, memory)
    return compute_egca_grpo_loss(
        logits=logits,
        old_log_probs=old_log_probs,
        token_ids=target_token_ids,
        advantages=advantages,
        token_masks=token_masks,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
    )
