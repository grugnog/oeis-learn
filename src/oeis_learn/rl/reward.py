"""Reward functions, composite dense-to-sparse shaping, non-triviality gating, and potential-based reward shaping."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np
from oeis_learn.data.models import CompositeRewardBreakdown, ExecutionResult, NonTrivialityEvaluation, PotentialState


def compute_empirical_variance(output: Sequence[int]) -> float:
    """Computes empirical variance Var_n[P(n)] across evaluated sequence terms."""
    if not output or len(output) < 2:
        return 0.0
    arr = np.array(output, dtype=np.float64)
    var = float(np.var(arr))
    return var


def compute_input_sensitivity(output: Sequence[int]) -> float:
    """Computes empirical input sensitivity S_input(P) = sum |P(n+1) - P(n)|."""
    if not output or len(output) < 2:
        return 0.0
    arr = np.array(output, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    return float(np.sum(diffs))


def compute_second_difference_energy(output: Sequence[int]) -> float:
    """Computes empirical second difference energy sum |Delta^2 P(n)| to measure curvature."""
    if not output or len(output) < 3:
        return 0.0
    arr = np.array(output, dtype=np.float64)
    d2 = np.abs(np.diff(np.diff(arr)))
    return float(np.sum(d2))


def evaluate_non_triviality(
    output: Sequence[int],
    target_terms: Sequence[int],
    wat_code: Optional[str] = None,
    variance_threshold: float = 1.0e-6,
    penalty_value: float = -0.5,
) -> NonTrivialityEvaluation:
    """Evaluates whether generated output is non-trivial relative to target sequence dynamics.

    Gating conditions:
    1. If target is dynamic (Var > 1e-6), constant outputs (Var < 1e-6 or S_input == 0) are gated out.
    2. If target has non-linear curvature (sum |Delta^2 y| > 1.0), trivial linear/identity outputs
       (sum |Delta^2 P| < 1e-6) that mismatch initial terms are gated out.
    """
    check_len = min(len(output), len(target_terms)) if output else 0
    if check_len < 2:
        return NonTrivialityEvaluation(
            output_variance=0.0,
            target_variance=compute_empirical_variance(target_terms),
            input_sensitivity=0.0,
            has_param_binding=True if (wat_code and "$n" in wat_code) else False,
            is_non_trivial=False,
            penalty=penalty_value,
        )

    out_var = compute_empirical_variance(output[:check_len])
    tgt_var = compute_empirical_variance(target_terms[:check_len])
    sensitivity = compute_input_sensitivity(output[:check_len])
    out_d2 = compute_second_difference_energy(output[:check_len])
    tgt_d2 = compute_second_difference_energy(target_terms[:check_len])
    has_param = True if (wat_code and "$n" in wat_code) else False

    # Check 1: Constant shortcut on dynamic target
    if tgt_var > variance_threshold:
        if out_var < variance_threshold or sensitivity < 1.0e-6:
            return NonTrivialityEvaluation(
                output_variance=out_var,
                target_variance=tgt_var,
                input_sensitivity=sensitivity,
                has_param_binding=has_param,
                is_non_trivial=False,
                penalty=penalty_value,
            )

    # Check 2: Linear identity shortcut on non-linear target
    # If target has strong curvature (tgt_d2 > 2.0) and output is purely linear (out_d2 < 1e-6),
    # verify if output actually matches the first 3 terms. If not, penalize as an identity shortcut.
    if tgt_d2 > 2.0 and out_d2 < 1.0e-6:
        first_terms_match = (
            check_len >= 3 and list(output[:3]) == list(target_terms[:3])
        )
        if not first_terms_match:
            return NonTrivialityEvaluation(
                output_variance=out_var,
                target_variance=tgt_var,
                input_sensitivity=sensitivity,
                has_param_binding=has_param,
                is_non_trivial=False,
                penalty=penalty_value,
            )

    return NonTrivialityEvaluation(
        output_variance=out_var,
        target_variance=tgt_var,
        input_sensitivity=sensitivity,
        has_param_binding=has_param,
        is_non_trivial=True,
        penalty=0.0,
    )


def compute_cross_input_mutual_information_proxy(
    batch_outputs: List[Sequence[int]],
    temperature: float = 0.1,
) -> List[float]:
    """Computes batch-level cross-input mutual information proxy R_MI.

    Penalizes policies whose output representations remain identical across distinct sequence tasks.
    """
    B = len(batch_outputs)
    if B <= 1:
        return [0.0] * B

    # Construct normalized output feature vectors
    max_len = max(len(out) for out in batch_outputs) if batch_outputs else 20
    vectors = []
    for out in batch_outputs:
        padded = list(out) + [0] * (max_len - len(out))
        arr = np.array(padded[:max_len], dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm > 1.0e-8:
            vectors.append(arr / norm)
        else:
            vectors.append(arr)

    matrix = np.array(vectors)  # (B, max_len)
    # Cosine similarity matrix S_i,j
    sim_matrix = np.clip(np.dot(matrix, matrix.T), -1.0, 1.0)  # (B, B)

    mi_rewards = []
    for i in range(B):
        # Exclude self-similarity diagonal
        other_sims = [sim_matrix[i, j] for j in range(B) if j != i]
        if other_sims:
            scaled = np.array(other_sims) / max(temperature, 1.0e-4)
            # Log-sum-exp
            max_s = np.max(scaled)
            lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
            r_mi = -float(lse - np.log(len(other_sims)))
        else:
            r_mi = 0.0
        mi_rewards.append(r_mi)

    return mi_rewards


def compute_pbrs_potential(
    wat_code: str,
    phi_comp_scale: float = 0.2,
    phi_bind_scale: float = 0.3,
) -> PotentialState:
    """Computes Potential-Based Reward Shaping potential Phi(s) over AST completion states."""
    if not wat_code:
        return PotentialState(step=0, structural_phase="EMPTY", phi_comp=0.0, phi_bind=0.0, total_potential=0.0)

    phi_comp = 0.0
    phi_bind = 0.0
    phase = "BODY"

    if 'export "compute"' in wat_code and '(param $n' in wat_code:
        phi_comp += phi_comp_scale
        phase = "HEADER_VALID"

    if 'local.get $n' in wat_code or 'local.tee' in wat_code:
        phi_bind += phi_bind_scale

    if 'loop' in wat_code or 'br_if' in wat_code:
        phi_bind += (phi_bind_scale * 0.5)

    total_phi = phi_comp + phi_bind
    return PotentialState(
        step=len(wat_code.split()),
        structural_phase=phase,
        phi_comp=phi_comp,
        phi_bind=phi_bind,
        total_potential=total_phi,
    )


def compute_binary_reward(
    exec_result: ExecutionResult,
    target_terms: Sequence[int],
    n_terms: Optional[int] = None,
    reward_pass: float = 1.0,
    reward_fail: float = -1.0,
) -> Tuple[float, Optional[int]]:
    """Evaluates strict binary outcome reward (+1.0 / -1.0) and divergence index."""
    check_len = n_terms if n_terms is not None else min(len(target_terms), len(exec_result.output))

    if exec_result.status != "SUCCESS":
        divergence_step = len(exec_result.output)
        return reward_fail, divergence_step

    if len(exec_result.output) < check_len:
        return reward_fail, len(exec_result.output)

    for idx in range(check_len):
        if exec_result.output[idx] != target_terms[idx]:
            return reward_fail, idx

    return reward_pass, None


def compute_composite_reward(
    exec_result: ExecutionResult,
    target_terms: Sequence[int],
    n_terms: int = 20,
    w_comp: float = 0.2,
    w_prefix: float = 1.0,
    w_dist: float = 0.5,
    epoch: int = 1,
    anneal_epochs: int = 25,
    divergence_token_idx: Optional[int] = None,
    wat_code: Optional[str] = None,
    enable_pbrs: bool = True,
    gamma_pbrs: float = 0.99,
) -> CompositeRewardBreakdown:
    """Computes dense-to-sparse composite reward with non-triviality gating, cosine annealing, and PBRS.

    Args:
        exec_result: Execution output from sandbox
        target_terms: Ground truth sequence terms
        n_terms: Number of sequence terms to evaluate (default 20)
        w_comp: Initial weight for compiler validity
        w_prefix: Initial weight for prefix match length
        w_dist: Initial weight for numerical proximity
        epoch: Current training epoch
        anneal_epochs: Horizon over which dense shaping terms decay via cosine schedule
        divergence_token_idx: Token index corresponding to state divergence (if known)
        wat_code: Source WAT code for AST potential and non-triviality evaluation
        enable_pbrs: Whether to apply Potential-Based Reward Shaping
        gamma_pbrs: Discount factor for PBRS
    """
    check_len = min(n_terms, len(target_terms))

    # 1. Compiler validation component
    if exec_result.status == "SUCCESS":
        r_comp = 0.2
    elif exec_result.status == "OUT_OF_FUEL":
        r_comp = -0.2
    else:
        r_comp = -0.5

    # 2. Prefix match length & divergence index
    prefix_matches = 0
    divergence_step = None
    if exec_result.status == "SUCCESS":
        out_len = min(check_len, len(exec_result.output))
        for idx in range(out_len):
            if exec_result.output[idx] == target_terms[idx]:
                prefix_matches += 1
            else:
                divergence_step = idx
                break
        if prefix_matches == check_len:
            divergence_step = None
        elif divergence_step is None:
            divergence_step = prefix_matches
    else:
        divergence_step = len(exec_result.output)

    r_prefix = prefix_matches / max(1, check_len)

    # 3. Normalized numerical distance component
    if exec_result.status == "SUCCESS" and exec_result.output:
        dist_errors = []
        for idx in range(min(check_len, len(exec_result.output))):
            diff = abs(exec_result.output[idx] - target_terms[idx])
            dist_errors.append(math.tanh(0.1 * min(diff, 1000.0)))
        r_dist = 1.0 - (sum(dist_errors) / max(1, len(dist_errors)))
    else:
        r_dist = 0.0

    # 4. Non-Triviality Gating Check
    non_trivial_eval = evaluate_non_triviality(
        output=exec_result.output,
        target_terms=target_terms,
        wat_code=wat_code,
    )
    if not non_trivial_eval.is_non_trivial and prefix_matches < check_len:
        # Zero out surrogate rewards and assign static non-triviality penalty
        r_prefix = 0.0
        r_dist = 0.0
        r_comp = min(r_comp, non_trivial_eval.penalty)

    # 5. Potential-Based Reward Shaping (PBRS)
    potential_shaping = 0.0
    if enable_pbrs and wat_code:
        pot_state = compute_pbrs_potential(wat_code)
        potential_shaping = pot_state.total_potential

    # 6. Strict exact match binary reward
    if exec_result.status == "SUCCESS" and prefix_matches == check_len:
        r_exact = 1.0
    else:
        r_exact = -1.0

    # Cosine annealing factor for dense auxiliary signals
    alpha = math.cos(min(1.0, epoch / max(1, anneal_epochs)) * (math.pi / 2.0))

    r_total = (
        r_exact
        + alpha * (w_comp * r_comp + w_prefix * r_prefix + w_dist * r_dist)
        + (alpha * potential_shaping if enable_pbrs else 0.0)
    )

    return CompositeRewardBreakdown(
        r_comp=r_comp,
        r_prefix=r_prefix,
        r_dist=r_dist,
        r_exact=r_exact,
        r_total=r_total,
        divergence_step=divergence_step,
        divergence_token_idx=divergence_token_idx,
        is_non_trivial=non_trivial_eval.is_non_trivial,
        output_variance=non_trivial_eval.output_variance,
        input_sensitivity=non_trivial_eval.input_sensitivity,
        potential_shaping=potential_shaping,
    )


# =========================================================================
# Phase 4 Parsimony-Regularized RLVR & Lexicographical Ranking
# =========================================================================


def compute_dense_log_distance_reward(
    outputs: Sequence[int],
    target_terms: Sequence[int],
) -> float:
    """Computes continuous dense log-distance return R_dense(P, Y) = 1/20 sum 1 / (1 + log10(|P(n) - y_n| + 1))."""
    check_len = min(len(outputs), len(target_terms)) if outputs else 0
    if check_len == 0:
        return 0.0

    scores = []
    for idx in range(check_len):
        diff = abs(outputs[idx] - target_terms[idx])
        # log10(diff + 1)
        log_d = math.log10(diff + 1.0)
        score = 1.0 / (1.0 + log_d)
        scores.append(score)

    # Pad missing terms with 0 score
    if check_len < len(target_terms):
        scores.extend([0.0] * (len(target_terms) - check_len))

    return float(np.mean(scores))


def compute_validity_reward(
    waste_ratio: float,
    threshold: float = 0.30,
    kappa: float = 2.0,
) -> float:
    """Computes intermediate validity reward with hard waste cutoff:

    R_validity = 0.1 * exp(-kappa * waste_ratio) if waste_ratio <= threshold else 0.0.
    """
    if waste_ratio > threshold:
        return 0.0
    return float(0.1 * math.exp(-kappa * waste_ratio))


def compute_covariant_parsimony_penalty(
    lengths: Sequence[int],
    rewards: Sequence[float],
    waste_ratios: Sequence[float],
    lambda_waste: float = 0.20,
) -> List[float]:
    """Computes Covariant Parsimony Pressure (CPP) returns across a group of rollouts:

    c_k = Cov(lengths, rewards) / (Var(lengths) + eps)
    R_CPP = R - max(0, -c_k) * (length - min_length) - lambda_waste * waste_ratio
    """
    G = len(lengths)
    if G == 0:
        return []
    if G == 1:
        return [float(rewards[0] - lambda_waste * waste_ratios[0])]

    lens_arr = np.array(lengths, dtype=np.float64)
    rews_arr = np.array(rewards, dtype=np.float64)
    waste_arr = np.array(waste_ratios, dtype=np.float64)

    var_len = float(np.var(lens_arr))
    if var_len < 1e-8:
        c_k = 0.0
    else:
        cov_matrix = np.cov(lens_arr, rews_arr)
        cov_val = float(cov_matrix[0, 1]) if cov_matrix.shape == (2, 2) else 0.0
        c_k = cov_val / (var_len + 1e-8)

    min_len = float(np.min(lens_arr))
    penalty_coef = max(0.0, -c_k)

    cpp_rewards = []
    for idx in range(G):
        length_penalty = penalty_coef * (lens_arr[idx] - min_len)
        waste_penalty = lambda_waste * waste_arr[idx]
        r_cpp = float(rews_arr[idx] - length_penalty - waste_penalty)
        cpp_rewards.append(r_cpp)

    return cpp_rewards


def compute_lexicographic_advantages(
    group_results: Sequence[Tuple[float, int]],
) -> List[float]:
    """Computes normalized ordinal advantages via Lexicographical Ranking (R_exec succ -|P_opt|).

    group_results: List of (reward, opt_token_length)
    Returns: Normalized advantages in [-1.0, 1.0]
    """
    G = len(group_results)
    if G <= 1:
        return [0.0] * G

    # Pair each candidate with its original index
    indexed_candidates = list(enumerate(group_results))

    # Sort lexicographically: higher reward first; if reward equal, shorter length first
    indexed_candidates.sort(key=lambda x: (x[1][0], -x[1][1]), reverse=True)

    # Assign ranks: 1 is best, G is worst
    # In case of exact tie on both reward and length, average rank
    ordinal_advs = [0.0] * G
    for rank_idx, (orig_idx, _) in enumerate(indexed_candidates):
        # rank from 1 to G
        rank = G - rank_idx  # 1 for worst, G for best
        # Normalized formula: 2 * (rank - 1) / (G - 1) - 1.0
        adv = (2.0 * (rank - 1.0) / (G - 1.0)) - 1.0
        ordinal_advs[orig_idx] = float(adv)

    return ordinal_advs

