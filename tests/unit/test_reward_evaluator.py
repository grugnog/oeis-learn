"""Unit tests for binary outcome reward evaluator."""

import pytest
from oeis_learn.data.models import ExecutionResult
from oeis_learn.rl.reward import compute_binary_reward


def test_binary_reward_exact_match():
    target = [0, 1, 1, 2, 3, 5, 8]
    res = ExecutionResult(status="SUCCESS", consumed_fuel=150, output=[0, 1, 1, 2, 3, 5, 8])
    rew, div = compute_binary_reward(res, target)

    assert rew == 1.0
    assert div is None


def test_binary_reward_partial_match_fails():
    target = [0, 1, 1, 2, 3, 5, 8]
    # Diverges at index 4 (output has 4 instead of 3)
    res = ExecutionResult(status="SUCCESS", consumed_fuel=150, output=[0, 1, 1, 2, 4, 6, 8])
    rew, div = compute_binary_reward(res, target)

    assert rew == -1.0
    assert div == 4


def test_binary_reward_trap_fails():
    target = [0, 1, 1, 2, 3, 5, 8]
    res = ExecutionResult(status="OUT_OF_FUEL", consumed_fuel=10000, output=[0, 1])
    rew, div = compute_binary_reward(res, target)

    assert rew == -1.0
    assert div == 2


def test_composite_reward_shaping_and_annealing():
    from oeis_learn.rl.reward import compute_composite_reward

    target = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    res_success = ExecutionResult(status="SUCCESS", consumed_fuel=200, output=[0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    res_partial = ExecutionResult(status="SUCCESS", consumed_fuel=200, output=[0, 1, 1, 2, 0, 0, 0, 0, 0, 0])

    # Early epoch (epoch 1): dense rewards active
    r_early = compute_composite_reward(res_partial, target, epoch=1, anneal_epochs=25)
    assert r_early.r_prefix > 0.0
    assert r_early.r_exact == -1.0
    # r_total should be higher than strict -1.0 due to partial credit
    assert r_early.r_total > -1.0

    # Late epoch (epoch 25): dense rewards annealed to strict binary
    r_late = compute_composite_reward(res_partial, target, epoch=25, anneal_epochs=25)
    assert abs(r_late.r_total - (-1.0)) < 1e-4

    # Exact match gets positive reward in both early and late
    r_exact_early = compute_composite_reward(res_success, target, epoch=1, anneal_epochs=25)
    assert r_exact_early.r_total > 1.0
    r_exact_late = compute_composite_reward(res_success, target, epoch=25, anneal_epochs=25)
    assert abs(r_exact_late.r_total - 1.0) < 1e-4


def test_non_triviality_gating_constant_shortcut():
    from oeis_learn.rl.reward import (
        compute_composite_reward,
        compute_empirical_variance,
        compute_input_sensitivity,
        evaluate_non_triviality,
    )

    # Dynamic target: Powers of 2 (1, 2, 4, 8, 16, 32...)
    target = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Degenerate constant candidate: [16, 16, 16, 16...]
    const_output = [16] * 10

    var = compute_empirical_variance(const_output)
    assert var == 0.0

    sens = compute_input_sensitivity(const_output)
    assert sens == 0.0

    eval_res = evaluate_non_triviality(const_output, target)
    assert not eval_res.is_non_trivial
    assert eval_res.penalty == -0.5

    # In composite reward, non-triviality gate zeros out surrogate prefix and distance rewards
    res_const = ExecutionResult(status="SUCCESS", consumed_fuel=50, output=const_output)
    r_comp = compute_composite_reward(res_const, target, epoch=1, anneal_epochs=25)
    assert not r_comp.is_non_trivial
    assert r_comp.r_prefix == 0.0
    assert r_comp.r_dist == 0.0
    assert r_comp.r_comp <= -0.5


def test_non_triviality_gating_dynamic_program_passes():
    from oeis_learn.rl.reward import (
        compute_composite_reward,
        compute_empirical_variance,
        compute_input_sensitivity,
        evaluate_non_triviality,
    )

    target = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    dynamic_output = [1, 2, 4, 8, 15, 30, 60, 120, 240, 480]  # Near match with dynamics

    var = compute_empirical_variance(dynamic_output)
    assert var > 10.0

    sens = compute_input_sensitivity(dynamic_output)
    assert sens > 10.0

    eval_res = evaluate_non_triviality(dynamic_output, target, wat_code='(local.get $n)')
    assert eval_res.is_non_trivial
    assert eval_res.penalty == 0.0

    res_dyn = ExecutionResult(status="SUCCESS", consumed_fuel=100, output=dynamic_output)
    r_comp = compute_composite_reward(res_dyn, target, epoch=1, anneal_epochs=25, wat_code='(local.get $n)')
    assert r_comp.is_non_trivial
    assert r_comp.r_prefix > 0.0
    assert r_comp.r_dist > 0.0


def test_non_triviality_gating_linear_identity_shortcut_on_nonlinear_target():
    from oeis_learn.rl.reward import (
        compute_composite_reward,
        evaluate_non_triviality,
    )

    # Quadratic target: 0, 1, 4, 9, 16, 25, 36, 49, 64, 81
    target = [n * n for n in range(10)]
    # Linear identity candidate: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (matches first 2 terms, then diverges on curvature)
    identity_output = list(range(10))

    eval_res = evaluate_non_triviality(identity_output, target)
    assert not eval_res.is_non_trivial
    assert eval_res.penalty == -0.5

    res = ExecutionResult(status="SUCCESS", consumed_fuel=100, output=identity_output)
    r_comp = compute_composite_reward(res, target, epoch=1, anneal_epochs=25)
    assert not r_comp.is_non_trivial
    assert r_comp.r_prefix == 0.0
    assert r_comp.r_dist == 0.0
    assert r_comp.r_comp <= -0.5
