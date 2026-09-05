"""Extrapolation Horizon (N+K, K=100) Generalization Verifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
from oeis_learn.data.models import ExecutionResult
from oeis_learn.sandbox.runner import WasmRunner


@dataclass
class ExtrapolationDetailedResult:
    """Detailed evidence of extrapolation verification."""

    passed: bool
    is_qualified: bool
    observed_match: bool
    unseen_match: bool
    first_observed_divergence: Optional[int]
    first_unseen_divergence: Optional[int]
    requested_horizon: int
    available_horizon: int
    execution_result: Optional[ExecutionResult] = None
    divergence_expected: Optional[int] = None
    divergence_actual: Optional[int] = None
    reason: Optional[str] = None


class ExtrapolationVerifier:
    """Verifies that synthesized candidate algorithms generalize to unseen extrapolation terms."""

    def __init__(self, runner: Optional[WasmRunner] = None, n_train: int = 20, k_extrapolate: int = 100):
        self.runner = runner or WasmRunner(fuel_budget=10000)
        self.n_train = n_train
        self.k_extrapolate = k_extrapolate

    def verify(
        self,
        wat_code: str,
        ground_truth_terms: Sequence[int],
    ) -> bool:
        """Verifies candidate WAT program output against extrapolated ground-truth terms."""
        detailed = self.verify_detailed(wat_code, ground_truth_terms)
        return detailed.passed

    def verify_detailed(
        self,
        wat_code: str,
        ground_truth_terms: Sequence[int],
        result_profile: str = "i64_scalar_v1",
        precomputed_result: Optional[ExecutionResult] = None,
    ) -> ExtrapolationDetailedResult:
        """Evaluates detailed observed and unseen term comparisons."""
        total_requested = self.n_train + self.k_extrapolate
        available_horizon = len(ground_truth_terms)
        is_qualified = available_horizon >= total_requested
        eval_horizon = min(total_requested, available_horizon)

        if eval_horizon == 0:
            return ExtrapolationDetailedResult(
                passed=False,
                is_qualified=False,
                observed_match=False,
                unseen_match=False,
                first_observed_divergence=None,
                first_unseen_divergence=None,
                requested_horizon=total_requested,
                available_horizon=available_horizon,
                reason="No ground truth terms provided",
            )

        res = precomputed_result
        if res is None:
            res = self.runner.run_single(
                wat_code,
                terms_to_generate=eval_horizon,
                result_profile=result_profile,
            )

        if res.status != "SUCCESS":
            return ExtrapolationDetailedResult(
                passed=False,
                is_qualified=is_qualified,
                observed_match=False,
                unseen_match=False,
                first_observed_divergence=0,
                first_unseen_divergence=0,
                requested_horizon=total_requested,
                available_horizon=available_horizon,
                execution_result=res,
                reason=f"Execution error: {res.status} ({res.error})",
            )

        output = res.output
        if len(output) < eval_horizon:
            return ExtrapolationDetailedResult(
                passed=False,
                is_qualified=is_qualified,
                observed_match=False,
                unseen_match=False,
                first_observed_divergence=len(output) if len(output) < self.n_train else None,
                first_unseen_divergence=len(output) - self.n_train if len(output) >= self.n_train else 0,
                requested_horizon=total_requested,
                available_horizon=available_horizon,
                execution_result=res,
                reason=f"Output too short: generated {len(output)} < requested {eval_horizon}",
            )

        # 1. Check observed terms (0..min(n_train, eval_horizon)-1)
        obs_limit = min(self.n_train, eval_horizon)
        first_obs = None
        div_exp = None
        div_act = None
        for i in range(obs_limit):
            if int(output[i]) != int(ground_truth_terms[i]):
                first_obs = i
                div_exp = int(ground_truth_terms[i])
                div_act = int(output[i])
                break

        observed_match = (first_obs is None)

        # 2. Check unseen terms (n_train..eval_horizon-1)
        first_uns = None
        if eval_horizon > self.n_train:
            unseen_count = eval_horizon - self.n_train
            for j in range(unseen_count):
                idx = self.n_train + j
                if int(output[idx]) != int(ground_truth_terms[idx]):
                    first_uns = j
                    if div_exp is None:
                        div_exp = int(ground_truth_terms[idx])
                        div_act = int(output[idx])
                    break
            unseen_match = (first_uns is None)
        else:
            unseen_match = True

        passed = observed_match and unseen_match

        return ExtrapolationDetailedResult(
            passed=passed,
            is_qualified=is_qualified,
            observed_match=observed_match,
            unseen_match=unseen_match,
            first_observed_divergence=first_obs,
            first_unseen_divergence=first_uns,
            requested_horizon=total_requested,
            available_horizon=available_horizon,
            execution_result=res,
            divergence_expected=div_exp,
            divergence_actual=div_act,
            reason=None if passed else "Divergence detected",
        )
