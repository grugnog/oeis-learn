"""Integration tests for Transition Gate criteria evaluation."""

from __future__ import annotations

import pytest
from oeis_learn.curriculum.transition_gate import TransitionGateEvaluator


def test_transition_gate_all_passing():
    evaluator = TransitionGateEvaluator(
        max_advantage_collapse_rate=0.15,
        min_grammar_validity=0.985,
        min_candidate_pass_rate=0.75,
    )

    # 10 groups, each has 4 candidates that compute triangular numbers a(n) = n*(n+1)/2
    valid_cand = """(module
      (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
        (local $n64 i64)
        local.get $n i64.extend_i32_u local.set $n64
        local.get $n64 local.get $n64 i64.const 1 i64.add i64.mul i64.const 2 i64.div_u
        i64.const 0 i64.const 0 i64.const 0
      )
    )"""
    target = [n * (n + 1) // 2 for n in range(20)]

    rollout_groups = [[valid_cand] * 4 for _ in range(10)]
    target_sequences = [target for _ in range(10)]

    res = evaluator.evaluate_rollouts(rollout_groups, target_sequences)
    assert res["gate_passed"] is True
    assert res["advantage_collapse_rate"] == 0.0
    assert res["grammar_validity"] == 1.0
    assert res["candidate_pass_rate"] == 1.0


def test_transition_gate_fails_on_acr():
    evaluator = TransitionGateEvaluator(
        max_advantage_collapse_rate=0.15,
        min_grammar_validity=0.985,
        min_candidate_pass_rate=0.75,
    )

    valid_cand = """(module
      (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
        (local $n64 i64)
        local.get $n i64.extend_i32_u local.set $n64
        local.get $n64 local.get $n64 i64.const 1 i64.add i64.mul i64.const 2 i64.div_u
        i64.const 0 i64.const 0 i64.const 0
      )
    )"""
    # Bad candidate: outputs 0
    bad_cand = """(module
      (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
        i64.const 0 i64.const 0 i64.const 0 i64.const 0
      )
    )"""
    target = [n * (n + 1) // 2 for n in range(20)]

    # 4 good groups, 6 completely bad groups -> ACR = 6/10 = 0.60 > 0.15
    rollout_groups = [[valid_cand] * 4 for _ in range(4)] + [[bad_cand] * 4 for _ in range(6)]
    target_sequences = [target for _ in range(10)]

    res = evaluator.evaluate_rollouts(rollout_groups, target_sequences)
    assert res["gate_passed"] is False
    assert res["advantage_collapse_rate"] == 0.60
