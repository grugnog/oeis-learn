"""Qualified carried-state recurrence execution; old recognition is retired."""

from oeis_learn.synthesis.reference import evaluate
from oeis_learn.synthesis.fitting import fit_program

SOURCE = "(program (local $b0 bigint) (local $b1 bigint) (local $b2 bigint) (local $c0 control) (set $b1 (int 1)) (loop $c0 (to_control $index) (set $b2 (add (get $b0) (get $b1))) (set $b0 (get $b1)) (set $b1 (get $b2))) (return (get $b0)))"


def test_order2_carried_state_has_independent_exact_results():
    expected = [0, 1]
    for _ in range(18):
        expected.append(sum(expected[-2:]))
    result = evaluate(SOURCE, list(map(str, range(20))))
    assert result.outcome == "ok" and list(map(int, result.values)) == expected


def test_old_recurrence_skeleton_is_explicitly_unsupported():
    result = fit_program("(module (func (result i64) i64.const_?))", [0, 1, 1, 2, 3])
    assert result.outcome == "unsupported"
