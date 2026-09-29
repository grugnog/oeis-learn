"""Unit tests for execution tracer divergence step locator."""

from oeis_learn.sandbox.tracer import locate_divergence_token_span


def test_locate_divergence_token_span():
    wat = '(module (func (export "compute") (param $n i32) (result i64) (i64.add (local.get $n) (i64.const 1))))'
    t_start, t_end = locate_divergence_token_span(wat, divergence_step=2, total_tokens=20)

    assert t_start >= 0
    assert t_end > t_start
    assert t_end <= 20
