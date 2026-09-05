"""Unit tests for Extrapolation Horizon (N+K, K=100) verifier."""

import pytest
from oeis_learn.curriculum.extrapolation import ExtrapolationVerifier


def test_extrapolation_verifier_exact_polynomial():
    verifier = ExtrapolationVerifier(n_train=10, k_extrapolate=20)

    # Program: a(n) = 2*n + 1
    wat = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (i64.add
                (i64.mul (i64.extend_i32_s (local.get $n)) (i64.const 2))
                (i64.const 1)
            )
        )
    )
    """
    ground_truth = [2 * n + 1 for n in range(50)]
    assert verifier.verify(wat, ground_truth) is True

    # Check incorrect program (e.g. 2n + 2)
    bad_wat = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (i64.add
                (i64.mul (i64.extend_i32_s (local.get $n)) (i64.const 2))
                (i64.const 2)
            )
        )
    )
    """
    assert verifier.verify(bad_wat, ground_truth) is False


def test_extrapolation_verifier_k100_horizon_and_mdl():
    from oeis_learn.curriculum.mdl_verifier import MdlVerifier

    verifier_100 = ExtrapolationVerifier(n_train=20, k_extrapolate=100)
    mdl_verifier = MdlVerifier(threshold=1.20)

    # Program: Triangular numbers a(n) = n*(n+1)/2
    wat = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (local $n64 i64)
            (local.set $n64 (i64.extend_i32_u (local.get $n)))
            (i64.div_u
                (i64.mul (local.get $n64) (i64.add (local.get $n64) (i64.const 1)))
                (i64.const 2)
            )
        )
    )
    """
    ground_truth = [n * (n + 1) // 2 for n in range(120)]
    assert verifier_100.verify(wat, ground_truth) is True

    # Check MDL ratio
    ratio, byte_size, lz_comp = mdl_verifier.compute_mdl_ratio(wat, ground_truth)
    assert ratio <= 1.20
    assert mdl_verifier.verify(wat, ground_truth) is True
