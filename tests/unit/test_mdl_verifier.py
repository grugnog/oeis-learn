"""Unit tests for Minimum Description Length (MDL) Anti-Memorization verifier."""

import pytest
from oeis_learn.curriculum.mdl_verifier import MdlVerifier, get_wat_byte_size


def test_mdl_verifier_compact_program():
    verifier = MdlVerifier(threshold=1.2)
    # Compact loop / polynomial program
    wat = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (i64.mul (i64.extend_i32_s (local.get $n)) (i64.const 3))
        )
    )
    """
    seq = [3 * n for n in range(50)]
    ratio, byte_size, lz = verifier.compute_mdl_ratio(wat, seq)

    assert byte_size > 0
    assert ratio <= 1.2
    assert verifier.verify(wat, seq) is True


def test_mdl_verifier_rejects_huge_lookup_table():
    verifier = MdlVerifier(threshold=1.2)
    # Hardcoded lookup table for 35 terms
    consts = " ".join(f"(i64.const {i})" for i in range(35))
    wat = f"""
    (module
        (func (export "compute") (param $n i32) (result i64)
            {consts}
            (i64.const 0)
        )
    )
    """
    seq = [1, 2, 3] * 10
    assert verifier.verify(wat, seq) is False
