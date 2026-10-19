"""Reuse the qualified canonical family key for concrete archive structures."""

from oeis_learn.data.bigint_programs import canonical_family


def structural_identity(source):
    return canonical_family(source)
