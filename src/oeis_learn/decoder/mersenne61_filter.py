"""Mersenne-61 Fast Modular Rank Filter for Multi-Limb Skeletons.

Projects 256-bit recurrence systems into finite field F_{2^61 - 1} via bitwise limb folding
and applies Gaussian elimination with partial pivoting to classify rank and consistency
in < 0.5 milliseconds.
"""

from __future__ import annotations

import time
from typing import List, Tuple
from oeis_learn.data.models import ModularFilterCertificate

M61_PRIME = (1 << 61) - 1  # 2305843009213693951
M61_MASK = M61_PRIME


def fold61_u64(u: int) -> int:
    """Folds a 64-bit unsigned integer into F_{2^61 - 1}."""
    return (u & M61_MASK) + (u >> 61)


def project_i256_to_m61(limbs: Tuple[int, int, int, int]) -> int:
    """Projects 4-limb 256-bit integer to F_{2^61 - 1} using fast modular limb powers."""
    l0, l1, l2, l3 = limbs
    f0 = fold61_u64(l0 & 0xFFFFFFFFFFFFFFFF)
    f1 = fold61_u64(l1 & 0xFFFFFFFFFFFFFFFF)
    f2 = fold61_u64(l2 & 0xFFFFFFFFFFFFFFFF)
    f3 = fold61_u64(l3 & 0xFFFFFFFFFFFFFFFF)

    # 2^64 mod (2^61-1) = 8, 2^128 mod p = 64, 2^192 mod p = 512
    # 2^256 mod (2^61-1) = 4096
    u_mod = (f0 + 8 * f1 + 64 * f2 + 512 * f3) % M61_PRIME
    # Correct if signed negative (bit 255 set)
    is_neg = (l3 & (1 << 63)) != 0
    if is_neg:
        return (u_mod - 4096) % M61_PRIME
    return u_mod


def int_to_m61(val: int) -> int:
    """Projects any integer into F_{2^61 - 1}."""
    return val % M61_PRIME


def m61_inv(a: int) -> int:
    """Computes multiplicative inverse in F_{2^61 - 1} via Fermat's Little Theorem."""
    return pow(a % M61_PRIME, M61_PRIME - 2, M61_PRIME)


def evaluate_modular_filter(
    matrix_a: List[List[int]],
    vector_b: List[int],
    unknown_count: int,
) -> ModularFilterCertificate:
    """Evaluates rank and consistency of linear system A * x = b over F_{2^61 - 1}.

    Returns ModularFilterCertificate with status:
      - 'CONSISTENT': rank(A) == rank([A|b]) == k
      - 'UNDERDETERMINED': rank(A) < k
      - 'INCONSISTENT': rank([A|b]) > rank(A)
    """
    start_t = time.perf_counter()
    m = len(matrix_a)
    k = unknown_count

    if m == 0 or k == 0:
        elapsed_us = (time.perf_counter() - start_t) * 1_000_000.0
        return ModularFilterCertificate(
            status="UNDERDETERMINED",
            prime=M61_PRIME,
            augmented_rank=0,
            coefficient_rank=0,
            unknown_count=k,
            elapsed_microseconds=elapsed_us,
            penalty_reward=-0.50,
        )

    # Build augmented matrix M in F_p: m rows, k+1 cols
    M: List[List[int]] = []
    for i in range(m):
        row = [int_to_m61(matrix_a[i][j]) for j in range(k)]
        row.append(int_to_m61(vector_b[i]))
        M.append(row)

    # Gaussian elimination with partial pivoting over F_p
    row_idx = 0
    p = M61_PRIME

    for col in range(k):
        # Find pivot
        pivot_row = None
        for r in range(row_idx, m):
            if M[r][col] != 0:
                pivot_row = r
                break

        if pivot_row is None:
            # Column is all zeros below row_idx
            continue

        # Swap rows
        M[row_idx], M[pivot_row] = M[pivot_row], M[row_idx]

        # Invert pivot
        inv_piv = pow(M[row_idx][col], p - 2, p)
        # Normalize pivot row
        for c in range(col, k + 1):
            M[row_idx][c] = (M[row_idx][c] * inv_piv) % p

        # Eliminate other rows
        for r in range(m):
            if r != row_idx and M[r][col] != 0:
                factor = M[r][col]
                for c in range(col, k + 1):
                    M[r][c] = (M[r][c] - factor * M[row_idx][c]) % p

        row_idx += 1
        if row_idx >= m:
            break

    # Determine rank of coefficient matrix A and augmented matrix M
    coeff_rank = 0
    augmented_rank = 0

    for r in range(m):
        has_coeff = any(M[r][c] != 0 for c in range(k))
        has_aug = has_coeff or (M[r][k] != 0)
        if has_coeff:
            coeff_rank += 1
        if has_aug:
            augmented_rank += 1

    elapsed_us = (time.perf_counter() - start_t) * 1_000_000.0

    if augmented_rank > coeff_rank:
        status = "INCONSISTENT"
        penalty = -0.50
    elif coeff_rank < k:
        status = "UNDERDETERMINED"
        penalty = -0.50
    else:
        status = "CONSISTENT"
        penalty = 0.0

    return ModularFilterCertificate(
        status=status,
        prime=p,
        augmented_rank=augmented_rank,
        coefficient_rank=coeff_rank,
        unknown_count=k,
        elapsed_microseconds=elapsed_us,
        penalty_reward=penalty,
    )
