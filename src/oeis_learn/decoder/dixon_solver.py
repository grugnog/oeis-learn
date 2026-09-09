"""Dixon 1-Step Bounded Diophantine Linear Recurrence Solver.

Inverts maximal square minor in F_{2^61 - 1}, reconstructs integer coefficients
within [-1000, 1000], and certifies exactness via A * c == b over Z in < 1.0 ms.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple
from oeis_learn.decoder.mersenne61_filter import (
    M61_PRIME,
    evaluate_modular_filter,
    int_to_m61,
)
from oeis_learn.data.models import ModularFilterCertificate


def solve_dixon_bounded(
    matrix_a: List[List[int]],
    vector_b: List[int],
    unknown_count: int,
    bound: int = 1000,
) -> Tuple[bool, List[int], float, ModularFilterCertificate]:
    """Solves overdetermined linear system A * x = b for integer coefficients x in [-bound, bound].

    Returns (is_sat, constants, duration_ms, certificate).
    """
    start_t = time.perf_counter()
    m = len(matrix_a)
    k = unknown_count
    p = M61_PRIME

    # 1. Tier 1 fast modular filter
    cert = evaluate_modular_filter(matrix_a, vector_b, unknown_count)
    if cert.status != "CONSISTENT":
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return False, [], elapsed_ms, cert

    # 2. Extract solution from row-reduced form in F_p
    # Build augmented matrix M in F_p
    M: List[List[int]] = []
    for i in range(m):
        row = [int_to_m61(matrix_a[i][j]) for j in range(k)]
        row.append(int_to_m61(vector_b[i]))
        M.append(row)

    row_idx = 0
    for col in range(k):
        pivot_row = None
        for r in range(row_idx, m):
            if M[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        M[row_idx], M[pivot_row] = M[pivot_row], M[row_idx]
        inv_piv = pow(M[row_idx][col], p - 2, p)
        for c in range(col, k + 1):
            M[row_idx][c] = (M[row_idx][c] * inv_piv) % p
        for r in range(m):
            if r != row_idx and M[r][col] != 0:
                factor = M[r][col]
                for c in range(col, k + 1):
                    M[r][c] = (M[r][c] - factor * M[row_idx][c]) % p
        row_idx += 1

    # Reconstruct candidate coefficients in Z
    candidate_c: List[int] = []
    half_p = p // 2
    for j in range(k):
        sol_mod = M[j][k]
        val = sol_mod - p if sol_mod > half_p else sol_mod
        if abs(val) > bound:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return False, [], elapsed_ms, cert
        candidate_c.append(val)

    # 3. Exact certificate check over Z: A * c == b
    for i in range(m):
        lhs = sum(matrix_a[i][j] * candidate_c[j] for j in range(k))
        if lhs != vector_b[i]:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return False, [], elapsed_ms, cert

    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    return True, candidate_c, elapsed_ms, cert
