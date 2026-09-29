"""Deterministic Algebraic Sequence Transformations for SSL positive pair generation."""

from __future__ import annotations

import math
from typing import List, Sequence


def partial_sum_transform(seq: Sequence[int]) -> List[int]:
    """Computes partial sums: b(n) = sum_{k=0}^n a(k)."""
    res = []
    curr = 0
    for x in seq:
        curr += x
        res.append(curr)
    return res


def first_difference_transform(seq: Sequence[int]) -> List[int]:
    """Computes first forward difference: b(n) = a(n+1) - a(n)."""
    if len(seq) <= 1:
        return [0]
    return [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]


def binomial_transform(seq: Sequence[int]) -> List[int]:
    """Computes binomial transform: b(n) = sum_{k=0}^n C(n, k) * a(k)."""
    n_len = min(len(seq), 30)  # Bound to avoid combinatorial explosions on long sequences
    res = []
    for n in range(n_len):
        val = 0
        for k in range(n + 1):
            val += math.comb(n, k) * seq[k]
        res.append(val)
    return res


def shift_transform(seq: Sequence[int], k: int = 1) -> List[int]:
    """Computes sequence shift T_k: b(n) = a(n + k)."""
    if k >= len(seq):
        return [0]
    return list(seq[k:])


def alternating_sign_transform(seq: Sequence[int]) -> List[int]:
    """Computes alternating sign transform: b(n) = (-1)^n * a(n)."""
    return [((-1) ** n) * seq[n] for n in range(len(seq))]


def termwise_sum_transform(seq_a: Sequence[int], seq_b: Sequence[int]) -> List[int]:
    """Computes termwise sequence addition: c(n) = a(n) + b(n)."""
    min_len = min(len(seq_a), len(seq_b))
    return [seq_a[i] + seq_b[i] for i in range(min_len)]


def generate_algebraic_pair(seq: Sequence[int], transform_type: str = "partial_sum") -> List[int]:
    """Applies named algebraic transform to generate a positive representation pair."""
    if transform_type == "partial_sum":
        return partial_sum_transform(seq)
    elif transform_type == "difference":
        return first_difference_transform(seq)
    elif transform_type == "binomial":
        return binomial_transform(seq)
    elif transform_type == "shift":
        return shift_transform(seq, k=1)
    elif transform_type == "alternating":
        return alternating_sign_transform(seq)
    else:
        return list(seq)
