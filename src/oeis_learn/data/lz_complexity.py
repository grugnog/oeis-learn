"""Lempel-Ziv (LZ78 / LZ76) complexity computation for integer sequences."""

from __future__ import annotations

import math
from typing import Sequence, Union


def lempel_ziv_complexity(sequence: Union[Sequence[int], str]) -> int:
    """Computes the Lempel-Ziv complexity (number of distinct substrings in greedy factorization).

    Args:
        sequence: An integer sequence or string representation.

    Returns:
        The integer count of unique phrases in the LZ factorization.
    """
    if isinstance(sequence, (list, tuple)):
        # Convert to compact delimited string or character sequence
        s = ",".join(map(str, sequence))
    else:
        s = str(sequence)

    if not s:
        return 0

    n = len(s)
    phrases = set()
    i = 0
    k = 1
    while i + k <= n:
        sub = s[i : i + k]
        if sub not in phrases:
            phrases.add(sub)
            i += k
            k = 1
        else:
            k += 1

    # If residual characters remain at the end
    if i < n:
        phrases.add(s[i:])

    return len(phrases)


def normalized_lz_complexity(sequence: Union[Sequence[int], str]) -> float:
    """Computes normalized Lempel-Ziv complexity: C_norm = (C(s) * log2(n)) / n.

    Args:
        sequence: Sequence of integers or string.

    Returns:
        Normalized complexity proxy as a float.
    """
    if isinstance(sequence, (list, tuple)):
        s = ",".join(map(str, sequence))
    else:
        s = str(sequence)

    n = len(s)
    if n <= 1:
        return 1.0

    c = lempel_ziv_complexity(s)
    return float((c * math.log2(n)) / n)
