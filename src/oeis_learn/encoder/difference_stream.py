"""Difference & p-Adic Stream (S3): Finite differences and p-adic valuation embeddings."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from oeis_learn.encoder.magnitude_stream import compute_signed_log_scalar

PRIMES = [2, 3, 5, 7, 11, 13]
MAX_VALUATION = 16


def compute_p_adic_valuation(val: int, p: int, max_k: int = MAX_VALUATION) -> int:
    """Computes the p-adic valuation v_p(x) = max{k : p^k divides x}, capped at max_k."""
    if val == 0:
        return max_k
    val = abs(val)
    k = 0
    while k < max_k and val % p == 0:
        val //= p
        k += 1
    return k


def compute_sequence_differences_and_padic(
    seq: Sequence[Union[int, float]],
    primes: Optional[Sequence[int]] = None,
    max_valuation: int = MAX_VALUATION,
) -> Tuple[List[float], List[float], List[List[int]]]:
    """Computes first differences, second differences, and p-adic valuations for a sequence."""
    n = len(seq)
    d1_scalars: List[float] = []
    d2_scalars: List[float] = []
    padic_vals: List[List[int]] = []

    active_primes = primes if primes is not None else PRIMES
    int_seq = [int(x) for x in seq]

    # Compute raw differences
    d1_raw = [0] + [int_seq[i] - int_seq[i - 1] for i in range(1, n)]
    d2_raw = [0, 0] + [d1_raw[i] - d1_raw[i - 1] for i in range(2, n)]

    for i in range(n):
        d1_scalars.append(compute_signed_log_scalar(d1_raw[i]))
        d2_scalars.append(compute_signed_log_scalar(d2_raw[i]))
        p_vals = [compute_p_adic_valuation(int_seq[i], p, max_k=max_valuation) for p in active_primes]
        padic_vals.append(p_vals)

    return d1_scalars, d2_scalars, padic_vals


class DifferenceStream(nn.Module):
    """Encodes step dynamics via finite differences and p-adic prime valuation ordinals."""

    def __init__(
        self,
        d_model: int = 256,
        d_padic: int = 16,
        primes: Optional[Sequence[int]] = None,
        max_valuation: int = MAX_VALUATION,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_padic = d_padic
        self.primes = list(primes) if primes is not None else list(PRIMES)
        self.max_valuation = max_valuation
        self.num_primes = len(self.primes)

        # Embedding for each p-adic valuation (0 to max_valuation inclusive = max_valuation + 1 categories)
        self.padic_embeddings = nn.ModuleList([
            nn.Embedding(self.max_valuation + 1, d_padic) for _ in range(self.num_primes)
        ])

        # Input: 2 continuous difference scalars (d1, d2) + (num_primes * d_padic)
        total_input_dim = 2 + (self.num_primes * d_padic)
        self.projection = nn.Sequential(
            nn.Linear(total_input_dim, d_model, dtype=torch.float32),
            nn.LayerNorm(d_model, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward_features(
        self,
        diff_scalars: torch.Tensor,   # shape: (batch, seq_len, 2)
        padic_indices: torch.Tensor,  # shape: (batch, seq_len, 6)
    ) -> torch.Tensor:
        """Forward pass with precomputed difference scalars and padic valuation indices."""
        batch_size, seq_len, _ = diff_scalars.shape
        diff_scalars = diff_scalars.to(dtype=torch.float32)
        padic_indices = padic_indices.to(dtype=torch.long)

        padic_embeds = []
        for p_idx in range(self.num_primes):
            p_embed = self.padic_embeddings[p_idx](padic_indices[:, :, p_idx])  # (batch, seq_len, d_padic)
            padic_embeds.append(p_embed)

        all_padic = torch.cat(padic_embeds, dim=-1)  # (batch, seq_len, num_primes * d_padic)
        combined = torch.cat([diff_scalars, all_padic], dim=-1)  # (batch, seq_len, 2 + num_primes * d_padic)
        from typing import cast
        return cast(torch.Tensor, self.projection(combined))

    def forward(
        self, sequences: Sequence[Sequence[Union[int, float]]], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Processes raw batch of integer sequences."""
        batch_size = len(sequences)
        max_len = max((len(seq) for seq in sequences), default=0)

        diff_tensor = torch.zeros((batch_size, max_len, 2), dtype=torch.float32, device=device)
        padic_tensor = torch.zeros((batch_size, max_len, self.num_primes), dtype=torch.long, device=device)

        for b_idx, seq in enumerate(sequences):
            if not seq:
                continue
            d1s, d2s, pvals = compute_sequence_differences_and_padic(
                seq, primes=self.primes, max_valuation=self.max_valuation
            )
            for t_idx in range(len(seq)):
                diff_tensor[b_idx, t_idx, 0] = d1s[t_idx]
                diff_tensor[b_idx, t_idx, 1] = d2s[t_idx]
                padic_tensor[b_idx, t_idx] = torch.tensor(pvals[t_idx], dtype=torch.long, device=device)

        return self.forward_features(diff_tensor, padic_tensor)


class DifferenceStreamEncoder(DifferenceStream):
    """Tri-Stream Encoder v2 Difference Stream computing normalized Newton forward difference quotients."""

    def compute_newton_quotients(self, seq: Sequence[Union[int, float]]) -> torch.Tensor:
        """Computes normalized Newton forward difference quotients:

        D^(1)_i = y_{i+1} - y_i
        D^(2)_i = (y_{i+2} - 2*y_{i+1} + y_i) / 2!
        D^(3)_i = (y_{i+3} - 3*y_{i+2} + 3*y_{i+1} - y_i) / 3!
        """
        int_seq = [float(x) for x in seq]
        n = len(int_seq)
        diff_matrix = torch.zeros((n, 3), dtype=torch.float32)

        for i in range(n):
            # D^(1)
            if i + 1 < n:
                d1 = int_seq[i + 1] - int_seq[i]
            elif i >= 1:
                d1 = int_seq[i] - int_seq[i - 1]
            else:
                d1 = 0.0
            diff_matrix[i, 0] = d1

            # D^(2)
            if i + 2 < n:
                d2 = (int_seq[i + 2] - 2.0 * int_seq[i + 1] + int_seq[i]) / 2.0
            elif i + 1 < n and i >= 1:
                d2 = (int_seq[i + 1] - 2.0 * int_seq[i] + int_seq[i - 1]) / 2.0
            elif i >= 2:
                d2 = (int_seq[i] - 2.0 * int_seq[i - 1] + int_seq[i - 2]) / 2.0
            else:
                d2 = 0.0
            diff_matrix[i, 1] = d2

            # D^(3)
            if i + 3 < n:
                d3 = (int_seq[i + 3] - 3.0 * int_seq[i + 2] + 3.0 * int_seq[i + 1] - int_seq[i]) / 6.0
            elif i + 2 < n and i >= 1:
                d3 = (int_seq[i + 2] - 3.0 * int_seq[i + 1] + 3.0 * int_seq[i] - int_seq[i - 1]) / 6.0
            elif i + 1 < n and i >= 2:
                d3 = (int_seq[i + 1] - 3.0 * int_seq[i] + 3.0 * int_seq[i - 1] - int_seq[i - 2]) / 6.0
            elif i >= 3:
                d3 = (int_seq[i] - 3.0 * int_seq[i - 1] + 3.0 * int_seq[i - 2] - int_seq[i - 3]) / 6.0
            else:
                d3 = 0.0
            diff_matrix[i, 2] = d3

        return diff_matrix

    def forward_from_terms(
        self, sequences: Sequence[Sequence[Union[int, float]]], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Processes raw batch of integer sequences and computes projected representations."""
        return self.forward(sequences, device=device)

