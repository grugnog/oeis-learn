"""Magnitude Stream (S1): Signed continuous log-magnitude scalar projection."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union, cast
import torch
import torch.nn as nn


def compute_signed_log_scalar(val: Union[int, float, str]) -> float:
    """Computes signed continuous log-magnitude: v = sign(x) * (1 + log10(|x| + 1)).

    For 0, returns 0.0.
    Handles unbounded integers without overflow.
    """
    if isinstance(val, str):
        val = int(val)

    if val == 0:
        return 0.0

    sign = 1.0 if val > 0 else -1.0
    abs_val = abs(val)

    # For standard numeric range
    if isinstance(abs_val, int) and abs_val > 10**300:
        # Approximate log10 via length of integer string
        num_str = str(abs_val)
        n_digits = len(num_str)
        top_digits = float(num_str[:15]) / (10**14)
        log_val = (n_digits - 1) + math.log10(top_digits)
    else:
        log_val = math.log10(float(abs_val) + 1.0)

    return float(sign * (1.0 + log_val))


class MagnitudeStream(nn.Module):
    """Encodes integer terms using signed log-magnitude continuous scalar projection."""

    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(1, d_ff, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, dtype=torch.float32),
            nn.LayerNorm(d_model, dtype=torch.float32),
        )

    def forward_scalars(self, scalar_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass taking precomputed scalar tensor of shape (batch, seq_len, 1) or (batch, seq_len)."""
        if scalar_tensor.dim() == 2:
            scalar_tensor = scalar_tensor.unsqueeze(-1)
        scalar_tensor = scalar_tensor.to(dtype=torch.float32)
        return cast(torch.Tensor, self.mlp(scalar_tensor))

    def forward(self, sequences: Sequence[Sequence[Union[int, float]]], device: Optional[torch.device] = None) -> torch.Tensor:
        """Processes raw batch of sequences of integers/floats."""
        batch_size = len(sequences)
        max_len = max((len(seq) for seq in sequences), default=0)

        scalars = torch.zeros((batch_size, max_len, 1), dtype=torch.float32, device=device)
        for b_idx, seq in enumerate(sequences):
            for t_idx, term in enumerate(seq):
                scalars[b_idx, t_idx, 0] = compute_signed_log_scalar(term)

        return cast(torch.Tensor, self.mlp(scalars))
