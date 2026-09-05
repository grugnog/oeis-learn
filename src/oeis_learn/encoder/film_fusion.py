"""Hierarchical Two-Stage Feature-wise Linear Modulation (FiLM) Fusion Block."""

from __future__ import annotations

from typing import cast
import torch
import torch.nn as nn


class HierarchicalFilmFusion(nn.Module):
    """Fuses Tri-Stream continuous representations via two sequential FiLM stages:

    Stage 1: S2 (modulo spectrum) modulates S1 (magnitude) -> H_12
    Stage 2: S3 (differences & p-adics) modulates H_12 -> Z
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Film generator 1: maps S2 into (gamma1, beta1)
        self.film1 = nn.Linear(d_model, 2 * d_model, dtype=torch.float32)

        # Film generator 2: maps S3 into (gamma2, beta2)
        self.film2 = nn.Linear(d_model, 2 * d_model, dtype=torch.float32)

        self.layer_norm = nn.LayerNorm(d_model, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor) -> torch.Tensor:
        """Fuses stream representations.

        Args:
            s1: Magnitude stream embedding (batch, seq_len, d_model) in FP32
            s2: Modulo spectrum embedding (batch, seq_len, d_model) in FP32
            s3: Difference & padic embedding (batch, seq_len, d_model) in FP32

        Returns:
            Unified continuous representation Z (batch, seq_len, d_model) in FP32
        """
        # Enforce FP32
        s1 = s1.to(dtype=torch.float32)
        s2 = s2.to(dtype=torch.float32)
        s3 = s3.to(dtype=torch.float32)

        # Stage 1: S2 modulates S1
        params1 = self.film1(s2)
        gamma1, beta1 = torch.chunk(params1, 2, dim=-1)
        # Use 1 + gamma1 for residual-like multiplicative scaling
        h12 = (1.0 + gamma1) * s1 + beta1

        # Stage 2: S3 modulates H12
        params2 = self.film2(s3)
        gamma2, beta2 = torch.chunk(params2, 2, dim=-1)
        z = (1.0 + gamma2) * h12 + beta2

        z = self.layer_norm(z)
        z = self.dropout(z)
        return cast(torch.Tensor, z)
