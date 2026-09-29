"""Unified Tri-Stream Continuous Neural Encoder (Strict FP32)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Union
import torch
import torch.nn as nn
from oeis_learn.encoder.difference_stream import DifferenceStream, DifferenceStreamEncoder
from oeis_learn.encoder.film_fusion import HierarchicalFilmFusion
from oeis_learn.encoder.heads import SummaryRegressionHeads, TriStreamPredictionHeads
from oeis_learn.encoder.magnitude_stream import MagnitudeStream
from oeis_learn.encoder.modulo_stream import ModuloSpectrumStream, ModuloStreamEncoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding in strict FP32."""

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        res: torch.Tensor = self.dropout(x)
        return res


class TriStreamEncoder(nn.Module):
    """Tri-Stream Continuous Neural Encoder combining S1, S2, and S3 streams via Hierarchical FiLM

    and bidirectional Transformer encoding in strict FP32 precision.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        primes: Optional[Sequence[int]] = None,
        max_valuation: int = 16,
        moduli_count: int = 16,
        base_moduli: Optional[Sequence[int]] = None,
        use_film: bool = True,
        enable_summary_tokens: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_film = use_film
        self.enable_summary_tokens = enable_summary_tokens

        # Streams
        self.s1_magnitude = MagnitudeStream(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.s2_modulo = ModuloStreamEncoder(d_model=d_model, primes=primes, dropout=dropout)
        self.s3_diff_padic = DifferenceStreamEncoder(
            d_model=d_model, primes=primes, max_valuation=max_valuation, dropout=dropout
        )

        # Direct linear concatenation projection (Phase 4) or Hierarchical FiLM Fusion (legacy)
        self.concat_projection = nn.Sequential(
            nn.Linear(d_model * 3, d_model, dtype=torch.float32),
            nn.LayerNorm(d_model, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = HierarchicalFilmFusion(d_model=d_model, dropout=dropout)

        # Learnable global summary tokens: [z_affine, z_geom]
        if self.enable_summary_tokens:
            self.summary_tokens = nn.Parameter(torch.randn(1, 2, d_model, dtype=torch.float32) * 0.02)
        else:
            self.summary_tokens = None

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        # Bidirectional Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dtype=torch.float32,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.final_norm = nn.LayerNorm(d_model, dtype=torch.float32)

        # Auxiliary heads
        self.aux_heads = TriStreamPredictionHeads(d_model=d_model)
        self.summary_heads = SummaryRegressionHeads(d_model=d_model)

    def forward_from_sequences(
        self,
        sequences: Sequence[Sequence[Union[int, float]]],
        pad_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encodes raw integer sequences directly."""
        s1 = self.s1_magnitude(sequences, device=device)
        s2 = self.s2_modulo(sequences, device=device)
        s3 = self.s3_diff_padic(sequences, device=device)

        return self.forward(s1, s2, s3, pad_mask=pad_mask)

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        s3: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass enforcing strict FP32 precision and zero NaN/Inf check."""
        # Assert FP32 precision
        assert s1.dtype == torch.float32, f"s1 must be float32, got {s1.dtype}"
        assert s2.dtype == torch.float32, f"s2 must be float32, got {s2.dtype}"
        assert s3.dtype == torch.float32, f"s3 must be float32, got {s3.dtype}"

        batch_size = s1.size(0)

        # Phase 4 Direct linear concatenation or legacy FiLM fusion
        if self.use_film:
            fused = self.fusion(s1, s2, s3)
        else:
            cat_features = torch.cat([s1, s2, s3], dim=-1)  # (batch, seq_len, 3 * d_model)
            fused = self.concat_projection(cat_features)    # (batch, seq_len, d_model)

        # Prepend learnable summary tokens: [z_affine, z_geom, s_0, ..., s_19]
        if self.enable_summary_tokens and self.summary_tokens is not None:
            sum_tokens = self.summary_tokens.expand(batch_size, -1, -1)
            fused = torch.cat([sum_tokens, fused], dim=1)  # (batch, 2 + seq_len, d_model)
            if pad_mask is not None:
                false_prefix = torch.zeros((batch_size, 2), dtype=torch.bool, device=pad_mask.device)
                pad_mask = torch.cat([false_prefix, pad_mask], dim=1)

        # Add positional encoding
        embedded = self.pos_encoder(fused)

        # Transformer Encoding
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=pad_mask)
        encoded = self.final_norm(encoded)

        # Sanity check for numerical stability
        if torch.isnan(encoded).any() or torch.isinf(encoded).any():
            raise FloatingPointError("TriStreamEncoder produced NaN or Inf in output activations.")

        from typing import cast
        return cast(torch.Tensor, encoded)

        # Hierarchical FiLM fusion
        fused = self.fusion(s1, s2, s3)

        # Add positional encoding
        embedded = self.pos_encoder(fused)

        # Transformer Encoding
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=pad_mask)
        encoded = self.final_norm(encoded)

        # Sanity check for numerical stability
        if torch.isnan(encoded).any() or torch.isinf(encoded).any():
            raise FloatingPointError("TriStreamEncoder produced NaN or Inf in output activations.")

        from typing import cast
        return cast(torch.Tensor, encoded)
