"""Autoregressive Transformer Decoder for WebAssembly Text generation."""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn
from oeis_learn.decoder.wat_grammar import PAD_ID, VOCAB_SIZE


class PositionalEncodingDecoder(nn.Module):
    """Sinusoidal positional encoding for decoder tokens."""

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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        res: torch.Tensor = self.dropout(x)
        return res


class WatTransformerDecoder(nn.Module):
    """Transformer Decoder generating WAT program tokens conditioned on encoded representations Z."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_heads: int = 4,
        n_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = PAD_ID,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx, dtype=torch.float32)
        self.pos_encoder = PositionalEncodingDecoder(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dtype=torch.float32,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        self.final_norm = nn.LayerNorm(d_model, dtype=torch.float32)
        self.lm_head = nn.Linear(d_model, vocab_size, dtype=torch.float32)

    def generate_causal_mask(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generates standard lower-triangular causal attention mask."""
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask

    def project_logits_chunked(self, hidden: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        """Projects hidden states into vocabulary logits using mini-chunk projections for VRAM bounding."""
        c_size = chunk_size or self.chunk_size
        if hidden.size(1) <= c_size:
            return self.lm_head(hidden)

        chunks = []
        for i in range(0, hidden.size(1), c_size):
            h_chunk = hidden[:, i : i + c_size]
            chunks.append(self.lm_head(h_chunk))
        return torch.cat(chunks, dim=1)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass computing token logits across sequence.

        Args:
            tgt_tokens: (batch, tgt_len) integer token IDs
            memory: (batch, src_len, d_model) encoded latent representations Z
            tgt_mask: Optional causal attention mask
            tgt_key_padding_mask: Optional key padding mask (auto-derived if None)
            memory_key_padding_mask: Optional encoder memory key padding mask
        """
        tgt_len = tgt_tokens.size(1)
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(tgt_len, device=tgt_tokens.device)

        # Enforce exact key padding mask to prevent attention & gradient diffusion across PAD tokens
        if tgt_key_padding_mask is None and (tgt_tokens == self.pad_idx).any():
            tgt_key_padding_mask = (tgt_tokens == self.pad_idx)

        embed = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)
        embed = self.pos_encoder(embed)

        hidden = self.transformer_decoder(
            tgt=embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        hidden = self.final_norm(hidden)
        logits = self.project_logits_chunked(hidden)
        from typing import cast
        return cast(torch.Tensor, logits)
