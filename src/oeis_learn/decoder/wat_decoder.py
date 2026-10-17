"""Autoregressive Transformer Decoder for WebAssembly Text generation."""

from __future__ import annotations

import math
import json
from typing import Optional

import torch
import torch.nn as nn

from oeis_learn.decoder.wat_grammar import PAD_ID, VOCAB_SIZE


class PositionalEncodingDecoder(nn.Module):
    """Sinusoidal positional encoding for decoder tokens with dynamic length headroom."""

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self._build_pe(max_len)

    def _build_pe(self, length: int, device: Optional[torch.device] = None) -> None:
        target_device = device or torch.device("cpu")
        position = torch.arange(0, length, dtype=torch.float32, device=target_device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=target_device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model, dtype=torch.float32, device=target_device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def _extend_pe(self, target_len: int, device: torch.device) -> None:
        current_len = self.pe.size(1) if hasattr(self, "pe") else 0
        new_len = max(target_len, current_len * 2, 512)
        self._build_pe(new_len, device=device)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        pe_key = prefix + "pe"
        if pe_key in state_dict:
            loaded = state_dict[pe_key]
            if loaded.ndim == 3 and loaded.size(0) == 1 and loaded.size(2) == self.d_model:
                loaded_len = loaded.size(1)
                if loaded_len != self.pe.size(1):
                    # Checkpoints are deserialized on CPU. Resize on the module's
                    # device, retaining its dtype and the saved values, so loading
                    # cannot strand a positional buffer away from GPU parameters.
                    if loaded_len > self.pe.size(1):
                        dtype = self.pe.dtype
                        self._build_pe(loaded_len, device=self.pe.device)
                        self.pe = self.pe.to(dtype=dtype)
                    extended = self.pe.detach().clone()
                    extended[:, :loaded_len].copy_(loaded)
                    state_dict[pe_key] = extended
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len, device=x.device)
        x = x + self.pe[:, :seq_len]
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
        logit_cap_threshold: Optional[float] = 30.0,
        codec_sha256: Optional[str] = None,
        position_style: str = "sinusoidal",
        attention_backend: str = "sdpa",
        token_scale: float = 1.0,
        position_scale: float = 1.0,
        input_version: Optional[str] = None,
    ):
        super().__init__()
        from oeis_learn.fresh.bootstrap_contract import decoder_controls

        selected = decoder_controls(
            token_scale=token_scale, position_scale=position_scale, dropout=dropout
        )
        self.token_scale = selected["token_scale"]
        self.position_scale = selected["position_scale"]
        self.input_version = input_version
        if input_version not in (None, "decoder-inputs/v1"):
            raise ValueError("unknown decoder input version")
        if input_version is None and (self.token_scale != 1 or self.position_scale != 1):
            raise ValueError("non-default decoder scales require decoder-inputs/v1")
        if input_version is not None:
            if position_style != "sinusoidal":
                raise ValueError("decoder-inputs/v1 requires sinusoidal positions")
            identity = json.dumps(dict(version=input_version, **selected), sort_keys=True).encode()
            self.register_buffer("_input_identity", torch.tensor(list(identity), dtype=torch.uint8))
        self.codec_sha256 = codec_sha256
        if codec_sha256 is not None:
            from oeis_learn.decoder.program_codec import (
                FOUNDATION_VOCAB_SIZE,
                codec_digest,
            )
            from oeis_learn.decoder.program_codec import (
                PAD_ID as BODY_PAD,
            )

            if (
                codec_sha256 != codec_digest()
                or vocab_size != FOUNDATION_VOCAB_SIZE
                or pad_idx != BODY_PAD
            ):
                raise ValueError("incompatible foundation codec configuration")
            self.register_buffer(
                "_codec_identity", torch.tensor(list(codec_sha256.encode()), dtype=torch.uint8)
            )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.chunk_size = chunk_size
        self.logit_cap_threshold = logit_cap_threshold
        if position_style not in ("sinusoidal", "rope", "prefix_ancestor"):
            raise ValueError("unknown decoder position style")
        if attention_backend not in ("eager", "sdpa"):
            raise ValueError("unknown attention backend")
        self.position_style, self.attention_backend = position_style, attention_backend
        self.ancestor_embedding = (
            nn.Embedding(7, d_model, padding_idx=0) if position_style == "prefix_ancestor" else None
        )

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_idx, dtype=torch.float32
        )
        self.pos_encoder = PositionalEncodingDecoder(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

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

    def generate_causal_mask(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generates standard lower-triangular causal attention mask."""
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask

    def project_logits_chunked(
        self, hidden: torch.Tensor, chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """Projects hidden states into vocabulary logits using mini-chunk projections for VRAM bounding."""
        c_size = chunk_size or self.chunk_size
        if hidden.size(1) <= c_size:
            return self.lm_head(hidden)

        chunks = []
        for i in range(0, hidden.size(1), c_size):
            h_chunk = hidden[:, i : i + c_size]
            chunks.append(self.lm_head(h_chunk))
        return torch.cat(chunks, dim=1)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        input_key = prefix + "_input_identity"
        if self.input_version is not None:
            supplied = state_dict.get(input_key)
            if supplied is None or not torch.equal(supplied.cpu(), self._input_identity.cpu()):
                error_msgs.append("missing or incompatible decoder input identity")
        elif input_key in state_dict:
            error_msgs.append("versioned decoder inputs cannot load into a legacy decoder")
        key = prefix + "_codec_identity"
        if self.codec_sha256 is not None:
            supplied = state_dict.get(key)
            if supplied is None or not torch.equal(supplied.cpu(), self._codec_identity.cpu()):
                error_msgs.append(
                    "missing or incompatible foundation codec identity; legacy weights forbidden"
                )
        elif key in state_dict:
            error_msgs.append("foundation weights cannot load into a legacy decoder")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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
            tgt_key_padding_mask = tgt_tokens == self.pad_idx

        embed = self.embed_tokens(tgt_tokens)
        positions = torch.arange(tgt_len, device=tgt_tokens.device)[None, :].expand(
            tgt_tokens.size(0), -1
        )
        embed = self.position_embeddings(embed, positions, tgt_tokens)

        if self.position_style == "rope":
            hidden = self._rotary_forward(
                embed, memory, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask
            )
        else:
            hidden = self.transformer_decoder(
                tgt=embed,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        hidden = self.final_norm(hidden)
        logits = self.project_logits_chunked(hidden)

        # Hyperbolic tangent logit soft-capping (C_cap = 30.0) to stabilize vocabulary expansion
        logits = self.cap_logits(logits)

        from typing import cast

        return cast(torch.Tensor, logits)

    def cap_logits(self, logits):
        if self.logit_cap_threshold is not None and self.logit_cap_threshold > 0:
            cap = float(self.logit_cap_threshold)
            return cap * torch.tanh(logits / cap)
        return logits

    def embed_tokens(self, tokens):
        """Shared full/cached token scaling; 1.0 retains the historical sqrt(d) factor."""
        return self.token_embedding(tokens) * (math.sqrt(self.d_model) * self.token_scale)

    def position_embeddings(self, hidden, positions, tokens):
        if self.position_style == "rope":
            return self.pos_encoder.dropout(hidden)
        maximum = tokens.size(1)
        if maximum > self.pos_encoder.pe.size(1):
            self.pos_encoder._extend_pe(maximum, hidden.device)
        hidden = hidden + self.pos_encoder.pe[0][positions].to(hidden.dtype) * self.position_scale
        if self.ancestor_embedding is not None:
            from oeis_learn.decoder.prefix_positions import ancestor_ids
            from oeis_learn.decoder.program_codec import ID_TO_TOKEN as BODY_WORDS
            from oeis_learn.decoder.wat_grammar import ID_TO_TOKEN as LEGACY_WORDS

            words = BODY_WORDS if self.codec_sha256 is not None else LEGACY_WORDS
            ancestors = ancestor_ids(tokens, words)
            selected = ancestors.gather(1, positions[..., None].expand(-1, -1, ancestors.size(-1)))
            hidden = hidden + self.ancestor_embedding(selected).sum(-2)
        return self.pos_encoder.dropout(hidden)

    def _rotary_forward(self, hidden, memory, padding, memory_padding, mask):
        from oeis_learn.decoder.inference_cache import attention_output, project, rotary

        batch, length, _ = hidden.shape
        positions = torch.arange(length, device=hidden.device)[None, :].expand(batch, -1)
        allowed = (mask == 0 if mask.dtype != torch.bool else ~mask)[None, None].expand(
            batch, -1, -1, -1
        )
        if padding is not None:
            allowed = allowed & ~padding[:, None, None, :]
        cross_allowed = torch.ones(
            (batch, 1, 1, memory.size(1)), device=hidden.device, dtype=torch.bool
        )
        if memory_padding is not None:
            cross_allowed &= ~memory_padding[:, None, None, :]
        for layer in self.transformer_decoder.layers:
            value = layer.norm1(hidden)
            q, k, v = (project(layer.self_attn, value, i) for i in range(3))
            hidden = hidden + layer.dropout1(
                attention_output(
                    layer.self_attn,
                    rotary(q, positions),
                    rotary(k, positions),
                    v,
                    allowed,
                    self.attention_backend,
                )
            )
            q = project(layer.multihead_attn, layer.norm2(hidden), 0)
            hidden = hidden + layer.dropout2(
                attention_output(
                    layer.multihead_attn,
                    q,
                    project(layer.multihead_attn, memory, 1),
                    project(layer.multihead_attn, memory, 2),
                    cross_allowed,
                    self.attention_backend,
                )
            )
            hidden = hidden + layer.dropout3(
                layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(hidden)))))
            )
        return hidden

    def prefill(
        self,
        tokens,
        memory,
        *,
        capacity=1024,
        lengths=None,
        memory_key_padding_mask=None,
        backend=None,
    ):
        from oeis_learn.decoder.inference_cache import create_cache, decode_step

        if torch.is_grad_enabled():
            raise ValueError("prefill is inference-only")
        if tokens.ndim != 2 or tokens.size(0) != memory.size(0) or tokens.size(1) > capacity:
            raise ValueError("invalid prefill shape/capacity")
        sizes = tokens.ne(self.pad_idx).sum(-1).tolist() if lengths is None else list(lengths)
        if len(sizes) != tokens.size(0) or any(
            type(n) is not int or not 1 <= n <= tokens.size(1) for n in sizes
        ):
            raise ValueError("nonempty right-padded prefill required")
        for row, size in enumerate(sizes):
            if (
                tokens[row, :size].eq(self.pad_idx).any()
                or tokens[row, size:].ne(self.pad_idx).any()
            ):
                raise ValueError("prefill must be right padded")
        cache = create_cache(
            self, memory, capacity, memory_key_padding_mask, backend or self.attention_backend
        )
        outputs = [
            decode_step(self, tokens[:, i], cache, active=[i < n for n in sizes])
            for i in range(tokens.size(1))
        ]
        return torch.cat(outputs, 1), cache

    def decode_step(self, tokens, cache, **kwargs):
        from oeis_learn.decoder.inference_cache import decode_step

        return decode_step(self, tokens, cache, **kwargs)
