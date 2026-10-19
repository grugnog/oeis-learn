"""Historical numeric features supplemented by exact signed-byte inputs.

The source representation is exact; neither learned projection is claimed injective.
Only one bidirectional Transformer is used, as in the matched controls.
"""

import torch
from torch import nn
from .integer_bytes import encode_sequences
from .tri_stream_encoder import TriStreamEncoder


class HybridEncoder(TriStreamEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.byte_embedding = nn.Embedding(256, 4)
        self.byte_projection = nn.Linear(65 * 4 + 2, self.d_model)
        self.byte_gate = nn.Parameter(torch.full((self.d_model,), -2.0))

    def forward_from_sequences(self, sequences, pad_mask=None, device=None):
        actual = next(self.parameters()).device
        batch = encode_sequences(sequences, device=actual)
        if pad_mask is not None and not torch.equal(pad_mask, batch.padding):
            raise ValueError("padding must match visible lengths")
        features = torch.cat(
            [
                self.byte_embedding(batch.magnitudes).flatten(-2),
                self.byte_embedding(batch.differences).flatten(-2),
                batch.signs[..., None].float(),
                batch.difference_signs[..., None].float(),
            ],
            dim=-1,
        )
        s1 = self.s1_magnitude(sequences, device=actual)
        s2 = self.s2_modulo(sequences, device=actual)
        s3 = self.s3_diff_padic(sequences, device=actual)
        fused = (
            self.fusion(s1, s2, s3)
            if self.use_film
            else self.concat_projection(torch.cat([s1, s2, s3], dim=-1))
        )
        fused = fused + self.byte_gate.sigmoid() * self.byte_projection(features)
        hidden = self.transformer_encoder(
            self.pos_encoder(fused), src_key_padding_mask=batch.padding
        )
        return self.final_norm(hidden).masked_fill(batch.padding[..., None], 0)


class ChunkHybridEncoder(TriStreamEncoder):
    """Explicit width-extensible profile; historical HybridEncoder stays intact."""

    def __init__(self, *, dropout=0.0, group_mask=None, ablation_mode="post_training", **kwargs):
        from .large_integer_features import PRIMES
        from .integer_chunks import ChunkEncoder

        fixed = dict(
            d_model=256,
            n_heads=4,
            n_encoder_layers=4,
            d_ff=1024,
            primes=PRIMES,
            base_moduli=PRIMES,
            max_valuation=32,
            use_film=True,
            enable_summary_tokens=False,
            input_profile="hybrid-chunks/v1",
        )
        for key, value in kwargs.items():
            if key in fixed and value != fixed[key]:
                raise ValueError(f"hybrid-chunks/v1 requires frozen {key}")
        fixed.update(kwargs)
        super().__init__(dropout=dropout, **fixed)
        self.chunk_encoder = ChunkEncoder(dropout=dropout)
        self.byte_gate = nn.Parameter(torch.full((256,), -2.0))
        from .ablation import InputMask

        self.group_mask = group_mask or InputMask()
        if not isinstance(self.group_mask, InputMask) or ablation_mode not in {
            "post_training",
            "retrained",
        }:
            raise ValueError("explicit ablation mask/mode required")
        self.ablation_mode = ablation_mode
        self.register_buffer(
            "input_group_mask", torch.tensor(self.group_mask.bits(), dtype=torch.bool)
        )
        self.register_buffer("input_ablation_mode", torch.tensor(ablation_mode == "retrained"))
        if ablation_mode == "retrained":
            branches = (
                (self.group_mask.magnitude, [self.s1_magnitude]),
                (self.group_mask.modular, [self.s2_modulo, self.fusion.film1]),
                (self.group_mask.difference, [self.s3_diff_padic, self.fusion.film2]),
                (self.group_mask.exact_bytes, [self.chunk_encoder]),
            )
            for enabled, modules in branches:
                if not enabled:
                    for module in modules:
                        module.requires_grad_(False)
            if not self.group_mask.exact_bytes:
                self.byte_gate.requires_grad_(False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ("input_group_mask", "input_ablation_mode"):
            expected = getattr(self, name)
            incoming = state_dict.get(prefix + name)
            if incoming is not None and not torch.equal(incoming.cpu(), expected.cpu()):
                raise RuntimeError("input mask/mode checkpoint identity mismatch")
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ablation_report(self):
        inactive = set()
        for enabled, modules in (
            (self.group_mask.magnitude, [self.s1_magnitude]),
            (self.group_mask.modular, [self.s2_modulo, self.fusion.film1]),
            (self.group_mask.difference, [self.s3_diff_padic, self.fusion.film2]),
            (self.group_mask.exact_bytes, [self.chunk_encoder]),
        ):
            if not enabled:
                inactive.update(id(p) for m in modules for p in m.parameters())
        if not self.group_mask.exact_bytes:
            inactive.add(id(self.byte_gate))
        return dict(
            mask_sha256=self.group_mask.sha256,
            mode=self.ablation_mode,
            total_parameters=sum(p.numel() for p in self.parameters()),
            active_parameters=sum(p.numel() for p in self.parameters() if id(p) not in inactive),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward_from_sequences(self, sequences, pad_mask=None, device=None):
        from .integer_chunks import encode_chunks

        actual = next(self.parameters()).device
        if device is not None and torch.device(device) != actual:
            raise ValueError("explicit device differs from model device")
        batch = encode_chunks(sequences, device=actual)
        if pad_mask is not None and not torch.equal(pad_mask, batch.padding):
            raise ValueError("padding must match exact visible lengths")
        s1 = self.s1_magnitude(sequences, device=actual)
        s2 = self.s2_modulo(sequences, device=actual)
        s3 = self.s3_diff_padic(sequences, device=actual)
        fused = self.fusion(s1, s2, s3, group_mask=self.group_mask)
        if self.group_mask.exact_bytes:
            fused = fused + self.byte_gate.sigmoid() * self.chunk_encoder(batch)
        hidden = self.transformer_encoder(
            self.pos_encoder(fused), src_key_padding_mask=batch.padding
        )
        result = self.final_norm(hidden).masked_fill(batch.padding[..., None], 0)
        if not torch.isfinite(result).all():
            raise FloatingPointError("nonfinite bigint input encoding")
        return result
