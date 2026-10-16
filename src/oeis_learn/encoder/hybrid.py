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
