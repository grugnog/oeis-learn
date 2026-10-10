"""Exact source representation before projection; learned embeddings are lossy."""

from dataclasses import dataclass

import torch
from torch import nn

from oeis_learn.experiments.profiles import I256_MIN, I256_MAX


def encode_integer(value, width=32):
    if type(value) is not int or not -(1 << (8 * width - 1)) <= value < (1 << (8 * width - 1)):
        raise ValueError("exact signed integer outside encoding range")
    return int(value < 0), tuple(abs(value).to_bytes(width, "little"))


def decode_integer(sign, magnitude):
    if type(sign) is not int or sign not in (0, 1) or len(magnitude) not in (32, 33):
        raise ValueError("invalid signed-byte encoding")
    if any(type(b) is not int or not 0 <= b <= 255 for b in magnitude):
        raise ValueError("invalid magnitude bytes")
    value = int.from_bytes(bytes(magnitude), "little")
    if sign and value == 0:
        raise ValueError("negative zero forbidden")
    result = -value if sign else value
    encode_integer(result, len(magnitude))
    return result


def valuation_two(value):
    if value == 0:
        return 257  # separate zero tag, not a clipped finite valuation
    return (abs(value) & -abs(value)).bit_length() - 1


@dataclass(frozen=True)
class IntegerBatch:
    signs: torch.Tensor
    magnitudes: torch.Tensor
    difference_signs: torch.Tensor
    differences: torch.Tensor
    residues: torch.Tensor
    valuations: torch.Tensor
    indices: torch.Tensor
    padding: torch.Tensor
    lengths: tuple[int, ...]


def encode_sequences(sequences, *, device=None, max_terms=20):
    if not sequences or any(not 1 <= len(row) <= max_terms for row in sequences):
        raise ValueError("visible sequences must contain 1..20 terms")
    length = max(map(len, sequences))
    signs, mags, ds, dm, residues, vals, padding = [], [], [], [], [], [], []
    for row in sequences:
        if any(type(v) is not int or not I256_MIN <= v <= I256_MAX for v in row):
            raise ValueError("exact signed256 terms required")
        values = [encode_integer(v) for v in row]
        diffs = [encode_integer(0 if i == 0 else v - row[i - 1], 33) for i, v in enumerate(row)]
        pad = length - len(row)
        signs.append([s for s, _ in values] + [0] * pad)
        mags.append([b for _, b in values] + [(0,) * 32] * pad)
        ds.append([s for s, _ in diffs] + [0] * pad)
        dm.append([b for _, b in diffs] + [(0,) * 33] * pad)
        residues.append([[v % m for m in (2, 4, 8, 16)] for v in row] + [[0] * 4] * pad)
        vals.append([valuation_two(v) for v in row] + [257] * pad)
        padding.append([False] * len(row) + [True] * pad)
    tensor = lambda x: torch.tensor(x, device=device, dtype=torch.long)
    return IntegerBatch(
        tensor(signs),
        tensor(mags),
        tensor(ds),
        tensor(dm),
        tensor(residues),
        tensor(vals),
        torch.arange(length, device=device).expand(len(sequences), -1),
        torch.tensor(padding, device=device, dtype=torch.bool),
        tuple(map(len, sequences)),
    )


class IntegerByteEncoder(nn.Module):
    def __init__(
        self, d_model=384, n_heads=6, n_encoder_layers=4, d_ff=1536, dropout=0.0, byte_width=4
    ):
        super().__init__()
        self.d_model = d_model
        self.bytes = nn.Embedding(256, byte_width)
        self.magnitude_projection = nn.Linear(32 * byte_width, d_model)
        self.difference_projection = nn.Linear(33 * byte_width, d_model)
        self.sign = nn.Embedding(2, d_model)
        self.difference_sign = nn.Embedding(2, d_model)
        self.index = nn.Embedding(20, d_model)
        self.length = nn.Embedding(21, d_model)
        self.residues = nn.ModuleList(nn.Embedding(m, d_model) for m in (2, 4, 8, 16))
        self.valuation = nn.Embedding(258, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, "gelu", batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            layer, n_encoder_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward_from_sequences(self, sequences, pad_mask=None, device=None):
        actual_device = next(self.parameters()).device
        if device is not None and torch.device(device) != actual_device:
            raise ValueError("encoder and requested device differ")
        batch = encode_sequences(sequences, device=actual_device)
        if pad_mask is not None and not torch.equal(pad_mask, batch.padding):
            raise ValueError("padding must match actual visible lengths")
        return self.forward(batch)

    def forward(self, batch):
        hidden = self.magnitude_projection(self.bytes(batch.magnitudes).flatten(-2))
        hidden = hidden + self.difference_projection(self.bytes(batch.differences).flatten(-2))
        hidden = hidden + self.sign(batch.signs) + self.difference_sign(batch.difference_signs)
        hidden = hidden + self.index(batch.indices) + self.valuation(batch.valuations)
        lengths = torch.tensor(batch.lengths, device=hidden.device)
        hidden = hidden + self.length(lengths)[:, None]
        for i, embedding in enumerate(self.residues):
            hidden = hidden + embedding(batch.residues[..., i])
        hidden = self.transformer_encoder(hidden, src_key_padding_mask=batch.padding)
        return self.final_norm(hidden).masked_fill(batch.padding[..., None], 0)
