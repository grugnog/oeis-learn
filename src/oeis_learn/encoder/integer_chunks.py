"""Shared eight-byte chunk encoder; parameter shapes never depend on width."""

from dataclasses import dataclass
import math
import torch
from torch import nn
from .large_integer_features import exact


@dataclass(frozen=True)
class ChunkBatch:
    magnitudes: torch.Tensor
    differences: torch.Tensor
    chunk_lengths: torch.Tensor
    difference_chunk_lengths: torch.Tensor
    lengths: torch.Tensor
    difference_lengths: torch.Tensor
    signs: torch.Tensor
    difference_signs: torch.Tensor
    padding: torch.Tensor


def encode_chunks(sequences, *, device=None, max_bits=4096):
    if (
        not sequences
        or len(sequences) > 1024
        or any(not seq or len(seq) > 128 for seq in sequences)
    ):
        raise ValueError("nonempty bounded sequences required")
    if type(max_bits) is not int or not 1 <= max_bits <= 131072:
        raise ValueError("unsupported input bit bound")
    rows = []
    for seq in sequences:
        values = [exact(x, max_bits) for x in seq]
        differences = [0] + [b - a for a, b in zip(values, values[1:])]
        rows.append((values, differences))
    batch = len(rows)
    time = max(len(x) for x, _ in rows)
    largest = max(max(1, (abs(n).bit_length() + 7) // 8) for row in rows for xs in row for n in xs)
    chunks = (largest + 7) // 8
    arrays = []
    for which in (0, 1):
        raw = torch.zeros((batch, time, chunks, 8), dtype=torch.long, device=device)
        chunk_lengths = torch.zeros((batch, time, chunks), dtype=torch.long, device=device)
        lengths = torch.zeros((batch, time), dtype=torch.long, device=device)
        signs = torch.zeros((batch, time), dtype=torch.float32, device=device)
        for b, row in enumerate(rows):
            for t, n in enumerate(row[which]):
                data = abs(n).to_bytes(max(1, (abs(n).bit_length() + 7) // 8), "little")
                lengths[b, t] = len(data)
                signs[b, t] = 0 if n == 0 else (1 if n > 0 else -1)
                for c, start in enumerate(range(0, len(data), 8)):
                    part = list(data[start : start + 8])
                    chunk_lengths[b, t, c] = len(part)
                    raw[b, t, c, : len(part)] = torch.tensor(part, dtype=torch.long, device=device)
        arrays.append((raw, chunk_lengths, lengths, signs))
    a, d = arrays
    padding = a[2] == 0
    return ChunkBatch(a[0], d[0], a[1], d[1], a[2], d[2], a[3], d[3], padding)


class GrowingPosition(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pe", torch.empty(0, 64), persistent=False)

    def forward(self, x):
        length = x.shape[-2]
        if self.pe.shape[0] < length or self.pe.device != x.device or self.pe.dtype != x.dtype:
            size = max(length, 2 * self.pe.shape[0], 8)
            positions = torch.arange(size, device=x.device, dtype=torch.float32)[:, None]
            frequency = torch.exp(
                torch.arange(0, 64, 2, device=x.device, dtype=torch.float32)
                * (-math.log(10000) / 64)
            )
            pe = torch.empty((size, 64), device=x.device, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(positions * frequency)
            pe[:, 1::2] = torch.cos(positions * frequency)
            self.pe = pe.to(dtype=x.dtype)
        return x + self.pe[:length]


class ChunkEncoder(nn.Module):
    def __init__(self, *, dropout=0.0):
        super().__init__()
        self.byte_embedding = nn.Embedding(256, 4)
        self.chunk_projection = nn.Linear(33, 64)
        self.position = GrowingPosition()
        self.transformer = nn.TransformerEncoderLayer(
            64, 4, 256, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.projection = nn.Linear(132, 256)

    def pool(self, raw, lengths):
        shape = raw.shape
        raw = raw.reshape(-1, shape[-2], 8)
        lengths = lengths.reshape(-1, shape[-2])
        byte_mask = torch.arange(8, device=raw.device)[None, None, :] < lengths[..., None]
        embedded = self.byte_embedding(raw) * byte_mask[..., None]
        x = self.chunk_projection(
            torch.cat([embedded.flatten(-2), lengths[..., None].float() / 8], -1)
        )
        padding = lengths == 0
        safe = padding.clone()
        safe[padding.all(-1), 0] = False
        x = self.transformer(self.position(x), src_key_padding_mask=safe)
        x = x.masked_fill(padding[..., None], 0)
        pooled = x.sum(-2) / (~padding).sum(-1, keepdim=True).clamp_min(1)
        return pooled.reshape(*shape[:-2], 64)

    def forward(self, batch):
        value = self.pool(batch.magnitudes, batch.chunk_lengths)
        difference = self.pool(batch.differences, batch.difference_chunk_lengths)
        scalars = torch.stack(
            [
                batch.signs,
                batch.difference_signs,
                torch.log1p(batch.lengths.float()),
                torch.log1p(batch.difference_lengths.float()),
            ],
            -1,
        )
        return self.projection(torch.cat([value, difference, scalars], -1)).masked_fill(
            batch.padding[..., None], 0
        )
