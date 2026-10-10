"""Bounded inference-only K/V state for the existing Transformer layers.

Parameters stay in WatTransformerDecoder; caches never enter its state_dict.
Normal in-place optimizer/load changes invalidate a cache via tensor versions.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def model_signature(model):
    return tuple((id(p), p._version, str(p.device), str(p.dtype)) for p in model.parameters())


def tensor_signature(value):
    if value.is_inference():
        raise ValueError(
            "cache prompt needs a versioned tensor; construct it under no_grad, outside inference_mode"
        )
    return (id(value), value._version, tuple(value.shape), str(value.device), str(value.dtype))


def split_heads(value, heads):
    return value.reshape(value.size(0), value.size(1), heads, -1).transpose(1, 2)


def project(attention, value, part):
    width = attention.embed_dim
    bias = attention.in_proj_bias
    return split_heads(
        F.linear(
            value,
            attention.in_proj_weight[part * width : (part + 1) * width],
            None if bias is None else bias[part * width : (part + 1) * width],
        ),
        attention.num_heads,
    )


def rotary(value, positions):
    width = value.size(-1)
    if width % 2:
        raise ValueError("RoPE requires an even attention head width")
    frequency = torch.exp(
        torch.arange(0, width, 2, device=value.device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    angles = positions[:, None, :, None].to(torch.float32) * frequency
    even, odd = value[..., 0::2], value[..., 1::2]
    result = torch.stack(
        (even * angles.cos() - odd * angles.sin(), even * angles.sin() + odd * angles.cos()), -1
    )
    return result.flatten(-2).to(value.dtype)


def attention_output(attention, q, k, v, allowed, backend):
    if backend == "sdpa":
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=attention.dropout if attention.training else 0.0
        )
    elif backend == "eager":
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        output = (
            F.dropout(
                torch.softmax(scores.masked_fill(~allowed, float("-inf")), -1),
                attention.dropout,
                attention.training,
            )
            @ v
        )
    else:
        raise ValueError("unknown attention backend")
    output = output.transpose(1, 2).reshape(q.size(0), q.size(2), -1)
    return attention.out_proj(output)


@dataclass
class LayerCache:
    keys: torch.Tensor
    values: torch.Tensor
    cross_keys: torch.Tensor
    cross_values: torch.Tensor


@dataclass
class DecoderCache:
    owner: int
    signature: tuple
    memory: torch.Tensor
    memory_signature: tuple
    memory_padding: torch.Tensor
    layers: list[LayerCache]
    lengths: list[int]
    capacity: int
    backend: str
    position: str
    tokens: torch.Tensor
    finished: list[bool]
    valid: bool = True

    def reset(self):
        self.valid = False
        self.layers.clear()

    def reorder(self, indices):
        if not self.valid:
            raise ValueError("invalidated cache")
        order = list(indices)
        if not order or any(type(i) is not int or not 0 <= i < len(self.lengths) for i in order):
            raise ValueError("invalid beam reorder")
        index = torch.tensor(order, device=self.memory.device, dtype=torch.long)
        self.memory = self.memory.index_select(0, index)
        self.memory_signature = tensor_signature(self.memory)
        self.memory_padding = self.memory_padding.index_select(0, index)
        self.tokens = self.tokens.index_select(0, index)
        for layer in self.layers:
            for name in ("keys", "values", "cross_keys", "cross_values"):
                setattr(layer, name, getattr(layer, name).index_select(0, index))
        self.lengths = [self.lengths[i] for i in order]
        self.finished = [self.finished[i] for i in order]
        return self


def create_cache(model, memory, capacity, memory_padding=None, backend="sdpa"):
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("K/V caches require eval mode; teacher forcing remains uncached")
    if type(capacity) is not int or not 1 <= capacity <= 1024:
        raise ValueError("cache capacity must be in 1..1024")
    if memory.ndim != 3 or memory.size(2) != model.d_model:
        raise ValueError("invalid encoded prompt")
    if backend not in ("sdpa", "eager"):
        raise ValueError("unknown attention backend")
    batch = memory.size(0)
    padding = (
        torch.zeros(memory.shape[:2], device=memory.device, dtype=torch.bool)
        if memory_padding is None
        else memory_padding.clone()
    )
    if padding.shape != memory.shape[:2] or padding.dtype != torch.bool or padding.all(-1).any():
        raise ValueError("each prompt needs at least one unmasked term")
    layers = []
    for layer in model.transformer_decoder.layers:
        heads = layer.self_attn.num_heads
        shape = (batch, heads, capacity, model.d_model // heads)
        layers.append(
            LayerCache(
                torch.zeros(shape, device=memory.device, dtype=memory.dtype),
                torch.zeros(shape, device=memory.device, dtype=memory.dtype),
                project(layer.multihead_attn, memory, 1),
                project(layer.multihead_attn, memory, 2),
            )
        )
    return DecoderCache(
        id(model),
        model_signature(model),
        memory,
        tensor_signature(memory),
        padding,
        layers,
        [0] * batch,
        capacity,
        backend,
        model.position_style,
        torch.full((batch, capacity), model.pad_idx, device=memory.device, dtype=torch.long),
        [False] * batch,
    )


def decode_step(model, tokens, cache, *, active=None, memory=None):
    """Append one token per active row; completed rows never acquire new keys."""
    if (
        not cache.valid
        or cache.owner != id(model)
        or cache.signature != model_signature(model)
        or cache.memory_signature != tensor_signature(cache.memory)
        or (memory is not None and tensor_signature(memory) != cache.memory_signature)
        or cache.position != model.position_style
        or model.training
        or any(module.training for module in model.modules())
    ):
        cache.reset()
        raise ValueError("stale model/prompt/device/dtype/position or training cache")
    if torch.is_grad_enabled():
        raise ValueError(
            "decode_step is inference-only; use no_grad with a versioned prompt tensor"
        )
    if tokens.ndim == 1:
        tokens = tokens[:, None]
    if tokens.shape != (len(cache.lengths), 1) or tokens.device != cache.memory.device:
        raise ValueError("one token required per cached row on the prompt device")
    enabled = [not done for done in cache.finished] if active is None else list(active)
    if len(enabled) != len(cache.lengths) or any(type(a) is not bool for a in enabled):
        raise ValueError("exact per-row active flags required")
    enabled = [a and not done for a, done in zip(enabled, cache.finished)]
    if any(a and length >= cache.capacity for a, length in zip(enabled, cache.lengths)):
        raise ValueError("cache capacity exhausted")
    positions = torch.tensor(cache.lengths, device=tokens.device)[:, None]
    safe_positions = positions.clamp(max=cache.capacity - 1)
    rows = torch.arange(tokens.size(0), device=tokens.device)
    enabled_tensor = torch.tensor(enabled, device=tokens.device)
    cache.tokens[rows[enabled_tensor], safe_positions[:, 0][enabled_tensor]] = tokens[:, 0][
        enabled_tensor
    ]
    hidden = model.token_embedding(tokens) * math.sqrt(model.d_model)
    hidden = model.position_embeddings(hidden, safe_positions, cache.tokens)
    used = max([length + int(a) for length, a in zip(cache.lengths, enabled)], default=1)
    used = max(1, used)
    for layer, state in zip(model.transformer_decoder.layers, cache.layers):
        normalized = layer.norm1(hidden)
        q, k, v = (project(layer.self_attn, normalized, i) for i in range(3))
        if model.position_style == "rope":
            q, k = rotary(q, positions), rotary(k, positions)
        state.keys[rows[enabled_tensor], :, safe_positions[:, 0][enabled_tensor], :] = k[
            enabled_tensor, :, 0, :
        ]
        state.values[rows[enabled_tensor], :, safe_positions[:, 0][enabled_tensor], :] = v[
            enabled_tensor, :, 0, :
        ]
        lengths = torch.tensor(
            [length + int(a) for length, a in zip(cache.lengths, enabled)], device=tokens.device
        )
        allowed = torch.arange(used, device=tokens.device)[None, :] < lengths[:, None]
        # Inactive empty rows get a harmless zero key; their logits are masked below.
        allowed[:, 0] |= lengths == 0
        hidden = hidden + layer.dropout1(
            attention_output(
                layer.self_attn,
                q,
                state.keys[:, :, :used],
                state.values[:, :, :used],
                allowed[:, None, None, :],
                cache.backend,
            )
        )
        q = project(layer.multihead_attn, layer.norm2(hidden), 0)
        hidden = hidden + layer.dropout2(
            attention_output(
                layer.multihead_attn,
                q,
                state.cross_keys,
                state.cross_values,
                (~cache.memory_padding)[:, None, None, :],
                cache.backend,
            )
        )
        hidden = hidden + layer.dropout3(
            layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(hidden)))))
        )
    for row, a in enumerate(enabled):
        cache.lengths[row] += int(a)
    logits = model.cap_logits(model.lm_head(model.final_norm(hidden)))
    return logits.masked_fill(~enabled_tensor[:, None, None], 0)
