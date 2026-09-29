"""Vocabulary Expansion and Convex Hull Semantic Projection Surgery.

Performs progressive vocabulary surgery on pre-trained Transformer decoders:
- Expands token embedding matrix E_in via convex hull semantic projection
- Calibrates unembedding linear head W_out to match mean pre-trained norm
- Applies hyperbolic tangent logit soft-capping (C_cap = 30.0)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from oeis_learn.decoder.wat_grammar import TOKEN_TO_ID, WAT_VOCABULARY

# Default semantic ancestors mapping for new multi-limb tokens: token -> [(ancestor_token, weight), ...]
SEMANTIC_ANCESTORS: Dict[str, List[Tuple[str, float]]] = {
    "result_i64_x4": [("result", 0.50), ("i64", 0.50)],
    "i256.add": [("i64.add", 0.80), ("func", 0.20)],
    "i256.sub": [("i64.sub", 0.80), ("func", 0.20)],
    "i256.mul_scalar": [("i64.mul", 0.80), ("func", 0.20)],
    "i256.const": [("i64.const", 0.80), ("func", 0.20)],
    "i256.zero": [("0", 0.80), ("i64.const", 0.20)],
    "$a0": [("$a", 0.85), ("i64", 0.15)],
    "$a1": [("$a", 0.70), ("1", 0.15), ("i64", 0.15)],
    "$a2": [("$a", 0.70), ("2", 0.15), ("i64", 0.15)],
    "$a3": [("$a", 0.70), ("3", 0.15), ("i64", 0.15)],
    "$b0": [("$b", 0.85), ("i64", 0.15)],
    "$b1": [("$b", 0.70), ("1", 0.15), ("i64", 0.15)],
    "$b2": [("$b", 0.70), ("2", 0.15), ("i64", 0.15)],
    "$b3": [("$b", 0.70), ("3", 0.15), ("i64", 0.15)],
    "$c0": [("$c", 0.85), ("i64", 0.15)],
    "$c1": [("$c", 0.70), ("1", 0.15), ("i64", 0.15)],
    "$c2": [("$c", 0.70), ("2", 0.15), ("i64", 0.15)],
    "$c3": [("$c", 0.70), ("3", 0.15), ("i64", 0.15)],
    "$d0": [("$d", 0.85), ("i64", 0.15)],
    "$d1": [("$d", 0.70), ("1", 0.15), ("i64", 0.15)],
    "$d2": [("$d", 0.70), ("2", 0.15), ("i64", 0.15)],
    "$d3": [("$d", 0.70), ("3", 0.15), ("i64", 0.15)],
    "$t0": [("$temp", 0.85), ("i64", 0.15)],
    "$t1": [("$temp", 0.70), ("1", 0.15), ("i64", 0.15)],
    "$t2": [("$temp", 0.70), ("2", 0.15), ("i64", 0.15)],
    "$t3": [("$temp", 0.70), ("3", 0.15), ("i64", 0.15)],
    "$sign": [("$temp", 0.70), ("i64", 0.30)],
}


def perform_vocabulary_surgery(
    decoder: nn.Module,
    old_vocab_size: int,
    new_vocab_size: int,
    old_token_to_id: Dict[str, int],
    new_token_to_id: Dict[str, int],
    logit_cap_threshold: float = 30.0,
) -> nn.Module:
    """Expands token embedding and linear head of decoder with convex hull semantic projection.

    Args:
        decoder: WatTransformerDecoder instance
        old_vocab_size: Previous vocabulary size (e.g. 84)
        new_vocab_size: Target vocabulary size (e.g. 109)
        old_token_to_id: Previous token to ID dictionary
        new_token_to_id: Expanded token to ID dictionary
        logit_cap_threshold: C_cap soft-capping threshold (30.0)
    """
    if new_vocab_size <= old_vocab_size:
        return decoder

    d_model = decoder.d_model
    pad_idx = decoder.pad_idx

    # 1. Allocate new embedding layer
    old_embed = decoder.token_embedding.weight.data  # (old_vocab, d_model)
    new_embedding = nn.Embedding(new_vocab_size, d_model, padding_idx=pad_idx, dtype=torch.float32)
    new_embed_weight = new_embedding.weight.data

    # Copy existing embeddings
    new_embed_weight[:old_vocab_size] = old_embed[:old_vocab_size]

    # 2. Allocate new lm_head layer
    old_head_w = decoder.lm_head.weight.data  # (old_vocab, d_model)
    has_bias = decoder.lm_head.bias is not None
    old_head_b = decoder.lm_head.bias.data if has_bias else None

    new_lm_head = nn.Linear(d_model, new_vocab_size, bias=has_bias, dtype=torch.float32)
    new_head_w = new_lm_head.weight.data
    new_head_b = new_lm_head.bias.data if has_bias else None

    new_head_w[:old_vocab_size] = old_head_w[:old_vocab_size]
    if has_bias and old_head_b is not None:
        new_head_b[:old_vocab_size] = old_head_b[:old_vocab_size]

    # Calculate mean norm of pre-trained head rows
    mean_head_norm = float(torch.norm(old_head_w, p=2, dim=1).mean().item())

    # 3. Project new token representations
    id_to_new_token = {idx: tok for tok, idx in new_token_to_id.items()}

    for new_idx in range(old_vocab_size, new_vocab_size):
        tok_name = id_to_new_token.get(new_idx, "")
        ancestors = SEMANTIC_ANCESTORS.get(tok_name, [])

        proj_embed = torch.zeros(d_model, dtype=torch.float32)
        proj_head = torch.zeros(d_model, dtype=torch.float32)
        proj_bias = 0.0
        total_weight = 0.0

        for anc_tok, weight in ancestors:
            if anc_tok in old_token_to_id:
                anc_idx = old_token_to_id[anc_tok]
                proj_embed += weight * old_embed[anc_idx]
                proj_head += weight * old_head_w[anc_idx]
                if has_bias and old_head_b is not None:
                    proj_bias += weight * float(old_head_b[anc_idx].item())
                total_weight += weight

        if total_weight > 0:
            proj_embed = proj_embed / total_weight
            proj_head = proj_head / total_weight
            if has_bias:
                proj_bias = proj_bias / total_weight
        else:
            # Fallback: mean of existing tokens
            proj_embed = old_embed.mean(dim=0)
            proj_head = old_head_w.mean(dim=0)
            if has_bias and old_head_b is not None:
                proj_bias = float(old_head_b.mean().item())

        # Norm calibration for unembedding head
        norm_val = float(torch.norm(proj_head, p=2).item())
        if norm_val > 1e-8:
            calibrated_head = proj_head * (mean_head_norm / norm_val)
        else:
            calibrated_head = proj_head

        new_embed_weight[new_idx] = proj_embed
        new_head_w[new_idx] = calibrated_head
        if has_bias and new_head_b is not None:
            new_head_b[new_idx] = proj_bias

    # Assign new modules to decoder
    decoder.token_embedding = new_embedding
    decoder.lm_head = new_lm_head
    decoder.vocab_size = new_vocab_size
    decoder.logit_cap_threshold = logit_cap_threshold

    return decoder
