"""Unreduced token losses that retain strict CUDA/HIP determinism."""

import torch.nn.functional as F


def token_cross_entropy(logits, targets, *, ignore_index=-100):
    """Return [batch, tokens] losses from [batch, tokens, vocabulary] logits.

    Flatten tokens into classification examples before calling cross entropy.
    The spatial CUDA/HIP NLL reduction uses atomics and rejects strict
    determinism; callers instead apply their existing token/program reduction
    to these unreduced classification losses.
    """
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:-1] != targets.shape:
        raise ValueError("token loss requires matching [batch, tokens] dimensions")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    return losses.reshape_as(targets)
