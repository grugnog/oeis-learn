"""Frozen prompt controls independent of target outputs or family labels."""

import random
from oeis_learn.synthesis.contracts import identity


def derangement(ids, seed):
    ids = tuple(sorted(ids))
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("shuffling requires distinct multiple prompts")
    targets = list(ids)
    rng = random.Random(seed)
    # Sattolo: a reproducible single cycle, hence no fixed points.
    for i in range(len(targets) - 1, 0, -1):
        j = rng.randrange(i)
        targets[i], targets[j] = targets[j], targets[i]
    return dict(zip(ids, targets))


def condition_memory(memory, padding, mode, *, order=None):
    import torch

    if (
        memory.ndim != 3
        or padding.shape != memory.shape[:2]
        or padding.dtype != torch.bool
        or padding.device != memory.device
    ):
        raise ValueError("valid aligned prompt masks required")
    if padding.all(1).any():
        raise ValueError("empty/all-padding input is not an absent marker")
    if mode == "correct":
        return memory, padding
    if mode == "shuffled":
        n = len(memory)
        if (
            order is None
            or sorted(order) != list(range(n))
            or any(i == j for i, j in enumerate(order))
        ):
            raise ValueError("fixed derangement required")
        return memory[list(order)], padding[list(order)]
    if mode == "absent":
        marker = memory.new_zeros((len(memory), 1, memory.shape[-1]))
        marker[:, :, 0] = 1
        return marker, torch.zeros((len(memory), 1), dtype=torch.bool, device=memory.device)
    raise ValueError("unknown conditioning control")


CONTROL_IDENTITY = identity(
    dict(
        version="prompt-controls/v1",
        shuffle="seeded-sattolo-sorted-ids",
        absent="one-valid-token:first-coordinate-one-rest-zero",
    )
)
