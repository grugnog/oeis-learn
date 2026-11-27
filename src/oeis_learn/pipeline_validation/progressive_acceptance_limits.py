"""Finite diagnostic storage bounds measured from the selected real model.

These fixture bounds are not research calibration or device qualification.
"""

import os
import shutil
from pathlib import Path

from oeis_learn.generalization.progressive.storage import StorageGuard


def measure(profile):
    from oeis_learn.generalization.progressive.models import construct

    encoder, decoder, counts = construct(profile, device="cpu")
    # Include persistent buffers (e.g. positional encodings), not just params.
    weights = sum(
        tensor.numel() * tensor.element_size()
        for model in (encoder, decoder)
        for tensor in model.state_dict().values()
    )
    del encoder, decoder
    # Weights + two Adam moments, serialization/context overhead and a second
    # such allowance. Round upward; never truncate the actual selected model.
    quantum = 16 << 20
    checkpoint = ((6 * weights + (8 << 20) + quantum - 1) // quantum) * quantum
    return {
        "scope": "DIAGNOSTIC_ONLY",
        "profile": profile["identity"],
        "parameters": counts["total_parameters"],
        "weight_bytes": weights,
        "checkpoint_bytes": checkpoint,
        # Two generations plus projected/imported predecessor and old outputs.
        "task_bytes": 4 * checkpoint + (32 << 20),
    }


def storage(root, measured, *, tasks):
    if type(tasks) is not int or not 1 <= tasks <= 16:
        raise ValueError("finite diagnostic task inventory required")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    value = {
        "checkpoint_bytes": measured["checkpoint_bytes"],
        "atomic_generations": 2,
        "milestone_pins": 0,
        "shard_bytes": 1 << 20,
        "package_bytes": 4 << 20,
        "task_store_bytes": tasks * measured["task_bytes"],
        "projection_bytes": tasks * (4 << 20),
        "journal_bytes": 128 << 20,
        "checkpoint_inodes": 128,
        "required_inodes": 16384 + tasks * 1024,
        "available_bytes": shutil.disk_usage(root).free,
        "available_inodes": os.statvfs(root).f_favail,
    }
    value["required_bytes"] = sum(
        value[k]
        for k in (
            "task_store_bytes",
            "projection_bytes",
            "journal_bytes",
            "shard_bytes",
            "package_bytes",
        )
    )
    StorageGuard(
        [root], max_bytes=value["available_bytes"], max_inodes=value["available_inodes"]
    ).check(value["required_bytes"], value["required_inodes"])
    return value
