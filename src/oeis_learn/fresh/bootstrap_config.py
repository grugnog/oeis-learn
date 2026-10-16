"""Closed controls for new bootstrap runs; never inferred for old inputs."""

import math


def learning_defaults():
    return dict(
        learning_rate=0.0001,
        validation_every_updates=250,
        max_updates=5000,
        patience=8,
        min_improvement=0.001,
        diagnostic_rows=8,
        diagnostic_tokens=256,
        diagnostic_seconds=120,
        curriculum=True,
        promotion_accuracy=0.8,
        replay=False,
        prefix_verified=True,
    )


def validate_learning(value):
    if not isinstance(value, dict) or set(value) != set(learning_defaults()):
        raise ValueError("closed bootstrap learning settings required")
    for name, low, high in (
        ("validation_every_updates", 1, 2000),
        ("max_updates", 1, 100000),
        ("patience", 1, 100),
        ("diagnostic_rows", 2, 32),
        ("diagnostic_tokens", 8, 1023),
        ("diagnostic_seconds", 10, 3000),
    ):
        if type(value[name]) is not int or not low <= value[name] <= high:
            raise ValueError(f"{name}: integer in {low}..{high} required")
    for name, low, high in (
        ("learning_rate", 1e-7, 0.01),
        ("min_improvement", 0.0, 1.0),
        ("promotion_accuracy", 0.5, 1.0),
    ):
        if (
            type(value[name]) not in (int, float)
            or not math.isfinite(value[name])
            or not low <= value[name] <= high
        ):
            raise ValueError(f"{name}: finite number in {low}..{high} required")
    for name in ("curriculum", "replay", "prefix_verified"):
        if type(value[name]) is not bool:
            raise ValueError(f"{name}: boolean required")
    return dict(value)
