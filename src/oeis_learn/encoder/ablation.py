"""Frozen four-group masks; post-training sensitivity differs from retrained utility."""

from dataclasses import dataclass, asdict
from oeis_learn.synthesis.contracts import identity


@dataclass(frozen=True)
class InputMask:
    magnitude: bool = True
    modular: bool = True
    difference: bool = True
    exact_bytes: bool = True

    def __post_init__(self):
        if any(type(x) is not bool for x in vars(self).values()):
            raise ValueError("boolean input mask required")

    @property
    def sha256(self):
        return identity({"version": "four-input-boundaries/v1", **asdict(self)})

    def bits(self):
        return tuple(vars(self).values())
