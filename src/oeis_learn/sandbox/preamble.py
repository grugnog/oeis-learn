"""Static Verified Multi-Limb Arithmetic Preamble Loader and Verifier."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional
from oeis_learn.data.models import StaticArithmeticPreamble

PREAMBLE_WAT_PATH = Path(__file__).resolve().parent / "preamble.wat"

STATIC_FUEL_COSTS: Dict[str, int] = {
    "i256_add": 55,
    "i256_sub": 56,
    "mul64_wide": 60,
    "i256_mul_scalar": 271,
    "i256.add": 55,
    "i256.sub": 56,
    "i256.mul_scalar": 271,
    "i256.const": 4,
    "i256.zero": 4,
}


def load_preamble_wat() -> str:
    """Loads the verified static arithmetic preamble WAT text."""
    if not PREAMBLE_WAT_PATH.exists():
        raise FileNotFoundError(f"Static arithmetic preamble not found at {PREAMBLE_WAT_PATH}")
    with open(PREAMBLE_WAT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_preamble_fuel_costs() -> Dict[str, int]:
    """Returns mapping of preamble functions and macros to instruction fuel costs."""
    return dict(STATIC_FUEL_COSTS)


def verify_zero_memory(wat_code: str) -> bool:
    """Verifies that a WAT module does not allocate or import linear memory."""
    # Check for (memory ...), (import ... (memory ...))
    # We enforce strict zero-linear-memory (max_memories=0).
    lower = wat_code.lower()
    if "(memory" in lower:
        return False
    return True


def get_static_preamble() -> StaticArithmeticPreamble:
    """Returns the StaticArithmeticPreamble entity instance."""
    wat_text = load_preamble_wat()
    return StaticArithmeticPreamble(
        version="1.0.0",
        fuel_costs=get_preamble_fuel_costs(),
        memory_limit_bytes=0,
        wat_code=wat_text,
    )
