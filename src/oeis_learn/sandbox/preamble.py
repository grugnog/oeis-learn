"""Checked multi-limb arithmetic preamble and legacy diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from oeis_learn.data.models import StaticArithmeticPreamble

PREAMBLE_WAT_PATH = Path(__file__).resolve().parent / "preamble.wat"

# Historical estimates are deliberately unavailable. Only runtime fuel is evidence.
STATIC_FUEL_COSTS: Dict[str, int] = {}


def load_preamble_wat() -> str:
    """Load the checked arithmetic preamble WAT text."""
    if not PREAMBLE_WAT_PATH.exists():
        raise FileNotFoundError(f"Static arithmetic preamble not found at {PREAMBLE_WAT_PATH}")
    with open(PREAMBLE_WAT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_preamble_fuel_costs() -> Dict[str, int]:
    """Returns mapping of preamble functions and macros to instruction fuel costs."""
    return dict(STATIC_FUEL_COSTS)


def verify_zero_memory(wat_code: str) -> bool:
    """Verifies that a WAT module does not allocate or import linear memory."""
    import wasmtime
    from oeis_learn.decoder.program_codec import _source_words

    try:
        # Never instantiate diagnostic input: arbitrary start functions can run.
        # Tokenization catches unexported memory definitions with any whitespace;
        # compilation rejects malformed text without relying on substring checks.
        words = _source_words(wat_code)
        with wasmtime.Engine() as engine, wasmtime.Module(engine, wat_code) as module:
            return not module.imports and not any(
                a == "(" and b == "memory" for a, b in zip(words, words[1:])
            )
    except (wasmtime.WasmtimeError, ValueError):
        return False


def get_static_preamble() -> StaticArithmeticPreamble:
    """Returns the StaticArithmeticPreamble entity instance."""
    wat_text = load_preamble_wat()
    return StaticArithmeticPreamble(
        version="2.0.0",
        fuel_costs=get_preamble_fuel_costs(),
        memory_limit_bytes=0,
        wat_code=wat_text,
    )
