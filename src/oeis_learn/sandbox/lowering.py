"""Macro Lowering Engine for WebAssembly Multi-Limb Operations.

Lowers high-level macros (i256.add, i256.sub, i256.mul_scalar, i256.const, i256.zero)
to pure value-stack static arithmetic preamble function calls.
"""

from __future__ import annotations

import re
from typing import Optional
from oeis_learn.sandbox.preamble import load_preamble_wat

_CONST_PAREN_REGEX = re.compile(r"\(\s*i256\.const\s+(-?\d+)\s*\)")
_CONST_BARE_REGEX = re.compile(r"\bi256\.const\s+(-?\d+)\b")
_ZERO_PAREN_REGEX = re.compile(r"\(\s*i256\.zero\s*\)")
_ZERO_BARE_REGEX = re.compile(r"\bi256\.zero\b")
_ADD_REGEX = re.compile(r"\bi256\.add\b")
_SUB_REGEX = re.compile(r"\bi256\.sub\b")
_MUL_REGEX = re.compile(r"\bi256\.mul_scalar\b")

_CACHED_PREAMBLE_FUNCS: Optional[str] = None


def get_macro_fuel_cost(macro_name: str) -> None:
    """Static estimates are unavailable; measure fuel on each actual execution."""
    return None  # Fuel is measured from Store consumption, not a static estimate.


def extract_preamble_funcs() -> str:
    """Extracts internal function definitions from preamble.wat, excluding outer (module ...)."""
    global _CACHED_PREAMBLE_FUNCS
    if _CACHED_PREAMBLE_FUNCS is not None:
        return _CACHED_PREAMBLE_FUNCS

    raw = load_preamble_wat()
    start = raw.index("(module") + len("(module")
    _CACHED_PREAMBLE_FUNCS = raw[start : raw.rfind(")")].strip()

    return _CACHED_PREAMBLE_FUNCS


def lower_macro_wat(wat_code: str) -> str:
    """Legacy diagnostic lowering; strict admission uses the validated AST path.

    Transforms:
      i256.add -> call $i256_add
      i256.sub -> call $i256_sub
      i256.mul_scalar -> call $i256_mul_scalar
      i256.const <C> -> i64.const <C> i64.const <sign> i64.const <sign> i64.const <sign>
      i256.zero -> i64.const 0 i64.const 0 i64.const 0 i64.const 0
    And injects the preamble functions inside the (module ...).
    """
    code = wat_code

    # 1. Lower constants
    def _rep_const_paren(m: re.Match) -> str:
        v = int(m.group(1))
        return " " + lower_constant(v) + " "

    def _rep_const_bare(m: re.Match) -> str:
        v = int(m.group(1))
        return " " + lower_constant(v) + " "

    if "i256.const" in code:
        code = _CONST_PAREN_REGEX.sub(_rep_const_paren, code)
        code = _CONST_BARE_REGEX.sub(_rep_const_bare, code)

    # 2. Lower zero
    if "i256.zero" in code:
        code = _ZERO_PAREN_REGEX.sub(
            " (i64.const 0) (i64.const 0) (i64.const 0) (i64.const 0) ", code
        )
        code = _ZERO_BARE_REGEX.sub(" i64.const 0 i64.const 0 i64.const 0 i64.const 0 ", code)

    # 3. Lower arithmetic macros
    needs_add = "i256.add" in code or "call $i256_add" in code
    needs_sub = "i256.sub" in code or "call $i256_sub" in code
    needs_mul = "i256.mul_scalar" in code or "call $i256_mul_scalar" in code

    if "i256.add" in code:
        code = _ADD_REGEX.sub(" call $i256_add ", code)
    if "i256.sub" in code:
        code = _SUB_REGEX.sub(" call $i256_sub ", code)
    if "i256.mul_scalar" in code:
        code = _MUL_REGEX.sub(" call $i256_mul_scalar ", code)

    # 4. If preamble functions are needed and not already present, inject them
    needs_preamble = needs_add or needs_sub or needs_mul

    if needs_preamble and "(func $i256_add" not in code:
        preamble_funcs = extract_preamble_funcs()
        # Find where (module starts (allowing arbitrary whitespace like "( module")
        match = re.search(r"\(\s*module\b", code)
        if match:
            insert_pos = match.end()
            # If code is already wrapped in a module, remove outer wrapping or inject cleanly inside the existing module
            code = code[:insert_pos] + "\n  " + preamble_funcs + "\n" + code[insert_pos:]
        else:
            # Wrap in module
            code = f"(module\n  {preamble_funcs}\n  {code}\n)"

    return code


def lower_constant(value: int) -> str:
    """Four valid signed i64 spellings of a checked signed-256 value."""
    if type(value) is not int or not -(1 << 255) <= value < (1 << 255):
        raise ValueError("signed-256 literal outside range")
    limbs = [(value >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
    return " ".join(f"i64.const {x if x < 1 << 63 else x - (1 << 64)}" for x in limbs)


def lower_body(program) -> str:
    """Lower a validated AST, never use legacy regex rewrites for admission."""
    from oeis_learn.experiments.profiles import I32_LOCAL_NAMES, I64_LOCAL_NAMES

    def emit(nodes):
        words = []
        for node in nodes:
            if node.op == "i256.const":
                words.append(lower_constant(node.arg))
            elif node.op == "i256.zero":
                words.append(lower_constant(0))
            elif node.op.startswith("i256."):
                words.append("call $" + node.op.replace(".", "_"))
            elif node.op in ("block", "loop", "if"):
                result = "(result " + " ".join(node.results) + ")" if node.results else ""
                body = emit(node.body)
                if node.op == "if":
                    body = (
                        "(then "
                        + body
                        + ")"
                        + (" (else " + emit(node.otherwise) + ")" if node.has_else else "")
                    )
                words.append(f"({node.op} {result} {body})")
            else:
                words.append(node.op + (f" {node.arg}" if node.arg is not None else ""))
        return " ".join(words)

    locals_ = " ".join(f"(local {n} i64)" for n in I64_LOCAL_NAMES)
    locals_ += " " + " ".join(f"(local {n} i32)" for n in I32_LOCAL_NAMES)
    return f'(module {extract_preamble_funcs()} (func (export "compute") (param $n i32) (result i64 i64 i64 i64) {locals_} {emit(program.instructions)}))'
