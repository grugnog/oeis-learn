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


def lower_with_map(program):
    """Emit guest instruction sites and block paths, excluding helper internals.

    Maps WAT lowering sites; it does not claim instrumented native byte offsets.
    """
    from oeis_learn.sandbox.wat_ast import Instruction, Program

    mapping, identities = [], {}

    def nodes(items, path=()):
        emitted = []
        for ordinal, node in enumerate(items):
            node_id = len(mapping)
            logical_path = path + (ordinal,)
            identities[id(node)] = node_id
            mapping.append(
                {"instruction_id": node_id, "op": node.op, "logical_path": list(logical_path)}
            )
            # Unique no-op comment marker, emitted through the same AST lowerer.
            marker = Instruction(f"(;;oeis-site-{node_id};;) ")
            mapped = Instruction(
                node.op,
                node.arg,
                nodes(node.body, logical_path + (0,)),
                nodes(node.otherwise, logical_path + (1,)),
                node.results,
                node.has_else,
            )
            emitted.extend((marker, mapped))
        return tuple(emitted)

    annotated = Program(nodes(program.instructions), program.canonical_source, program.body_tokens)
    lowered = lower_body(annotated)
    for item in mapping:
        item["lowered_wat_offset"] = lowered.index(f"(;;oeis-site-{item['instruction_id']};;)")
    return lowered, mapping, identities


def lower_logical(program):
    """Deterministic checked lowering; only trusted code creates state/memory."""
    from oeis_learn.sandbox.logical_ir import parse_logical, COMPARE, SHIFT
    from oeis_learn.sandbox.wide_helpers import helpers, decl, get

    if isinstance(program, str):
        program = parse_logical(program)
    # Reparse immutable source, so callers cannot substitute an untyped AST.
    program = parse_logical(program.canonical_source)
    globals_ = " ".join(
        f'(global $s{s}_{i} (export "__slot{s}_{i}") (mut i64) (i64.const 0))'
        for s in range(32)
        for i in range(4)
    )
    loops = []

    def expression(n):
        op, args = n.op, n.args
        if op == "const":
            return lower_constant(args[0])
        if op == "index":
            return "global.get $index global.get $index i64.const 63 i64.shr_s global.get $index i64.const 63 i64.shr_s global.get $index i64.const 63 i64.shr_s"
        if op == "get":
            return " ".join(f"global.get $s{args[0]}_{i}" for i in range(4))
        if op == "load":
            return expression(args[0]) + " call $array_load"
        if op in ("add", "sub", "mul", "div", "rem", "and", "or", "xor"):
            helper = "i256_" + op if op in ("add", "sub") else op
            return expression(args[0]) + " " + expression(args[1]) + f" call ${helper}"
        if op in COMPARE:
            wasm = {"eq": "eq", "ne": "ne", "lt": "lt_s", "le": "le_s", "gt": "gt_s", "ge": "ge_s"}[
                op
            ]
            return (
                expression(args[0])
                + " "
                + expression(args[1])
                + f" call $scmp i32.const 0 i32.{wasm}"
            )
        if op in SHIFT:
            return expression(args[0]) + f" i32.const {args[1]} call ${op}"
        if op == "select":
            return (
                expression(args[0])
                + f" if (result i64 i64 i64 i64) {expression(args[1])} else {expression(args[2])} end"
            )
        if op == "to_i32":
            return expression(args[0]) + " call $to_i32"
        if op == "from_i32":
            return expression(args[0]) + " call $from_i32"
        raise ValueError("unsupported typed expression")

    def statement(n):
        op, args = n.op, n.args
        if op == "set":
            return (
                expression(args[1])
                + " "
                + " ".join(f"global.set $s{args[0]}_{i}" for i in reversed(range(4)))
            )
        if op == "store":
            return expression(args[0]) + " " + expression(args[1]) + " call $array_store"
        if op == "if":
            return expression(args[0]) + f" if {statement(args[1])} else {statement(args[2])} end"
        if op == "while":
            key = len(loops)
            loops.append(f"(local $loop{key} i32)")
            content = " ".join(statement(a) for a in args[1:])
            return f"block $end{key} loop $again{key} {expression(args[0])} i32.eqz br_if $end{key} {content} br $again{key} end end"
        if op == "repeat":
            key = len(loops)
            loops.append(f"(local $loop{key} i32)")
            content = " ".join(statement(a) for a in args[1:])
            return f"i32.const {args[0]} local.set $loop{key} block $end{key} loop $again{key} local.get $loop{key} i32.eqz br_if $end{key} {content} local.get $loop{key} i32.const 1 i32.sub local.set $loop{key} br $again{key} end end"
        raise ValueError("unsupported typed statement")

    def function(args):
        return " ".join(statement(n) for n in args[:-1]) + " " + expression(args[-1].args[0])

    array_funcs = f"""(func $array_load {decl("param", "a")} (result i64 i64 i64 i64) (local $offset i32)
    {get("a")} call $bounds local.set $offset {" ".join(f"local.get $offset i64.load offset={i * 8}" for i in range(4))})
    (func $array_store {decl("param", "a")} {decl("param", "b")} (local $offset i32)
    {get("a")} call $bounds local.set $offset {" ".join(f"local.get $offset local.get $b{i} i64.store offset={i * 8}" for i in range(4))})
    (func $from_i32 (param $a i32) (result i64 i64 i64 i64) local.get $a i64.extend_i32_s {" ".join("local.get $a i32.const 31 i32.shr_s i64.extend_i32_s" for _ in range(3))})"""
    if program.profile == "pure":
        body = function(program.root.args)
        funcs = f'(func (export "compute") (param $n i32) (result i64 i64 i64 i64) {" ".join(loops)} local.get $n i64.extend_i32_s global.set $index {body})'
    else:
        init = " ".join(statement(n) for n in program.root.args[0].args)
        initlocals = " ".join(loops)
        loops.clear()
        nxt = function(program.root.args[1].args)
        reset = " ".join(f"i64.const 0 global.set $s{s}_{i}" for s in range(32) for i in range(4))
        funcs = f"""(func (export "init") (param $start i32) {initlocals} {reset}
        i32.const 0 i32.const 0 i32.const 65536 memory.fill
        local.get $start i64.extend_i32_s global.set $index {init})
        (func (export "next") (result i64 i64 i64 i64) {" ".join(loops)}
        global.get $index i64.const 2147483647 i64.gt_s if i32.const 1 global.set $__oeis_numeric_limit unreachable end
        {nxt} global.get $index i64.const 1 i64.add global.set $index)"""
    return f'(module {helpers()} (memory (export "__memory") 1 1) (global $index (export "__index") (mut i64) (i64.const 0)) {globals_} {array_funcs} {funcs})'
