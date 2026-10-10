"""Typed, bounded logical WAT v1. Separate from native limb-stack WAT.

Every slot/array cell holds one signed256 value. Shifts use literal counts;
source cannot declare memory, helpers, imports, or range-proof assertions.
"""

from dataclasses import dataclass
import re
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.experiments.profiles import I256_MIN, I256_MAX

VERSION = "logical-wat/v1"
BINARY = ("add", "sub", "mul", "div", "rem", "and", "or", "xor")
COMPARE = ("eq", "ne", "lt", "le", "gt", "ge")
SHIFT = ("shl", "shr_s", "shr_u")
WORDS = (
    "pure",
    "stream",
    "init",
    "next",
    "return",
    "set",
    "store",
    "if",
    "repeat",
    "while",
    "const",
    "index",
    "get",
    "load",
    "select",
    "to_i32",
    "from_i32",
    *BINARY,
    *COMPARE,
    *SHIFT,
)


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple

    def text(self):
        return (
            "("
            + " ".join([self.op, *[v.text() if isinstance(v, Node) else str(v) for v in self.args]])
            + ")"
        )


@dataclass(frozen=True)
class LogicalProgram:
    root: Node
    canonical_source: str
    profile: str

    @property
    def identity(self):
        return digest(
            {"version": VERSION, "profile": self.profile, "source": self.canonical_source}
        )


def profile_identity(mode="pure"):
    if mode not in ("pure", "stream"):
        raise ValueError("unknown logical profile")
    return {
        "version": VERSION,
        "mode": mode,
        "width": 256,
        "slots": 32,
        "array_cells": 1024,
        "memory_pages": 1,
        "shift_range": [0, 255],
        "operators": WORDS,
        "max_depth": 8,
        "max_loops": 3,
        "max_body_tokens": 1024,
        "numeric_semantics": "checked_signed256",
        "stream_origin": "emit_at_current_index_then_increment",
        "proof_link": "finite_conformance_only",
    }


def parse_logical(source):
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise ValueError("logical source cap")
    words = re.findall(r"\(|\)|[^\s()]+", source)
    if len(words) > 1024:
        raise ValueError("logical token cap")
    pos = 0
    nodes = 0

    def read(depth=0):
        nonlocal pos, nodes
        if depth > 8 or pos >= len(words):
            raise ValueError("logical nesting/syntax")
        word = words[pos]
        pos += 1
        if word == "(":
            if pos >= len(words) or words[pos] not in WORDS:
                raise ValueError("unsupported logical operation")
            op = words[pos]
            pos += 1
            args = []
            nodes += 1
            while pos < len(words) and words[pos] != ")":
                args.append(read(depth + 1))
            if pos >= len(words):
                raise ValueError("unclosed logical expression")
            pos += 1
            return Node(op, tuple(args))
        if re.fullmatch(r"0|-?[1-9][0-9]*", word):
            if len(word) > 79:
                raise ValueError("logical integer cap")
            return int(word)
        raise ValueError("logical syntax/unknown token")

    root = read()
    if pos != len(words) or not isinstance(root, Node) or root.op not in ("pure", "stream"):
        raise ValueError("one pure or stream root required")

    def literal(v, low, high):
        if type(v) is not int or not low <= v <= high:
            raise ValueError("logical operand range")

    def expression(n):
        if not isinstance(n, Node):
            raise ValueError("typed expression required")
        op, args = n.op, n.args
        if op == "const" and len(args) == 1:
            literal(args[0], I256_MIN, I256_MAX)
            return "wide"
        if op == "index" and not args:
            return "wide"
        if op == "get" and len(args) == 1:
            literal(args[0], 0, 31)
            return "wide"
        if op == "load" and len(args) == 1 and expression(args[0]) == "wide":
            return "wide"
        if (
            op in (*BINARY, *COMPARE)
            and len(args) == 2
            and all(expression(a) == "wide" for a in args)
        ):
            return "bool" if op in COMPARE else "wide"
        if op in SHIFT and len(args) == 2 and expression(args[0]) == "wide":
            literal(args[1], 0, 255)
            return "wide"
        if (
            op == "select"
            and len(args) == 3
            and expression(args[0]) == "bool"
            and expression(args[1]) == expression(args[2]) == "wide"
        ):
            return "wide"
        if op == "to_i32" and len(args) == 1 and expression(args[0]) == "wide":
            return "i32"
        if op == "from_i32" and len(args) == 1 and expression(args[0]) == "i32":
            return "wide"
        raise ValueError("invalid typed logical expression")

    def statement(n, loops=0):
        if not isinstance(n, Node):
            raise ValueError("statement required")
        op, args = n.op, n.args
        if op == "set" and len(args) == 2:
            literal(args[0], 0, 31)
            valid = expression(args[1]) == "wide"
        elif op == "store" and len(args) == 2:
            valid = all(expression(a) == "wide" for a in args)
        elif op == "if" and len(args) == 3:
            valid = expression(args[0]) == "bool"
            statement(args[1], loops)
            statement(args[2], loops)
        elif op == "while" and len(args) >= 2 and loops < 3:
            valid = expression(args[0]) == "bool"
            for a in args[1:]:
                statement(a, loops + 1)
        elif op == "repeat" and len(args) >= 2 and loops < 3:
            literal(args[0], 0, 120)
            valid = True
            for a in args[1:]:
                statement(a, loops + 1)
        else:
            raise ValueError("invalid logical statement")
        if not valid:
            raise ValueError("logical statement type")

    def function(args, returns):
        if returns:
            if (
                not args
                or not isinstance(args[-1], Node)
                or args[-1].op != "return"
                or len(args[-1].args) != 1
                or expression(args[-1].args[0]) != "wide"
            ):
                raise ValueError("wide return required")
            args = args[:-1]
        for a in args:
            statement(a)

    if root.op == "pure":
        function(root.args, True)
    elif (
        len(root.args) == 2
        and all(isinstance(a, Node) for a in root.args)
        and [a.op for a in root.args] == ["init", "next"]
    ):
        function(root.args[0].args, False)
        function(root.args[1].args, True)
    else:
        raise ValueError("init/next stream functions required")
    return LogicalProgram(root, root.text(), root.op)
