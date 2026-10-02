"""Strict body-only WAT codec and incremental validation for foundation/v1.

Integer delimiters are *tokens*, never WAT source. No weights from either
legacy codec are compatible. This validator checks syntax/types, not execution
or numeric overflow: admission still requires the execution/reference gates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field

from oeis_learn.experiments.profiles import (
    ALLOWED_OPERATORS,
    CODEC_PROFILE_ID,
    I32_LOCAL_NAMES,
    I64_LOCAL_NAMES,
    INPUT_LOCAL,
    NUMERIC_RANGES,
)

# Keep wrapper syntax tokens reserved (masked out of bodies). Ordering is frozen
# and content-addressed, including all three distinct special tokens.
FOUNDATION_VOCABULARY = tuple(
    dict.fromkeys(
        [
            "<pad>",
            "<bos>",
            "<eos>",
            "(",
            ")",
            "result",
            "i32",
            "i64",
            "module",
            "func",
            "export",
            "param",
            "local",
            '"compute"',
            '"generate_term"',
            "<int>",
            "-",
            "</int>",
        ]
        + list("0123456789")
        + [INPUT_LOCAL]
        + list(I64_LOCAL_NAMES)
        + list(I32_LOCAL_NAMES)
        + list(ALLOWED_OPERATORS)
    )
)
FOUNDATION_VOCAB_SIZE = len(FOUNDATION_VOCABULARY)
TOKEN_TO_ID = {t: i for i, t in enumerate(FOUNDATION_VOCABULARY)}
ID_TO_TOKEN = dict(enumerate(FOUNDATION_VOCABULARY))
PAD_ID, BOS_ID, EOS_ID = (TOKEN_TO_ID[t] for t in ("<pad>", "<bos>", "<eos>"))
INT_ID, NEG_ID, TERM_ID = (TOKEN_TO_ID[t] for t in ("<int>", "-", "</int>"))
DIGIT_IDS = frozenset(TOKEN_TO_ID[t] for t in "0123456789")
MAX_BODY_TOKENS = 1024
MAX_SOURCE_BYTES = 65536
LOCAL_TYPES = {
    INPUT_LOCAL: "i32",
    **dict.fromkeys(I32_LOCAL_NAMES, "i32"),
    **dict.fromkeys(I64_LOCAL_NAMES, "i64"),
}
RESULT = ("i64",) * 4


def codec_identity() -> dict:
    return {
        "id": CODEC_PROFILE_ID,
        "vocabulary": list(FOUNDATION_VOCABULARY),
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "pad_id": PAD_ID,
        "max_body_tokens": MAX_BODY_TOKENS,
        "max_source_bytes": MAX_SOURCE_BYTES,
    }


def codec_digest() -> str:
    payload = json.dumps(codec_identity(), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class CodecError(ValueError):
    """Invalid source, token stream, or incompatible model vocabulary."""


@dataclass
class _Frame:
    kind: str
    height: int
    result: list[str] = field(default_factory=list)
    unreachable: bool = False
    phase: str = "header"


@dataclass
class BodyState:
    """Incremental typed control stack; no hidden repairs or EOS insertion.

    Control frames implement Wasm's stack-polymorphic unreachable semantics.
    Blocks/ifs may declare result types; loops have no block parameters in this
    profile, so loop branch targets consume zero values.
    """

    stack: list[str] = field(default_factory=list)
    frames: list[_Frame] = field(
        default_factory=lambda: [_Frame("func", 0, list(RESULT), phase="body")]
    )
    mode: str = "body"
    pending: str = ""
    integer: str = ""
    count: int = 0
    finished: bool = False

    def _pop(self, types: tuple[str, ...] | list[str]) -> None:
        frame = self.frames[-1]
        for typ in reversed(types):
            if len(self.stack) == frame.height and frame.unreachable:
                continue
            if len(self.stack) <= frame.height:
                raise CodecError("operand stack underflow")
            actual = self.stack.pop()
            if typ != "*" and actual != "*" and actual != typ:
                raise CodecError(f"expected {typ}, got {actual}")

    def _unreachable(self) -> None:
        del self.stack[self.frames[-1].height :]
        self.frames[-1].unreachable = True

    def _end_values(self) -> None:
        self._pop(self.frames[-1].result)
        if len(self.stack) != self.frames[-1].height:
            raise CodecError("extra operands at end of control frame")

    def _branch(self, depth: int) -> None:
        if depth >= len(self.frames):
            raise CodecError("branch depth outside active scope")
        target = self.frames[-1 - depth]
        types = [] if target.kind == "loop" else target.result
        if self.pending == "br_if":
            self._pop(("i32",))
            self._pop(types)
            self.stack.extend(types)
        else:
            self._pop(types)
            self._unreachable()

    def _branch_depths(self) -> list[int]:
        result = []
        for depth in range(len(self.frames)):
            try:
                copy.deepcopy(self)._branch(depth)
            except CodecError:
                continue
            result.append(depth)
        return result

    def _range(self) -> tuple[int, int]:
        if self.pending in ("br", "br_if"):
            return 0, len(self.frames) - 1
        return NUMERIC_RANGES[self.pending.split(".")[0]]

    def _number(self, token: str) -> None:
        lo, hi = self._range()
        if token == "</int>":
            if not re.fullmatch(r"0|-?[1-9][0-9]*", self.integer):
                raise CodecError("incomplete or noncanonical integer")
            value = int(self.integer)
            if not lo <= value <= hi:
                raise CodecError("integer outside instruction range")
            if self.pending in ("br", "br_if"):
                self._branch(value)
            else:
                self.stack.extend(RESULT if self.pending == "i256.const" else [self.pending[:3]])
            self.mode, self.pending, self.integer = "body", "", ""
            return
        if token == "-" and not self.integer and lo < 0:
            self.integer = "-"
            return
        if token not in "0123456789" or len(token) != 1:
            raise CodecError("expected decimal digit or integer terminator")
        if self.integer in ("0", "-0") or (self.integer == "-" and token == "0"):
            raise CodecError("leading zero or negative zero")
        self.integer += token
        if self.pending in ("br", "br_if") and int(self.integer) not in self._branch_depths():
            raise CodecError("branch target has incompatible operand types")
        bound = -lo if self.integer.startswith("-") else hi
        # Bound length before conversion, including adversarial huge literals.
        magnitude = self.integer.lstrip("-")
        if len(magnitude) > len(str(bound)) or int(magnitude) > bound:
            raise CodecError("integer prefix exceeds instruction range")

    def consume(self, token_id: int) -> None:
        if type(token_id) is not int or token_id not in ID_TO_TOKEN:
            raise CodecError("unknown/non-integer token ID")
        if self.finished or self.count >= MAX_BODY_TOKENS:
            raise CodecError("body finished or token cap reached")
        token = ID_TO_TOKEN[token_id]
        if self.count == MAX_BODY_TOKENS - 1 and token != "<eos>":
            raise CodecError("last body slot is reserved for a valid EOS")
        self.count += 1
        if self.mode == "int_start":
            if token != "<int>":
                raise CodecError("expected <int>")
            self.mode = "integer"
            return
        if self.mode == "integer":
            self._number(token)
            return
        if self.mode == "local":
            if token not in LOCAL_TYPES:
                raise CodecError("undeclared local")
            typ = LOCAL_TYPES[token]
            if self.pending != "local.get":
                self._pop((typ,))
            if self.pending != "local.set":
                self.stack.append(typ)
            self.mode, self.pending = "body", ""
            return
        if self.mode == "result":
            if token == ")":
                if not self.frames[-1].result:
                    raise CodecError("empty result annotation")
                self.mode = "body"
                self.frames[-1].phase = "if_then" if self.frames[-1].kind == "if" else "body"
            elif token in ("i32", "i64"):
                self.frames[-1].result.append(token)
            else:
                raise CodecError("expected result type or closing parenthesis")
            return
        if self.mode in ("open", "header_open", "arm_open"):
            mode, self.mode = self.mode, "body"
            frame = self.frames[-1]
            if mode == "header_open" and token == "result":
                self.mode = "result"
                return
            if token in ("then", "else") and mode in ("header_open", "arm_open"):
                expected = "else" if frame.phase == "after_then" else "then"
                if frame.kind != "if" or token != expected:
                    raise CodecError("unexpected if arm")
                frame.phase = token
                return
            if mode == "arm_open" or (mode == "header_open" and frame.kind == "if"):
                raise CodecError("expected if arm")
            if mode == "header_open":
                frame.phase = "body"
            if token not in ("block", "loop", "if"):
                raise CodecError("only structured control may open here")
            if len(self.frames) > 8 or (
                token == "loop" and sum(f.kind == "loop" for f in self.frames) >= 3
            ):
                raise CodecError("control nesting limit exceeded")
            if token == "if":
                self._pop(("i32",))
            self.frames.append(_Frame(token, len(self.stack)))
            return
        frame = self.frames[-1]
        if frame.phase in ("header", "if_then", "after_then", "after_else"):
            if token == "(" and frame.phase != "after_else":
                self.mode = "header_open" if frame.phase == "header" else "arm_open"
                return
            if frame.kind == "if":
                if token == ")" and frame.phase in ("after_then", "after_else"):
                    if frame.phase == "after_then" and frame.result:
                        raise CodecError("value-producing if requires else")
                    self.frames.pop()
                    self.stack.extend(frame.result)
                    return
                raise CodecError("incomplete if structure")
            frame.phase = "body"
        if token == "<eos>":
            if len(self.frames) != 1:
                raise CodecError("EOS inside control scope")
            self._end_values()
            self.finished = True
        elif token == ")":
            if len(self.frames) == 1:
                raise CodecError("unmatched closing parenthesis")
            self._end_values()
            if frame.kind == "if":
                frame.phase = "after_" + frame.phase
                frame.unreachable = False
            else:
                self.frames.pop()
                self.stack.extend(frame.result)
        elif token == "(":
            if len(self.frames) > 8:
                raise CodecError("control nesting limit exceeded")
            self.mode = "open"
        elif token in ("i32.const", "i64.const", "i256.const", "br", "br_if"):
            self.pending, self.mode = token, "int_start"
            if token in ("br", "br_if") and not self._branch_depths():
                raise CodecError("no well-typed branch target")
        elif token in ("local.get", "local.set", "local.tee"):
            if token != "local.get":
                # Check availability without consuming until the local type is known.
                clone = copy.deepcopy(self)
                clone._pop(("*",))
            self.pending, self.mode = token, "local"
        elif token == "unreachable":
            self._unreachable()
        elif token == "return":
            self._pop(RESULT)
            self._unreachable()
        elif token == "drop":
            self._pop(("*",))
        elif token == "nop":
            pass
        elif token in ("i256.zero", "i256.add", "i256.sub", "i256.mul_scalar"):
            arity = {"i256.zero": 0, "i256.add": 8, "i256.sub": 8, "i256.mul_scalar": 5}[token]
            self._pop(("i64",) * arity)
            self.stack.extend(RESULT)
        elif token in ALLOWED_OPERATORS and token.startswith(("i32.", "i64.")):
            typ, op = token.split(".")
            if op == "wrap_i64":
                inputs, outputs = ("i64",), ("i32",)
            elif op in ("extend_i32_s", "extend_i32_u"):
                inputs, outputs = ("i32",), ("i64",)
            else:
                inputs = (typ,) * (1 if op == "eqz" else 2)
                outputs = (
                    "i32" if op in ("eqz", "eq", "ne", "lt_s", "gt_s", "le_s", "ge_s") else typ,
                )
            self._pop(inputs)
            self.stack.extend(outputs)
        else:
            raise CodecError(f"forbidden body token {token!r}")

    def allowed(self) -> list[int]:
        if self.finished or self.count >= MAX_BODY_TOKENS:
            return []
        result = []
        for token_id in ID_TO_TOKEN:
            try:
                copy.deepcopy(self).consume(token_id)
            except CodecError:
                continue
            result.append(token_id)
        return result


def _source_words(source: str) -> list[str]:
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise CodecError("source exceeds 64 KiB or is not text")
    # WAT line and nested block comments; never edit/reorder instructions.
    out, i, depth = [], 0, 0
    while i < len(source):
        pair = source[i : i + 2]
        if pair == "(;":
            depth += 1
            out.append(" ")
            i += 2
        elif depth and pair == ";)":
            depth -= 1
            i += 2
        elif depth:
            i += 1
        elif pair == ";;":
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
            out.append(" ")
        else:
            out.append(source[i])
            i += 1
    if depth:
        raise CodecError("unterminated block comment")
    return re.findall(r"[()]|[^\s()]+", "".join(out))


def encode_body(source: str) -> list[int]:
    ids = []
    for word in _source_words(source):
        if re.fullmatch(r"0|-?[1-9][0-9]*", word):
            ids.extend([INT_ID, *(TOKEN_TO_ID[c] for c in word), TERM_ID])
        elif word in TOKEN_TO_ID and not word.startswith("<"):
            ids.append(TOKEN_TO_ID[word])
        else:
            raise CodecError(f"unknown or noncanonical source token {word!r}")
        if len(ids) >= MAX_BODY_TOKENS:
            raise CodecError("body exceeds token cap")
    ids.append(EOS_ID)
    validate_body(ids)
    return ids


def validate_body(token_ids: list[int]) -> None:
    if not token_ids or len(token_ids) > MAX_BODY_TOKENS:
        raise CodecError("empty body or token cap exceeded")
    state = BodyState()
    for token_id in token_ids:
        state.consume(token_id)
    if not state.finished:
        raise CodecError("complete body must end with EOS")


def decode_body(token_ids: list[int]) -> str:
    validate_body(token_ids)
    parts, digits = [], []
    for token_id in token_ids[:-1]:
        token = ID_TO_TOKEN[token_id]
        if token == "<int>":
            digits = []
        elif token == "</int>":
            parts.append("".join(digits))
        elif token_id in DIGIT_IDS or token_id == NEG_ID:
            digits.append(token)
        else:
            parts.append(token)
    source = " ".join(parts)
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise CodecError("canonical source exceeds byte cap")
    return source


def allowed_tokens_at(prefix: list[int]) -> list[int]:
    state = BodyState()
    for token_id in prefix:
        state.consume(token_id)
    return state.allowed()
