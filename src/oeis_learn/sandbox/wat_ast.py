"""Frozen body-only AST; syntax/type checks reuse the complete codec contract."""

from __future__ import annotations

from dataclasses import dataclass
import re
from oeis_learn.decoder.program_codec import CodecError, encode_body, decode_body, _source_words
from oeis_learn.experiments.profiles import ALLOWED_OPERATORS, I64_LOCAL_NAMES, I32_LOCAL_NAMES


class ExecutionFailure(ValueError):
    def __init__(self, outcome: str, reason: str):
        super().__init__(reason)
        self.outcome, self.reason = outcome, reason


@dataclass(frozen=True)
class Instruction:
    op: str
    arg: int | str | None = None
    body: tuple[Instruction, ...] = ()
    otherwise: tuple[Instruction, ...] = ()
    results: tuple[str, ...] = ()
    has_else: bool = False


@dataclass(frozen=True)
class Program:
    instructions: tuple[Instruction, ...]
    canonical_source: str
    body_tokens: tuple[int, ...]


def parse(source: str) -> Program:
    """Reject wrappers/folded expressions and features outside the frozen body.

    Diagnostic folded-WAT normalization is intentionally not an admission path.
    Every accepted AST round trips to the same complete codec token sequence.
    """
    try:
        words = _source_words(source)
        for word in words:
            if (
                word
                in (
                    "module",
                    "func",
                    "export",
                    "param",
                    "local",
                    "import",
                    "memory",
                    "global",
                    "table",
                    "call",
                    "start",
                )
                or "__oeis" in word
            ):
                raise ExecutionFailure(
                    "unsupported", "guest wrapper, declaration or reserved helper namespace"
                )
            if word.startswith("$") and word not in (*I64_LOCAL_NAMES, *I32_LOCAL_NAMES, "$n"):
                raise ExecutionFailure("invalid_type", f"undeclared local: {word}")
            if (
                "." in word
                and not re.fullmatch(r"-?[0-9]+", word)
                and word not in ALLOWED_OPERATORS
            ):
                raise ExecutionFailure("unsupported", f"operator outside frozen inventory: {word}")
        tokens = encode_body(source)
        canonical = decode_body(tokens)
    except CodecError as exc:
        reason = str(exc)
        if "cap" in reason or "64 KiB" in reason or "nesting limit" in reason:
            outcome = "execution_limit"
        elif (
            "integer prefix exceeds instruction range" in reason
            or "integer outside instruction range" in reason
        ):
            outcome = "numeric_limit"
        elif any(
            x in reason
            for x in (
                "operand",
                "expected i32",
                "expected i64",
                "branch target",
                "undeclared local",
                "result signature",
            )
        ):
            outcome = "invalid_type"
        else:
            outcome = "invalid_syntax"
        raise ExecutionFailure(outcome, reason) from exc
    words = canonical.split()
    position = 0

    def sequence() -> tuple[Instruction, ...]:
        nonlocal position
        nodes = []
        while position < len(words) and words[position] != ")":
            word = words[position]
            position += 1
            if word == "(":
                op = words[position]
                position += 1
                results = []
                if words[position : position + 2] == ["(", "result"]:
                    position += 2
                    while words[position] != ")":
                        results.append(words[position])
                        position += 1
                    position += 1
                if op == "if":
                    position += 2  # ( then
                    body = sequence()
                    position += 1  # )
                    has_else = words[position : position + 2] == ["(", "else"]
                    otherwise = ()
                    if has_else:
                        position += 2
                        otherwise = sequence()
                        position += 1
                    position += 1  # if closing )
                else:
                    body, otherwise, has_else = sequence(), (), False
                    position += 1
                nodes.append(
                    Instruction(
                        op,
                        body=body,
                        otherwise=otherwise,
                        results=tuple(results),
                        has_else=has_else,
                    )
                )
            else:
                arg = None
                if word in (
                    "i32.const",
                    "i64.const",
                    "i256.const",
                    "br",
                    "br_if",
                    "local.get",
                    "local.set",
                    "local.tee",
                ):
                    arg = words[position]
                    position += 1
                    if not word.startswith("local."):
                        arg = int(arg)
                nodes.append(Instruction(word, arg))
        return tuple(nodes)

    instructions = sequence()
    return Program(instructions, canonical, tuple(tokens))
