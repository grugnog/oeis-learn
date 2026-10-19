"""Independent recursive source reader for the final-source arithmetic oracle.

No production parser, compiler, AST, codec or lowering imports. Syntax is the
versioned s-expression language, not arbitrary WAT. Locations count characters.
"""

from dataclasses import dataclass
import re

from .contracts import ResourceProfile


class SourceError(ValueError):
    pass


@dataclass(frozen=True)
class ReferenceProgram:
    declarations: tuple
    statements: tuple
    result: tuple | str


def parse(source, limits=ResourceProfile(), *, allow_parameters=False):
    if type(source) is not str or len(source.encode("utf-8")) > limits.source_bytes:
        raise SourceError("source limit")
    # Explicit lexer verifies every byte; comments/strings/Unicode atoms absent.
    tokens, pos = [], 0
    token = re.compile(r"\s+|[()]|[^\s()]+", re.ASCII)
    while pos < len(source):
        match = token.match(source, pos)
        if match is None:
            raise SourceError(f"invalid character at {pos}")
        text = match.group()
        if not text.isspace():
            if not text.isascii():
                raise SourceError("non-ASCII atom")
            tokens.append(text)
        pos = match.end()
    cursor, nodes = 0, 0

    def read(depth=0):
        nonlocal cursor, nodes
        nodes += 1
        if nodes > limits.ast_nodes or depth > limits.nesting or cursor >= len(tokens):
            raise SourceError("node/depth limit or incomplete source")
        value = tokens[cursor]
        cursor += 1
        if value == ")":
            raise SourceError("unexpected close")
        if value != "(":
            return value
        children = []
        while cursor < len(tokens) and tokens[cursor] != ")":
            children.append(read(depth + 1))
        if cursor == len(tokens):
            raise SourceError("unclosed expression")
        cursor += 1
        if not children:
            raise SourceError("empty expression")
        return tuple(children)

    tree = read()
    if cursor != len(tokens) or not isinstance(tree, tuple) or tree[0] != "program":
        raise SourceError("expected one complete program")
    env, declarations, body = {}, [], list(tree[1:])
    while body and isinstance(body[0], tuple) and body[0][0] == "local":
        entry = body.pop(0)
        if len(entry) != 3:
            raise SourceError("invalid declaration")
        _, name, kind = entry
        cap_prefix = {
            "bigint": (limits.bigint_locals, "b"),
            "control": (limits.control_locals, "c"),
            "bool": (limits.bool_locals, "q"),
        }
        if kind not in cap_prefix or not isinstance(name, str):
            raise SourceError("invalid local type")
        cap, prefix = cap_prefix[kind]
        if name not in {f"${prefix}{n}" for n in range(cap)} or name in env:
            raise SourceError("invalid/duplicate local")
        env[name] = kind
        declarations.append((name, kind))

    def number(text, bits):
        if (
            type(text) is not str
            or len(text) > bits * 30103 // 100000 + 2
            or re.fullmatch(r"0|-?[1-9][0-9]*", text, re.ASCII) is None
        ):
            raise SourceError("invalid literal")
        n = 0
        for digit in text.lstrip("-"):
            n = 10 * n + ord(digit) - 48
        if n.bit_length() > bits:
            raise SourceError("literal limit")
        return -n if text.startswith("-") else n

    def expr(value):
        if value == "$index":
            return "bigint"
        if not isinstance(value, tuple) or not isinstance(value[0], str):
            raise SourceError("expected typed expression")
        op, args = value[0], value[1:]
        if op == "param.get" and allow_parameters and len(args) == 1 and args[0] in {f"p{i}" for i in range(8)}:
            return "bigint"
        if op == "int" and len(args) == 1:
            number(args[0], limits.literal_bits)
            return "bigint"
        if op == "control" and len(args) == 1:
            n = number(args[0], 31)
            if not 0 <= n <= limits.control_max:
                raise SourceError("control literal range")
            return "control"
        if op == "get" and len(args) == 1 and isinstance(args[0], str) and args[0] in env:
            return env[args[0]]
        signatures = {
            "add": (("bigint", "bigint"), "bigint"),
            "sub": (("bigint", "bigint"), "bigint"),
            "mul": (("bigint", "bigint"), "bigint"),
            "eq": (("bigint", "bigint"), "bool"),
            "lt": (("bigint", "bigint"), "bool"),
            "le": (("bigint", "bigint"), "bool"),
            "to_control": (("bigint",), "control"),
            "to_bigint": (("control",), "bigint"),
            "increment": (("control",), "control"),
            "control.lt": (("control", "control"), "bool"),
            "true": ((), "bool"),
            "false": ((), "bool"),
        }
        if op == "select" and len(args) == 3:
            types = tuple(map(expr, args))
            if types[0] == "bool" and types[1] == types[2]:
                return types[1]
        if op in signatures:
            ins, out = signatures[op]
            if len(args) == len(ins) and tuple(map(expr, args)) == ins:
                return out
        raise SourceError(f"unsupported or ill-typed {op}")

    def statement(value, loop_counters=frozenset()):
        if not isinstance(value, tuple):
            raise SourceError("expected statement")
        op, args = value[0], value[1:]
        if op == "set" and len(args) == 2 and isinstance(args[0], str) and args[0] in env:
            if args[0] in loop_counters or expr(args[1]) != env[args[0]]:
                raise SourceError("invalid assignment or immutable loop counter")
        elif (
            op == "loop"
            and len(args) >= 3
            and isinstance(args[0], str)
            and env.get(args[0]) == "control"
        ):
            if args[0] in loop_counters or expr(args[1]) != "control":
                raise SourceError("invalid loop counter/bound")
            for child in args[2:]:
                statement(child, loop_counters | {args[0]})
        elif op == "if" and len(args) == 3 and expr(args[0]) == "bool":
            for branch, label in zip(args[1:], ("then", "else")):
                if not isinstance(branch, tuple) or branch[0] != label:
                    raise SourceError("invalid branch")
                for child in branch[1:]:
                    statement(child, loop_counters)
        else:
            raise SourceError(f"unsupported statement {op}")

    if not body or not isinstance(body[-1], tuple) or len(body[-1]) != 2 or body[-1][0] != "return":
        raise SourceError("one final bigint return required")
    if expr(body[-1][1]) != "bigint":
        raise SourceError("bigint return required")
    for item in body[:-1]:
        statement(item)
    return ReferenceProgram(tuple(declarations), tuple(body[:-1]), body[-1][1])
