"""Typed predictive grammar for bigint-codec/v1; never repairs completions.

Replaying a bounded prefix computes the exact next-token support. This parser is
for generation only; the independent reference has a separate source parser.
"""

from dataclasses import dataclass, field
from oeis_learn.synthesis.contracts import canonical_integer, ContractError


class CodecError(ValueError):
    pass


class Need(Exception):
    def __init__(self, choices):
        self.choices = set(choices)


class Reader:
    def __init__(self, tokens):
        self.tokens = tokens
        self.at = 0
        self.locals = {}
        self.parameters = []
        self.counters = set()
        self.depth = 0

    def take(self, choices):
        choices = set(choices)
        if self.at == len(self.tokens):
            raise Need(choices)
        token = self.tokens[self.at]
        if token not in choices:
            raise CodecError(f"unexpected {token!r} at token {self.at}")
        self.at += 1
        return token

    def open(self):
        self.take(["("] if self.depth < 32 else [])
        self.depth += 1

    def close(self):
        self.take([")"])
        self.depth -= 1

    def number(self, control=False):
        self.take(["<int>"])
        digits = ""
        bound = 65536 if control else (1 << 4096) - 1
        while True:
            allowed = set()
            if digits and digits != "-":
                allowed.add("</int>")
            if digits not in ("0", "-0"):
                for digit in "0123456789":
                    candidate = digits + digit
                    if candidate.startswith("-0") or (
                        candidate.startswith("0") and len(candidate) > 1
                    ):
                        continue
                    if len(candidate) <= 1235:
                        try:
                            n = canonical_integer(candidate)
                            if abs(n) <= bound:
                                allowed.add(digit)
                        except ContractError:
                            pass
            if not digits and not control:
                allowed.add("-")
            token = self.take(allowed)
            if token == "</int>":
                return
            digits += token

    def expr(self, kind):
        available = {name for name, t in self.locals.items() if t == kind}
        atoms = (["("] if self.depth < 32 else []) + (["$index"] if kind == "bigint" else [])
        first = self.take(atoms)
        if first == "$index":
            return
        self.depth += 1
        if self.depth > 32:
            raise CodecError("nesting limit")
        ops = {
            "bigint": {"int", "param.get", "add", "sub", "mul", "to_bigint", "select"},
            "control": {"control", "to_control", "increment", "select"},
            "bool": {"true", "false", "eq", "lt", "le", "control.lt", "select"},
        }[kind].copy()
        if available:
            ops.add("get")
        op = self.take(ops)
        if op in ("int", "control"):
            self.number(control=op == "control")
        elif op == "param.get":
            choices = self.parameters + (
                [f"p{len(self.parameters)}"] if len(self.parameters) < 8 else []
            )
            name = self.take(choices)
            if name not in self.parameters:
                self.parameters.append(name)
        elif op == "get":
            self.take(available)
        elif op in ("add", "sub", "mul", "eq", "lt", "le"):
            self.expr("bigint")
            self.expr("bigint")
        elif op == "control.lt":
            self.expr("control")
            self.expr("control")
        elif op == "to_bigint" or op == "increment":
            self.expr("control")
        elif op == "to_control":
            self.expr("bigint")
        elif op == "select":
            self.expr("bool")
            self.expr(kind)
            self.expr(kind)
        self.close()

    def statement_ops(self):
        out = {"if"}
        if set(self.locals) - self.counters:
            out.add("set")
        if any(t == "control" and n not in self.counters for n, t in self.locals.items()):
            out.add("loop")
        return out

    def statement(self, op):
        if op == "set":
            name = self.take(set(self.locals) - self.counters)
            self.expr(self.locals[name])
        elif op == "if":
            self.expr("bool")
            for label in ("then", "else"):
                self.open()
                self.take([label])
                self.sequence()
                self.close()
        elif op == "loop":
            name = self.take(
                n for n, t in self.locals.items() if t == "control" and n not in self.counters
            )
            self.expr("control")
            self.counters.add(name)
            self.open()
            first = self.take(self.statement_ops())
            self.statement(first)
            self.sequence()
            self.counters.remove(name)
        else:
            raise CodecError("invalid statement")
        self.close()

    def sequence(self):
        while True:
            token = self.take(["(", ")"] if self.depth < 32 else [")"])
            if token == ")":
                self.at -= 1
                return
            self.depth += 1
            if self.depth > 32:
                raise CodecError("nesting limit")
            self.statement(self.take(self.statement_ops()))

    def program(self):
        self.open()
        self.take(["program"])
        declarations = True
        while True:
            self.open()
            names = {
                **{f"$b{n}": "bigint" for n in range(16)},
                **{f"$c{n}": "control" for n in range(8)},
                **{f"$q{n}": "bool" for n in range(8)},
            }
            unused = set(names) - set(self.locals)
            ops = self.statement_ops() | {"return"}
            if declarations and unused:
                ops.add("local")
            op = self.take(ops)
            if op == "local":
                name = self.take(unused)
                self.take([names[name]])
                self.locals[name] = names[name]
                self.close()
            elif op == "return":
                self.expr("bigint")
                self.close()
                break
            else:
                declarations = False
                self.statement(op)
        self.close()
        self.take(["<eos>"])
        if self.at != len(self.tokens):
            raise CodecError("tokens after EOS")


def support(tokens):
    try:
        Reader(tokens).program()
    except Need as need:
        return need.choices
    return set()


@dataclass
class BodyState:
    tokens: list = field(default_factory=list)
    finished: bool = False

    def consume(self, token_id):
        from .bigint_codec import ID_TO_TOKEN, EOS_ID, MAX_BODY_TOKENS

        if type(token_id) is not int or token_id not in ID_TO_TOKEN or self.finished:
            raise CodecError("invalid token or tokens after EOS")
        token = ID_TO_TOKEN[token_id]
        if len(self.tokens) >= MAX_BODY_TOKENS or token not in support(self.tokens):
            raise CodecError("invalid typed prefix or token cap")
        self.tokens.append(token)
        self.finished = token_id == EOS_ID

    def allowed(self):
        from .bigint_codec import TOKEN_TO_ID, MAX_BODY_TOKENS

        if self.finished or len(self.tokens) >= MAX_BODY_TOKENS:
            return []
        return sorted(TOKEN_TO_ID[t] for t in support(self.tokens))


class EnvironmentTracker:
    def __init__(self):
        self.state = BodyState()

    def update(self, token):
        from .bigint_codec import TOKEN_TO_ID

        if token not in TOKEN_TO_ID:
            raise CodecError("unknown token")
        self.state.consume(TOKEN_TO_ID[token])

    def get_valid_next_tokens(self):
        return set(self.state.allowed())
