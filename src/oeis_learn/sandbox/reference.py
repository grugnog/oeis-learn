"""Independent bounded AST interpreter. No Wasmtime/lowering/production limbs.

Native words use Python integer modulo arithmetic. Logical wide macros are
computed exactly and range checked before the result is split into words.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from oeis_learn.sandbox.wat_ast import Program, ExecutionFailure
from oeis_learn.experiments.profiles import I64_LOCAL_NAMES, I32_LOCAL_NAMES


@dataclass(frozen=True)
class ReferenceTerm:
    value: int | None
    steps: int
    outcome: str | None = None
    reason: str | None = None


class _Branch(Exception):
    def __init__(self, depth, values):
        self.depth, self.values = depth, values


class Interpreter:
    def __init__(self, step_limit: int, deadline_ns: int):
        if type(step_limit) is not int or step_limit < 0:
            raise ValueError("reference steps must be nonnegative")
        self.limit, self.deadline, self.steps = step_limit, deadline_ns, 0
        self.stack, self.frames = [], [("func", 4)]
        self.locals = dict.fromkeys((*I64_LOCAL_NAMES, *I32_LOCAL_NAMES), 0)

    def tick(self):
        if self.steps >= self.limit:
            raise ExecutionFailure("execution_limit", "reference_steps")
        if time.monotonic_ns() >= self.deadline:
            raise ExecutionFailure("execution_limit", "reference_deadline")
        self.steps += 1

    @staticmethod
    def signed(value, bits):
        value %= 2**bits
        return value if value < 2 ** (bits - 1) else value - 2**bits

    def wide(self):
        values = self.stack[-4:]
        del self.stack[-4:]
        unsigned = sum((v % 2**64) * 2 ** (64 * i) for i, v in enumerate(values))
        return unsigned if unsigned < 2**255 else unsigned - 2**256

    def push_wide(self, value):
        if not -(2**255) <= value < 2**255:
            raise ExecutionFailure("numeric_limit", "checked_signed_256_overflow")
        for i in range(4):
            self.stack.append(self.signed(value // 2 ** (64 * i), 64))

    def branch(self, depth):
        kind, arity = self.frames[-1 - depth]
        count = 0 if kind == "loop" else arity
        values = self.stack[-count:] if count else []
        raise _Branch(depth, values)

    def native(self, op):
        typ, name = op.split(".")
        bits = 32 if typ == "i32" else 64
        if name.startswith("extend_i32"):
            value = self.stack.pop()
            self.stack.append(value if name.endswith("_s") else value % 2**32)
            return
        if name == "wrap_i64":
            self.stack.append(self.signed(self.stack.pop(), 32))
            return
        if name == "eqz":
            self.stack.append(int(self.stack.pop() == 0))
            return
        b, a = self.stack.pop(), self.stack.pop()
        if name in ("add", "sub", "mul"):
            value = a + b if name == "add" else a - b if name == "sub" else a * b
        elif name in ("and", "or", "xor"):
            value = a & b if name == "and" else a | b if name == "or" else a ^ b
        elif name in ("shl", "shr_s", "shr_u"):
            shift = b % bits
            value = (
                a << shift
                if name == "shl"
                else a >> shift
                if name == "shr_s"
                else (a % 2**bits) >> shift
            )
        elif name.startswith(("div_", "rem_")):
            if name.endswith("_u"):
                a, b = a % 2**bits, b % 2**bits
            if b == 0:
                raise ExecutionFailure("runtime_failure", "integer_divide_by_zero")
            if name == "div_s" and a == -(2 ** (bits - 1)) and b == -1:
                raise ExecutionFailure("runtime_failure", "integer_division_overflow")
            quotient = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
            value = quotient if name.startswith("div_") else a - quotient * b
        else:
            value = {
                "eq": a == b,
                "ne": a != b,
                "lt_s": a < b,
                "gt_s": a > b,
                "le_s": a <= b,
                "ge_s": a >= b,
            }[name]
            self.stack.append(int(value))
            return
        self.stack.append(self.signed(value, bits))

    def sequence(self, nodes):
        for node in nodes:
            self.tick()
            op = node.op
            if op in ("block", "loop", "if"):
                condition = self.stack.pop() if op == "if" else 1
                baseline = len(self.stack)
                self.frames.append((op, len(node.results)))
                try:
                    while True:
                        try:
                            self.sequence(node.body if condition else node.otherwise)
                            break
                        except _Branch as branch:
                            del self.stack[baseline:]
                            if branch.depth:
                                branch.depth -= 1
                                raise
                            self.stack.extend(branch.values)
                            if op != "loop":
                                break
                            self.tick()  # finite cost even for a bare backedge
                finally:
                    self.frames.pop()
            elif op == "br" or (op == "br_if" and self.stack.pop()):
                self.branch(node.arg)
            elif op == "return":
                self.branch(len(self.frames) - 1)
            elif op in ("br_if", "nop"):
                pass
            elif op == "unreachable":
                raise ExecutionFailure("runtime_failure", "unreachable")
            elif op == "drop":
                self.stack.pop()
            elif op.startswith("local."):
                if op == "local.get":
                    self.stack.append(self.locals[node.arg])
                elif op == "local.set":
                    self.locals[node.arg] = self.stack.pop()
                else:
                    self.locals[node.arg] = self.stack[-1]
            elif op in ("i32.const", "i64.const"):
                self.stack.append(node.arg)
            elif op == "i256.const":
                self.push_wide(node.arg)
            elif op == "i256.zero":
                self.push_wide(0)
            elif op in ("i256.add", "i256.sub"):
                b, a = self.wide(), self.wide()
                self.push_wide(a + b if op == "i256.add" else a - b)
            elif op == "i256.mul_scalar":
                b = self.stack.pop()
                self.push_wide(self.wide() * b)
            else:
                self.native(op)

    def run(self, program, index):
        self.locals["$n"] = index
        try:
            try:
                self.sequence(program.instructions)
            except _Branch as branch:
                if branch.depth != 0:
                    raise ExecutionFailure("runtime_failure", "reference_invalid_branch")
                self.stack = branch.values
            if len(self.stack) != 4:
                raise ExecutionFailure("runtime_failure", "reference_wrong_arity")
            return ReferenceTerm(self.wide(), self.steps)
        except ExecutionFailure as failure:
            return ReferenceTerm(None, self.steps, failure.outcome, failure.reason)


def execute_term(
    program: Program, index: int, *, step_limit=1_000_000, deadline_ns=None
) -> ReferenceTerm:
    if type(index) is not int or not 0 <= index <= 99:
        raise ValueError("index must be an exact rebased integer in 0..99")
    return Interpreter(
        step_limit, deadline_ns if deadline_ns is not None else time.monotonic_ns() + 2_000_000_000
    ).run(program, index)
