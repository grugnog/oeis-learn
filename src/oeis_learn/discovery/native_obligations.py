"""SMT-LIB serialization for independent certificates; solving is native only.

This is not a fitting API. Expressions come from the existing restricted logical
checker. The pinned native process has a fixed per-obligation timeout and no
model/assignment interface.
"""

from dataclasses import dataclass
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess


@dataclass(frozen=True, eq=False)
class Expr:
    text: str
    symbols: frozenset = frozenset()

    def __bool__(self):
        raise TypeError("symbolic condition is not a Python boolean")

    def _op(self, op, other):
        return expression(op, self, other)

    def __add__(self, other):
        return self._op("+", other)

    def __sub__(self, other):
        return self._op("-", other)

    def __mul__(self, other):
        return self._op("*", other)

    def __ge__(self, other):
        return self._op(">=", other)

    def __le__(self, other):
        return self._op("<=", other)

    def __gt__(self, other):
        return self._op(">", other)

    def __lt__(self, other):
        return self._op("<", other)

    def __eq__(self, other):
        return self._op("=", other)

    def __ne__(self, other):
        return Not(self._op("=", other))


def Int(name):
    if not re.fullmatch(r"(?:n|s(?:[0-9]|[12][0-9]|3[01]))", name):
        raise ValueError("controlled certificate symbol required")
    return Expr(name, frozenset([name]))


def IntVal(value):
    if type(value) is not int or abs(value).bit_length() > 4096:
        raise ValueError("bounded exact integer required")
    return Expr(str(value) if value >= 0 else f"(- {-value})")


def expression(op, *args):
    args = [v if isinstance(v, Expr) else IntVal(v) for v in args]
    if op in {"and", "or"} and not args:
        return Expr("true" if op == "and" else "false")
    return Expr(
        "(" + op + " " + " ".join(a.text for a in args) + ")",
        frozenset().union(*(a.symbols for a in args)),
    )


def And(*args):
    return expression("and", *args)


def Not(arg):
    return expression("not", arg)


def Implies(left, right):
    return expression("=>", left, right)


def discharge(condition):
    binary = Path(
        os.environ.get("OEIS_CERTIFICATE_SOLVER", "/opt/oeis-build-info/certificate-solver")
    )
    if not binary.is_file():
        return dict(result="unknown", solver="unavailable:pinned native certificate solver")
    request = dict(declarations=sorted(condition.symbols), assertion=Not(condition).text)
    body = json.dumps(request, separators=(",", ":")).encode()
    if len(body) > 1 << 20:
        raise ValueError("certificate obligation exceeds byte cap")
    try:
        p = subprocess.run(
            [str(binary)],
            input=body,
            capture_output=True,
            timeout=2.5,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        if p.returncode or len(p.stdout) > 4096:
            return dict(result="unknown", solver="native process failure")
        result = json.loads(p.stdout)
        if (
            set(result) != {"result", "solver"}
            or result["result"] not in {"sat", "unsat", "unknown"}
            or not isinstance(result["solver"], str)
        ):
            raise ValueError("invalid native certificate response")
        result["solver"] += ";binary-sha256=" + hashlib.sha256(binary.read_bytes()).hexdigest()
        return result
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return dict(result="unknown", solver="native timeout/protocol failure")
