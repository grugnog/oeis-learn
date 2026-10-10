"""Exact straight-line integer polynomial extraction from typed logical IR."""

import sympy as sp
from oeis_learn.sandbox.logical_ir import parse_logical

N = sp.Symbol("n", integer=True)


def extract_polynomial(source):
    program = parse_logical(source)
    if program.profile != "pure":
        raise NotImplementedError("straight-line pure polynomial profile")
    slots = [sp.Integer(0)] * 32
    intermediates = []
    poles = []

    def expr(node):
        op, a = node.op, node.args
        if op == "const":
            value = sp.Integer(a[0])
        elif op == "index":
            value = N
        elif op == "get":
            value = slots[a[0]]
        elif op in ("add", "sub", "mul", "div"):
            x, y = expr(a[0]), expr(a[1])
            if op == "add":
                value = x + y
            elif op == "sub":
                value = x - y
            elif op == "mul":
                value = x * y
            else:
                if y == 0:
                    raise ValueError("identically zero divisor")
                roots = sp.polys.polytools.ground_roots(y, N)
                poles.extend(int(r) for r in roots if r.is_integer)
                value = sp.cancel(x / y)
                # Logical division truncates: only exact integer polynomial
                # quotients are mathematical polynomial programs.
                try:
                    p = sp.Poly(value, N, domain=sp.ZZ)
                except (sp.PolynomialError, sp.CoercionFailed) as exc:
                    raise NotImplementedError("nonexact integer division") from exc
                value = p.as_expr()
        else:
            raise NotImplementedError("unsupported polynomial expression")
        value = sp.expand(value)
        intermediates.append(value)
        return value

    for node in program.root.args[:-1]:
        if node.op != "set":
            raise NotImplementedError("straight-line slot assignments only")
        slots[node.args[0]] = expr(node.args[1])
    result = expr(program.root.args[-1].args[0])
    coefficients = [str(c) for c in reversed(sp.Poly(result, N, domain=sp.ZZ).all_coeffs())]
    return {
        "coefficients": coefficients,
        "excluded_indices": sorted(set(poles)),
        "intermediates": [str(v) for v in intermediates],
    }


def check_polynomial(programs, coefficients, domain):
    extracted = [extract_polynomial(p) for p in programs]
    residual = sp.Integer(0)
    for c, item in zip(coefficients, extracted):
        residual += c * sum(int(v) * N**i for i, v in enumerate(item["coefficients"]))
    residual = sp.expand(residual)
    excluded = sorted({i for item in extracted for i in item["excluded_indices"]})
    if domain.get("excluded_indices", []) != excluded:
        raise ValueError("all intermediate denominator poles must be declared")
    return residual == 0, {
        "polynomials": extracted,
        "residual_coefficients": [str(v) for v in reversed(sp.Poly(residual, N).all_coeffs())],
    }
