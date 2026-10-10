"""Exact linear state-space invariant checking, recognized from full transitions."""

import sympy as sp
from oeis_learn.sandbox.logical_ir import parse_logical


def extract_state_space(source):
    program = parse_logical(source)
    if program.profile != "stream":
        raise NotImplementedError("stream state-space profile required")
    state = list(sp.symbols("s0:32"))
    initial = [sp.Integer(0)] * 32

    def expression(n, slots):
        op, a = n.op, n.args
        if op == "const":
            return sp.Integer(a[0])
        if op == "get":
            return slots[a[0]]
        if op in ("add", "sub", "mul"):
            x, y = expression(a[0], slots), expression(a[1], slots)
            value = {"add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y}[op]()
            if sp.Poly(value, *state).total_degree() > 1:
                raise NotImplementedError("nonlinear recurrence")
            return sp.expand(value)
        raise NotImplementedError("unsupported recurrence operation/indexing")

    for n in program.root.args[0].args:
        if n.op != "set":
            raise NotImplementedError("constant initial slots required")
        initial[n.args[0]] = expression(n.args[1], initial)
    slots = list(state)
    for n in program.root.args[1].args[:-1]:
        if n.op != "set":
            raise NotImplementedError("linear slot transitions required")
        slots[n.args[0]] = expression(n.args[1], slots)
    output = expression(program.root.args[1].args[-1].args[0], slots)

    # Homogeneous coordinate represents affine transitions exactly.
    def row(expr):
        return [expr.coeff(s) for s in state] + [expr.subs(dict.fromkeys(state, 0))]

    matrix = sp.Matrix([row(e) for e in slots] + [[0] * 32 + [1]])
    return matrix, sp.Matrix(initial + [sp.Integer(1)]), sp.Matrix([row(output)])


def check_recurrence(programs, coefficients, domain):
    if domain["start"] != 0 or domain.get("excluded_indices"):
        raise NotImplementedError("recurrence origin zero without exclusions required")
    systems = [extract_state_space(p) for p in programs]
    matrix = sp.diag(*(m for m, _, _ in systems))
    initial = sp.Matrix.vstack(*(s for _, s, _ in systems))
    output = sp.Matrix.hstack(*(c * o for c, (_, _, o) in zip(coefficients, systems)))
    # Cayley-Hamilton: dimension initial zero outputs suffice for this exact
    # extracted rational-linear transition. No recurrence inferred from samples.
    state = initial
    observations = []
    for _ in range(matrix.rows):
        observations.append(str((output * state)[0]))
        state = matrix * state
    return all(v == "0" for v in observations), {
        "dimension": matrix.rows,
        "initial_outputs": observations,
        "matrix": [[str(v) for v in row] for row in matrix.tolist()],
        "initial_state": [str(v) for v in initial],
        "output_row": [str(v) for v in output],
    }
