"""Restricted polynomial invariants/ranking functions with exact SMT obligations.

One typed while loop and slot assignments. Untrusted templates are proposals;
initiation, consecution, exit, decreasing/nonnegative ranking and intermediate
checked ranges are independently discharged. Solver UNKNOWN is preserved.
"""

import z3
from oeis_learn.sandbox.logical_ir import parse_logical


def check_loop(source, domain, template):
    if (
        set(template) != {"invariants", "ranking", "postcondition"}
        or not 1 <= len(template["invariants"]) <= 64
    ):
        raise ValueError("bounded loop templates required")
    program = parse_logical(source)
    if program.profile != "pure":
        raise NotImplementedError("pure restricted loop required")
    loops = [i for i, n in enumerate(program.root.args) if n.op == "while"]
    if len(loops) != 1:
        raise NotImplementedError("one restricted loop required")
    position = loops[0]
    loop = program.root.args[position]
    n = z3.Int("n")
    state = [z3.Int(f"s{i}") for i in range(32)]
    range_terms = []

    def expr(node, slots):
        op, a = node.op, node.args
        if op == "const":
            v = z3.IntVal(a[0])
        elif op == "index":
            v = n
        elif op == "get":
            v = slots[a[0]]
        elif op in ("add", "sub", "mul"):
            x, y = expr(a[0], slots), expr(a[1], slots)
            v = {"add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y}[op]()
        elif op in ("eq", "ne", "lt", "le", "gt", "ge"):
            x, y = expr(a[0], slots), expr(a[1], slots)
            return {
                "eq": lambda: x == y,
                "ne": lambda: x != y,
                "lt": lambda: x < y,
                "le": lambda: x <= y,
                "gt": lambda: x > y,
                "ge": lambda: x >= y,
            }[op]()
        else:
            raise NotImplementedError("restricted affine/polynomial loop expression")
        range_terms.append(v)
        return v

    def statements(nodes, slots):
        slots = list(slots)
        for node in nodes:
            if node.op != "set":
                raise NotImplementedError("restricted slot assignments")
            slots[node.args[0]] = expr(node.args[1], slots)
        return slots

    # Templates are typed expressions in the same controlled environment.
    def template_node(text, boolean=False):
        if boolean:
            p = parse_logical("(pure (return (select " + text + " (const 1) (const 0))))")
            return p.root.args[0].args[0].args[0]
        return parse_logical("(pure (return " + text + "))").root.args[0].args[0]

    invariants = [template_node(t, True) for t in template["invariants"]]
    ranking = template_node(template["ranking"])
    post = template_node(template["postcondition"], True)
    initial = statements(program.root.args[:position], [z3.IntVal(0)] * 32)
    initialization_ranges = list(range_terms)
    range_terms.clear()
    inv = z3.And(*[expr(t, state) for t in invariants])
    range_terms.clear()
    guard = expr(loop.args[0], state)
    guard_ranges = list(range_terms)
    range_terms.clear()
    after = statements(loop.args[1:], state)
    transition_ranges = list(range_terms)
    range_terms.clear()
    r = expr(ranking, state)
    ra = expr(ranking, after)
    range_terms.clear()
    bounds = z3.And(
        n >= domain["start"], *([n <= domain["end"]] if domain["end"] is not None else [])
    )
    bounds = z3.And(bounds, *[n != x for x in domain["excluded_indices"]])
    minimum, maximum = -(1 << 255), (1 << 255) - 1

    def safe(values):
        return z3.And(*[z3.And(v >= minimum, v <= maximum) for v in values])

    assumptions = z3.And(bounds, safe(state))
    inv_initial = z3.And(*[expr(t, initial) for t in invariants])
    range_terms.clear()
    inv_after = z3.And(*[expr(t, after) for t in invariants])
    range_terms.clear()
    post_exit = expr(post, state)
    range_terms.clear()
    if position + 2 != len(program.root.args):
        raise NotImplementedError("loop immediately precedes return")
    expr(program.root.args[-1].args[0], state)
    exit_ranges = list(range_terms)
    obligations = {
        "initiation": z3.Implies(bounds, z3.And(inv_initial, safe(initialization_ranges))),
        "consecution": z3.Implies(
            z3.And(assumptions, inv, guard),
            z3.And(inv_after, safe(guard_ranges + transition_ranges)),
        ),
        "ranking": z3.Implies(z3.And(assumptions, inv, guard), z3.And(r >= 0, ra >= 0, ra < r)),
        "exit": z3.Implies(
            z3.And(assumptions, inv, z3.Not(guard)), z3.And(post_exit, safe(exit_ranges))
        ),
    }
    results = {}
    for name, condition in obligations.items():
        solver = z3.Solver()
        solver.set(timeout=1500)
        solver.add(z3.Not(condition))
        result = solver.check()
        results[name] = str(result)
        if result != z3.unsat:
            return False, {
                "obligations": results,
                "reason": "unproved obligation",
                "solver": z3.get_version_string(),
            }
    return True, {
        "obligations": results,
        "solver": z3.get_version_string(),
        "scope": "checked_logical_ir_termination_and_postcondition",
    }


def propose_loop_templates(source, domain, *, cap=64):
    """Bounded task-independent countdown candidates; every obligation is replayed."""
    if not 1 <= cap <= 64:
        raise ValueError("loop template cap64")
    program = parse_logical(source)
    if program.profile != "pure":
        return {"result": "unsupported", "attempts": 0, "template": None}
    attempts = 0
    for slot in range(32):
        for lower in (0, 1):
            if attempts >= cap:
                return {"result": "unknown", "attempts": attempts, "template": None}
            attempts += 1
            template = {
                "invariants": [f"(ge (get {slot}) (const {lower}))"],
                "ranking": f"(sub (get {slot}) (const {lower}))",
                "postcondition": f"(eq (get {slot}) (const {lower}))",
            }
            try:
                valid, _ = check_loop(source, domain, template)
            except (NotImplementedError, ValueError):
                continue
            if valid:
                return {"result": "verified", "attempts": attempts, "template": template}
    return {"result": "unknown", "attempts": attempts, "template": None}
