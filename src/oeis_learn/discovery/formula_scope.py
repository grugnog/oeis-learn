"""Exact interval preimages and original definedness, independent of the CAS."""

import ast
from oeis_learn.discovery.formula_syntax import parse_formula, exact_value, expression_hash
from oeis_learn.experiments.artifacts import compute_canonical_digest


def statement(formulas, coefficients, transforms, domains, assumptions):
    return {
        "formulas": list(formulas),
        "expression_hashes": [expression_hash(f) for f in formulas],
        "coefficients": list(coefficients),
        "transforms": [list(t) for t in transforms],
        "operand_domains": domains,
        "assumptions": assumptions,
        "numeric_semantics": "mathematical_rational",
    }


def common_domain(formulas, transforms, domains):
    lower, upper, exclusions = 0, None, []
    empty = False
    for formula, (a, b), domain in zip(formulas, transforms, domains):
        if type(a) is not int or type(b) is not int or abs(a) > 1000000 or abs(b) > 1000000:
            raise ValueError("bounded exact transforms required")
        if (
            set(domain) != {"integer_only", "lower_bound", "upper_bound"}
            or domain["integer_only"] is not True
        ):
            raise ValueError("integer interval required")
        lo, hi = domain["lower_bound"], domain["upper_bound"]
        if type(lo) is not int or (hi is not None and type(hi) is not int):
            raise ValueError("exact interval bounds required")
        empty |= hi is not None and lo > hi
        if a == 0:
            empty |= b < lo or (hi is not None and b > hi)
        elif a > 0:
            lower = max(lower, -((b - lo) // a))
            if hi is not None:
                upper = (hi - b) // a if upper is None else min(upper, (hi - b) // a)
        else:
            upper = (lo - b) // a if upper is None else min(upper, (lo - b) // a)
            if hi is not None:
                lower = max(lower, -((b - hi) // a))
        tree = parse_formula(formula)
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                exclusions.append({"denominator": ast.unparse(node.right), "scale": a, "shift": b})
        if a == 0:
            try:
                exact_value(tree, b)
            except ZeroDivisionError:
                empty = True
    return {
        "integer_only": True,
        "lower_bound": lower,
        "upper_bound": upper,
        "excluded_points": [],
        "excluded_poles": exclusions,
    }, empty or (upper is not None and lower > upper)


def bind_scope(evidence, formulas, coefficients, transforms, domains, assumptions):
    scoped = statement(formulas, coefficients, transforms, domains, assumptions)
    domain, _ = common_domain(formulas, transforms, domains)
    evidence["statement"] = scoped
    evidence["statement_sha256"] = compute_canonical_digest(scoped)
    evidence["domain"] = domain
    return evidence
