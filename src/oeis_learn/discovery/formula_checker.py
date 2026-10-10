"""Independent rational-identity certificate checking with exact fractions.

After multiplying by all uncancelled denominators, a degree-D polynomial
vanishing at D+1 distinct defined points is zero. No CAS reduction is trusted.
"""

from oeis_learn.discovery.formula_syntax import (
    parse_formula,
    degree_bound,
    exact_value,
    expression_hash,
)


def relation_degree(trees):
    degrees = [degree_bound(t) for t in trees]
    return max(
        a + sum(d for j, (_, d) in enumerate(degrees) if j != i) for i, (a, b) in enumerate(degrees)
    )


def check_certificate(certificate):
    try:
        if (
            not isinstance(certificate, dict)
            or set(certificate)
            != {
                "version",
                "formulas",
                "expression_hashes",
                "coefficients",
                "transforms",
                "degree_bound",
                "points",
            }
            or certificate["version"] != "rational-evaluation/v1"
        ):
            return False
        formulas = certificate["formulas"]
        coeffs = certificate["coefficients"]
        transforms = certificate["transforms"]
        if (
            not isinstance(formulas, list)
            or not 1 <= len(formulas) <= 16
            or len(coeffs) != len(formulas)
            or len(transforms) != len(formulas)
        ):
            return False
        if any(type(c) is not int or c.bit_length() > 256 for c in coeffs):
            return False
        if any(
            not isinstance(t, (list, tuple))
            or len(t) != 2
            or any(type(v) is not int or abs(v) > 1000000 for v in t)
            for t in transforms
        ):
            return False
        if certificate["expression_hashes"] != [expression_hash(f) for f in formulas]:
            return False
        trees = [parse_formula(f) for f in formulas]
        degree = relation_degree(trees)
        if (
            degree > 512
            or type(certificate["degree_bound"]) is not int
            or certificate["degree_bound"] != degree
        ):
            return False
        points = certificate["points"]
        if (
            not isinstance(points, list)
            or len(points) != degree + 1
            or any(type(n) is not int or abs(n) > 1000000 for n in points)
            or len(set(points)) != len(points)
        ):
            return False
        return all(
            sum(c * exact_value(t, a * n + b) for c, t, (a, b) in zip(coeffs, trees, transforms))
            == 0
            for n in points
        )
    except (ValueError, TypeError, KeyError, ZeroDivisionError, SyntaxError, OverflowError):
        return False


def check_scoped_evidence(record):
    """Bind certificate/witness to its original statement and defined domain."""
    from fractions import Fraction
    from oeis_learn.discovery.formula_scope import common_domain
    from oeis_learn.experiments.artifacts import compute_canonical_digest

    try:
        certificate = record.get("certificate")
        envelope = (
            record.get("checker_report") if record.get("kind") == "proof_evidence" else record
        )
        statement = envelope["statement"]
        if set(statement) != {
            "formulas",
            "expression_hashes",
            "coefficients",
            "transforms",
            "operand_domains",
            "assumptions",
            "numeric_semantics",
        }:
            return False
        if compute_canonical_digest(statement) != record["statement_sha256"]:
            return False
        formulas, coeffs, transforms, domains = (
            statement[k] for k in ("formulas", "coefficients", "transforms", "operand_domains")
        )
        if not 1 <= len(formulas) <= 16 or any(
            len(v) != len(formulas) for v in (coeffs, transforms, domains)
        ):
            return False
        if any(type(c) is not int or c.bit_length() > 256 for c in coeffs):
            return False
        hashes = [expression_hash(f) for f in formulas]
        if (
            statement["expression_hashes"] != hashes
            or statement["numeric_semantics"] != "mathematical_rational"
        ):
            return False
        domain, empty = common_domain(formulas, transforms, domains)
        if domain != record["domain"] or record["assumptions"] != statement["assumptions"] or empty:
            return False
        if record["numeric_semantics"] != statement["numeric_semantics"]:
            return False
        if record.get("kind") == "proof_evidence":
            if record["expression_sha256"] != hashes or record["transforms"] != [
                {"scale": a, "shift": b} for a, b in transforms
            ]:
                return False
        elif (
            record["expression_hashes"] != hashes
            or [list(t) for t in record["transforms"]] != transforms
        ):
            return False
        result = record.get("result", record.get("outcome"))
        if result in ("verified", "FORMULA_IDENTITY"):
            if not certificate or any(
                certificate[k] != statement[k]
                for k in ("formulas", "expression_hashes", "coefficients", "transforms")
            ):
                return False
            # The rational certificate is a global algebraic identity, and the
            # statement retains the original expressions' definedness restriction.
            return check_certificate(certificate)
        if result in ("counterexample", "COUNTEREXAMPLE"):
            witness = record["witness"]
            index = witness["index"]
            if (
                type(index) is not int
                or index < domain["lower_bound"]
                or (domain["upper_bound"] is not None and index > domain["upper_bound"])
            ):
                return False
            values = [
                exact_value(parse_formula(f), a * index + b)
                for f, (a, b) in zip(formulas, transforms)
            ]
            residual = sum(c * v for c, v in zip(coeffs, values))
            return (
                residual != 0
                and witness["values"] == [str(v) for v in values]
                and Fraction(witness["residual"]) == residual
            )
        return False
    except (ValueError, TypeError, KeyError, ZeroDivisionError, SyntaxError, OverflowError):
        return False


def qualified_proof(evidence):
    from oeis_learn.experiments.qualified_models import seal_record

    result = {
        "FORMULA_IDENTITY": "verified",
        "COUNTEREXAMPLE": "counterexample",
        "UNKNOWN": "unknown",
        "UNSUPPORTED": "unsupported",
        "TIMEOUT": "timeout",
        "EMPTY_DOMAIN": "empty_domain",
    }[evidence["outcome"]]
    return seal_record(
        "proof_evidence",
        statement_sha256=evidence["statement_sha256"],
        expression_sha256=evidence["expression_hashes"],
        program_sha256=[],
        domain=evidence["domain"],
        assumptions=evidence["assumptions"],
        transforms=[{"scale": a, "shift": b} for a, b in evidence["transforms"]],
        numeric_semantics="mathematical_rational",
        evidence_kind="formula_identity",
        checker="fraction-checker",
        checker_version="fraction-checker/v1",
        budget_ns=2000000000,
        result=result,
        certificate=evidence.get("certificate"),
        witness=evidence.get("witness"),
        checker_report={"statement": evidence["statement"]}
        if result in ("verified", "counterexample")
        else None,
    )
