"""Audited known-structure projections for private synthetic practice families."""

import re

from oeis_learn.data.bigint_programs import render
from oeis_learn.decoder.bigint_codec import decode_body
from oeis_learn.generalization.targets import ground, pair
from oeis_learn.pipeline_validation.identity import validate
from oeis_learn.synthesis.contracts import canonical_integer
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.synthesis.reference_parser import parse
from .contracts import digest, seal
from .velocity_records import visible_prompt


def _indices(indices):
    if not isinstance(indices, list) or len(indices) != 100:
        raise ValueError("complete original practice indices required")
    ns = [canonical_integer(n) for n in indices]
    if ns != list(range(ns[0], ns[0] + 100)):
        raise ValueError("consecutive original practice indices required")


def oracle_projection(witness, commitment):
    """Check the private occurrence map, then expose just a typed skeleton and 20 indexed values."""
    fields = {"version", "family", "source", "indices", "values", "sharing", "occurrences"}
    validate(witness, fields, "round029-practice-witness/v1")
    if witness["identity"] != commitment:
        raise ValueError("practice witness differs from the registered commitment")
    indices = witness["indices"]
    _indices(indices)
    if not witness["family"] or witness["sharing"] not in {"explicit", "distinct-occurrences"}:
        raise ValueError("registered practice family and sharing policy required")
    truth = evaluate(witness["source"], indices)
    if truth.outcome != "ok" or list(truth.values) != witness["values"]:
        raise ValueError("private practice witness failed independent execution")
    prompt = visible_prompt(dict(indices=indices[:20], values=witness["values"][:20]))
    program = parse(witness["source"])
    tree = (
        "program",
        *(("local", name, kind) for name, kind in program.declarations),
        *program.statements,
        ("return", program.result),
    )
    mapped = witness["occurrences"]
    assignments = {}
    position = unknowns = 0
    unsupported = None
    visible_literals = {"first_index": indices[0], "last_visible_index": indices[19]}

    def project(node):
        nonlocal position, unknowns, unsupported
        if isinstance(node, str):
            return node
        if node[0] not in ("int", "control"):
            return tuple(project(child) for child in node)
        if position >= len(mapped):
            raise ValueError("missing private literal occurrence")
        item = mapped[position]
        if (
            set(item) != {"ordinal", "use"}
            or type(item["ordinal"]) is not int
            or item["ordinal"] != position
            or not isinstance(item["use"], str)
        ):
            raise ValueError("literal occurrence order differs from the parsed witness")
        position += 1
        use = item["use"]
        if re.fullmatch(r"p[0-9]+", use):
            unknowns += 1
            if node[0] == "control":
                unsupported = "unsupported_control"
            if use in assignments:
                if assignments[use] != node[1]:
                    raise ValueError("shared coefficient has inconsistent private values")
                if witness["sharing"] == "distinct-occurrences":
                    raise ValueError("private coefficient sharing differs from the registration")
            assignments[use] = node[1]
            return ("param.get", use)
        if use not in visible_literals or node[1] != visible_literals[use]:
            unsupported = "unsupported_structural_literal"
        return node

    projected = project(tree)
    if position != len(mapped):
        raise ValueError("unaccounted private literal occurrence")
    if unknowns > 8 or any(key not in {f"p{i}" for i in range(8)} for key in assignments):
        return dict(disposition="unsupported_occurrence_count")
    if unsupported:
        return dict(disposition=unsupported)
    if not unknowns:
        return dict(disposition="zero_unknowns")
    # pair reparses and grounds the skeleton, checks the codec and propagates control
    # dependence through locals and loops. No coefficient fitting happens in the preparer.
    proof = pair(
        witness["source"],
        render(projected),
        assignments,
        indices=indices,
        sharing=witness["sharing"],
    )
    if proof["protected_parameters"]:
        return dict(disposition="unsupported_control")
    return dict(
        disposition="ready", skeleton=dict(tokens=proof["parameter_tokens"]), visible=prompt
    )


def witnessed_ambiguity(witness, commitment, first, second):
    """Return finite evidence only when both legal assignments fit the true prefix and then diverge."""
    projection = oracle_projection(witness, commitment)
    if projection["disposition"] != "ready":
        raise ValueError("an audited oracle skeleton is required")
    prompt = projection["visible"]
    indices = witness["indices"]
    skeleton = decode_body(projection["skeleton"]["tokens"])
    sources = [ground(skeleton, values) for values in (first, second)]
    results = [evaluate(source, indices) for source in sources]
    if any(r.outcome != "ok" or list(r.values[:20]) != prompt["values"] for r in results):
        return None
    left, right = results
    different = next((i for i in range(20, 100) if left.values[i] != right.values[i]), None)
    if different is None:
        return None
    return seal(
        dict(
            disposition="witnessed_ambiguity",
            groundings=[digest(s) for s in sources],
            first_different_index=indices[different],
        )
    )
