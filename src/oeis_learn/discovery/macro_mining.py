"""Bounded mining of exactly expandable typed syntax from verified training views.

Macros are syntactic sharing, with trap/order preserving identical expansion.
No external corpus, handwritten algorithm, or output-equivalence substitution.
"""

from collections import Counter
import re
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.sandbox.logical_ir import Node, parse_logical


def expand_macros(source, dictionary, *, max_expansions=32):
    if (
        set(dictionary)
        != {
            "version",
            "definitions",
            "archive_root_sha256",
            "net_token_benefit",
            "dictionary_sha256",
        }
        or dictionary["version"] != "typed-macros/v1"
        or dictionary["dictionary_sha256"] != digest(dictionary, "dictionary_sha256")
        or len(dictionary["definitions"]) > 64
        or not 1 <= max_expansions <= 32
    ):
        raise ValueError("bound versioned macro dictionary required")
    definitions = dictionary["definitions"]
    count = 0

    def replace(text, ancestors=()):
        nonlocal count

        def one(match):
            nonlocal count
            name = match.group(1)
            if name not in definitions or name in ancestors:
                raise ValueError("missing/cyclic macro dependency")
            count += 1
            if count > max_expansions:
                raise ValueError("macro expansion cap32")
            return replace(definitions[name]["source"], ancestors + (name,))

        return re.sub(r"\(macro ([a-z][a-z0-9_]*)\)", one, text)

    # Check every dependency, including unused dictionary entries.
    for name, record in definitions.items():
        if (
            not re.fullmatch("[a-z][a-z0-9_]*", name)
            or set(record) != {"source", "type", "occurrences", "token_benefit"}
            or record["type"] not in ("wide", "bool", "native_stack")
            or record["token_benefit"] <= 0
        ):
            raise ValueError("typed beneficial macro required")
        old = count
        replace(record["source"], (name,))
        count = old
    return replace(source)


def mine_macros(archive, *, snapshot=None, max_macros=64):
    if not 1 <= max_macros <= 64:
        raise ValueError("macro cap64")
    from oeis_learn.sandbox.wat_ast import parse

    snapshot = archive.freeze() if snapshot is None else snapshot
    counts = Counter()
    types = {}
    # Reading this view enforces provenance, holdouts, and independent execution.
    for _, envelope in archive.training_view(snapshot):
        source = envelope["source"]
        try:
            program = parse_logical(source)
        except ValueError:
            program = parse(source)

            # Native structured nodes carry a codec-checked stack signature.
            def native(n):
                if n.op in ("block", "loop", "if"):
                    result = "( result " + " ".join(n.results) + " ) " if n.results else ""
                    body = " ".join(native(a) for a in n.body)
                    otherwise = " ".join(native(a) for a in n.otherwise)
                    text = (
                        f"( {n.op} {result}"
                        + (
                            f"( then {body} ) ( else {otherwise} )"
                            if n.op == "if" and n.has_else
                            else f"( then {body} )"
                            if n.op == "if"
                            else body
                        )
                        + " )"
                    )
                    counts[text] += 1
                    types[text] = "native_stack"
                    return text
                return n.op + (" " + str(n.arg) if n.arg is not None else "")

            for n in program.instructions:
                native(n)
        else:

            def walk(n):
                if not isinstance(n, Node):
                    return
                if n.op not in (
                    "pure",
                    "stream",
                    "init",
                    "next",
                    "set",
                    "store",
                    "if",
                    "repeat",
                    "while",
                    "return",
                ):
                    text = n.text()
                    counts[text] += 1
                    types[text] = "bool" if n.op in ("eq", "ne", "lt", "le", "gt", "ge") else "wide"
                for a in n.args:
                    walk(a)

            walk(program.root)
    definitions = {}
    benefit = 0
    ordered = []
    for text, count in counts.items():
        tokens = len(re.findall(r"\(|\)|[^\s()]+", text))
        saving = count * (tokens - 4) - (tokens + 4)
        if count >= 2 and saving > 0:
            ordered.append((saving, text, count))
    for saving, text, count in sorted(ordered, key=lambda r: (-r[0], r[1]))[:max_macros]:
        definitions[f"m{len(definitions)}"] = {
            "source": text,
            "type": types[text],
            "occurrences": count,
            "token_benefit": saving,
        }
        benefit += saving
    record = {
        "version": "typed-macros/v1",
        "definitions": definitions,
        "archive_root_sha256": snapshot["root_sha256"],
        "net_token_benefit": benefit,
    }
    record["dictionary_sha256"] = digest(record)
    return record
