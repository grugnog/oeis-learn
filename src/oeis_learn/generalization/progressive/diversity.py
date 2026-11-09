"""Independent finite-support counts, separate from consumed training exposure."""

from collections import Counter
from .contracts import digest


def summarize(rows, consumed):
    frequencies = Counter(consumed) if not isinstance(consumed, dict) else Counter(consumed)
    if (
        len(rows) > 262144
        or len(frequencies) > 262144
        or any(type(n) is not int or not 0 < n <= 2**63 - 1 for n in frequencies.values())
    ):
        raise ValueError("bounded diversity inventory required")
    table = {r["draw_identity"]: r for r in rows}
    if any(table[r["draw_identity"]] != r for r in rows) or any(
        i not in table for i in frequencies
    ):
        raise ValueError("consumed row outside original support")

    def count(selected):
        operations = Counter()
        events = Counter()
        bands = Counter()
        compositions = set()
        from oeis_learn.synthesis.reference_parser import parse

        for row, frequency in selected:
            p = parse(row["source"])
            edges = set()

            def visit(node):
                if isinstance(node, tuple):
                    op = node[0]
                    operations[op] += frequency
                    for child in node[1:]:
                        if isinstance(child, tuple):
                            edges.add((op, child[0]))
                            visit(child)

            visit(p.result)
            for statement in p.statements:
                visit(statement)
            compositions.add(
                digest([p.result[0] if isinstance(p.result, tuple) else p.result, sorted(edges)])
            )
            events.update({k: v * frequency for k, v in row.get("control_events", {}).items()})
            bands[
                "structured" if {"loop", "select"} & set(row["operations"]) else "arithmetic"
            ] += frequency
        return dict(
            examples=sum(n for row, n in selected),
            numeric_variants=len({digest(r["source"]) for r, n in selected}),
            canonical_structures=len({r["structure"] for r, n in selected}),
            finite_behavior_clusters=len({digest(r["values"]) for r, n in selected}),
            compositions=len(compositions),
            syntactic_operation_counts=dict(sorted(operations.items())),
            execution_events=dict(sorted(events.items())),
            difficulty_bands=dict(sorted(bands.items())),
        )

    return dict(
        available=count([(r, 1) for r in table.values()]),
        consumed=count([(table[i], n) for i, n in frequencies.items()]),
        unique_consumed=len(frequencies),
        behavior_scope="finite_100_terms_only",
        difficulty_rule="prospective syntax: structured if loop/select, otherwise arithmetic",
    )
