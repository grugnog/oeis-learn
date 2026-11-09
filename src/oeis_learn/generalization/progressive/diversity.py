"""Independent finite-support counts, separate from consumed training exposure."""

from collections import Counter
from .contracts import digest


def summarize(rows, consumed):
    if len(rows) > 262144 or len(consumed) > 1000000:
        raise ValueError("bounded diversity inventory required")
    table = {r["draw_identity"]: r for r in rows}
    if len(table) != len(rows) or any(i not in table for i in consumed):
        raise ValueError("consumed row outside original support")

    def count(selected):
        operations = Counter()
        events = Counter()
        bands = Counter()
        compositions = set()
        from oeis_learn.synthesis.reference_parser import parse

        for row in selected:
            p = parse(row["source"])
            edges = set()

            def visit(node):
                if isinstance(node, tuple):
                    op = node[0]
                    operations[op] += 1
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
            events.update(row.get("control_events", {}))
            bands[
                "structured" if {"loop", "select"} & set(row["operations"]) else "arithmetic"
            ] += 1
        return dict(
            examples=len(selected),
            numeric_variants=len({digest(r["source"]) for r in selected}),
            canonical_structures=len({r["structure"] for r in selected}),
            finite_behavior_clusters=len({digest(r["values"]) for r in selected}),
            compositions=len(compositions),
            syntactic_operation_counts=dict(sorted(operations.items())),
            execution_events=dict(sorted(events.items())),
            difficulty_bands=dict(sorted(bands.items())),
        )

    return dict(
        available=count(rows),
        consumed=count([table[i] for i in consumed]),
        unique_consumed=len(set(consumed)),
        behavior_scope="finite_100_terms_only",
        difficulty_rule="prospective syntax: structured if loop/select, otherwise arithmetic",
    )
