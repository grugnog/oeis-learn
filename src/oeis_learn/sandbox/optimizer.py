"""Conservative typed rewrites with replayable supported-domain evidence."""

from dataclasses import replace
import math
from oeis_learn.data.models import CanonicalProgramArtifact
from oeis_learn.decoder.wat_grammar import tokenize_wat
from oeis_learn.sandbox.wat_ast import parse, ExecutionFailure
from oeis_learn.sandbox.pipeline import digest_bytes


def _reads(nodes):
    return {node.arg for node in nodes if node.op == "local.get"} | set().union(
        *(_reads(n.body) | _reads(n.otherwise) for n in nodes)
    )


def _rewrite(nodes, live_out=frozenset(), allow_dead=True):
    live = set(live_out)
    backward = []
    for node in reversed(nodes):
        if node.op in ("block", "loop", "if"):
            # Preserve all writes through control flow. This is a conservative
            # fixed point: every read in the region is live at its boundary.
            live.update(_reads(node.body) | _reads(node.otherwise))
            node = replace(
                node,
                body=_rewrite(node.body, live, False),
                otherwise=_rewrite(node.otherwise, live, False),
            )
        elif node.op in ("br", "br_if", "return"):
            allow_dead = False
        elif node.op == "local.get":
            live.add(node.arg)
        elif node.op in ("local.set", "local.tee"):
            if allow_dead and node.op == "local.set" and node.arg not in live:
                node = replace(node, op="drop", arg=None)
            else:
                live.discard(node.arg)
        backward.append(node)
    nodes = tuple(reversed(backward))
    result = []
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if node.op == "nop":
            index += 1
            continue
        if (
            node.op in ("i32.const", "i64.const", "local.get")
            and index + 1 < len(nodes)
            and nodes[index + 1].op == "drop"
        ):
            index += 2
            continue
        result.append(node)
        index += 1
    return tuple(result)


def _render(nodes):
    words = []
    for node in nodes:
        if node.op in ("block", "loop", "if"):
            result = " ( result " + " ".join(node.results) + " )" if node.results else ""
            if node.op == "if":
                body = " ( then " + _render(node.body) + " )"
                if node.has_else:
                    body += " ( else " + _render(node.otherwise) + " )"
            else:
                body = " " + _render(node.body)
            words.append("( " + node.op + result + body + " )")
        else:
            words.append(node.op + (" " + str(node.arg) if node.arg is not None else ""))
    return " ".join(words)


def optimize_wat_program(wat_code, hard_waste_threshold=0.30):
    if (
        type(hard_waste_threshold) not in (float, int)
        or not math.isfinite(hard_waste_threshold)
        or not 0 <= hard_waste_threshold <= 1
    ):
        raise ValueError("waste threshold must be finite in 0..1")
    optimized = wat_code
    passes = []
    evidence = None
    try:
        program = parse(wat_code)
        rewritten = _rewrite(program.instructions)
        if rewritten != program.instructions:
            optimized = parse(_render(rewritten)).canonical_source
            evidence = {
                "version": "typed-rewrite/v1",
                "source": program.canonical_source,
                "result": optimized,
                "source_sha256": digest_bytes(program.canonical_source.encode()),
                "result_sha256": digest_bytes(optimized.encode()),
                "semantics": "typed_value_and_trap_equivalence",
                "rules": ["nop", "total_push_drop", "dead_local_set_to_drop"],
                "checker": "typed-rewrite-checker/v1",
            }
            from oeis_learn.sandbox.rewrite_checker import check_rewrite

            if not check_rewrite(evidence):
                raise ValueError("independent rewrite checker rejected transformation")
            passes = ["typed-nop-and-pure-drop-v1"]
            if (
                any(n.op == "local.set" for n in program.instructions)
                and rewritten != program.instructions
            ):
                passes.append("typed-dead-local-write-v1")
    except ExecutionFailure:
        pass
    raw_count = len(tokenize_wat(wat_code))
    opt_count = len(tokenize_wat(optimized))
    waste = max(0, raw_count - opt_count) / max(1, raw_count)
    return CanonicalProgramArtifact(
        wat_code,
        optimized,
        raw_count,
        opt_count,
        waste,
        passes,
        waste > hard_waste_threshold,
        evidence,
    )
