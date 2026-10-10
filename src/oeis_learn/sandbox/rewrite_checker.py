"""Independent structural checking of the small total/effect-safe rewrite catalog."""

from dataclasses import replace
from oeis_learn.sandbox.wat_ast import parse
from oeis_learn.sandbox.pipeline import digest_bytes


def _all_reads(nodes):
    result = set()
    for node in nodes:
        if node.op == "local.get":
            result.add(node.arg)
        result.update(_all_reads(node.body))
        result.update(_all_reads(node.otherwise))
    return result


def _check_sequence(before, after, live_out=frozenset(), allow_dead=True):
    # Compute certificates' permitted dead writes independently by suffix
    # liveness, then match deletions without accepting arbitrary source changes.
    live = set(live_out)
    expected = []
    for node in reversed(before):
        if node.op in ("block", "loop", "if"):
            live.update(_all_reads(node.body) | _all_reads(node.otherwise))
            # Control shape/order must match; recursive check is done below.
            expected.append((node, frozenset(live)))
            continue
        if node.op in ("br", "br_if", "return"):
            allow_dead = False
        if node.op == "local.get":
            live.add(node.arg)
        if node.op in ("local.set", "local.tee"):
            if allow_dead and node.op == "local.set" and node.arg not in live:
                node = replace(node, op="drop", arg=None)
            else:
                live.discard(node.arg)
        expected.append((node, frozenset()))
    expected.reverse()
    i = j = 0
    while i < len(expected):
        node, region_live = expected[i]
        if node.op == "nop":
            i += 1
            continue
        if (
            node.op in ("i32.const", "i64.const", "local.get")
            and i + 1 < len(expected)
            and expected[i + 1][0].op == "drop"
        ):
            i += 2
            continue
        if j >= len(after):
            return False
        other = after[j]
        if replace(node, body=(), otherwise=()) != replace(other, body=(), otherwise=()):
            return False
        if not _check_sequence(node.body, other.body, region_live, False) or not _check_sequence(
            node.otherwise, other.otherwise, region_live, False
        ):
            return False
        i += 1
        j += 1
    return j == len(after)


def check_rewrite(evidence):
    try:
        if set(evidence) != {
            "version",
            "source",
            "result",
            "source_sha256",
            "result_sha256",
            "semantics",
            "rules",
            "checker",
        }:
            return False
        if (
            evidence["version"] != "typed-rewrite/v1"
            or evidence["checker"] != "typed-rewrite-checker/v1"
            or evidence["semantics"] != "typed_value_and_trap_equivalence"
        ):
            return False
        if evidence["rules"] != ["nop", "total_push_drop", "dead_local_set_to_drop"]:
            return False
        before, after = parse(evidence["source"]), parse(evidence["result"])
        if (
            digest_bytes(before.canonical_source.encode()) != evidence["source_sha256"]
            or digest_bytes(after.canonical_source.encode()) != evidence["result_sha256"]
        ):
            return False
        return _check_sequence(before.instructions, after.instructions)
    except (ValueError, TypeError, KeyError):
        return False
