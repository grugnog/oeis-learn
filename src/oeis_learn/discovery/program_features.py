"""Typed structural evidence with source/semantics gated, bounded indexing."""

from collections import Counter
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.sandbox.logical_ir import Node, parse_logical, profile_identity


def program_features(source):
    program = parse_logical(source)
    names = {}
    subtrees = Counter()
    edges = []
    operators = Counter()

    def slot(v):
        if v not in names:
            names[v] = len(names)
        return names[v]

    def walk(n):
        if not isinstance(n, Node):
            return n
        operators[n.op] += 1
        args = []
        for i, a in enumerate(n.args):
            if n.op in ("get", "set") and i == 0:
                args.append(("wide-slot", slot(a)))
            elif n.op == "const":
                args.append(("signed256-parameter",))
            else:
                args.append(walk(a))
        value = (n.op, tuple(args))
        subtrees[digest(value)] += 1
        if n.op == "set":
            edges.append((slot(n.args[0]), digest(args[1])))
        return value

    structural = walk(program.root)
    return {
        "version": "program-features/v1",
        "source_sha256": digest(program.canonical_source),
        "ir_sha256": program.identity,
        "semantics_sha256": digest(
            [
                profile_identity(program.profile),
                __import__(
                    "oeis_learn.discovery.certificates", fromlist=["compiler_identity"]
                ).compiler_identity(),
            ]
        ),
        "structure_sha256": digest(structural),
        "subtrees": dict(sorted(subtrees.items())),
        "operators": dict(sorted(operators.items())),
        "dataflow": edges,
        "recurrence_candidate": program.profile == "stream",
        "equivalence_status": "not_claimed",
    }


class ProgramFeatureIndex:
    def __init__(self, archive, *, snapshot=None, max_entries=4096, max_bytes=16 << 20):
        from oeis_learn.experiments.artifacts import canonical_bytes

        if not 1 <= max_entries <= 4096 or not 0 < max_bytes <= 16 << 20:
            raise ValueError("bounded program feature index required")
        snapshot = archive.freeze() if snapshot is None else snapshot
        if len(snapshot["entries"]) > 4096:
            raise ValueError("discovery archive snapshot cap4096")
        self.rows = []
        size = 0
        # Every read goes through archive's verified training view, including
        # holdouts and origin checks; output fingerprints never merge sources.
        for key, envelope in archive.training_view(snapshot):
            source = envelope["source"]
            try:
                features = program_features(source)
            except ValueError:
                features = legacy_features(source)
            row = {
                "key": key,
                "source": source,
                "features": features,
                "genealogy": envelope["provenance"],
            }
            size += len(canonical_bytes(row))
            if len(self.rows) >= max_entries or size > max_bytes:
                break
            self.rows.append(row)
        self.identity = digest(self.rows)

    def validate(self):
        for row in self.rows:
            try:
                current = program_features(row["source"])
            except ValueError:
                current = legacy_features(row["source"])
            if row["features"] != current:
                raise ValueError("stale program semantics/source features")
        if self.identity != digest(self.rows):
            raise ValueError("feature index mutation")

    def ranked_pairs(self, *, max_pairs=32, max_comparisons=4096, seconds=2):
        """Globally rank considered typed-subtree pairs; incomplete work is UNKNOWN."""
        import heapq
        import time

        if not 1 <= max_pairs <= 4096 or not 1 <= max_comparisons <= 4096 or not 0 < seconds <= 2:
            raise ValueError("bounded structural retrieval required")
        self.validate()
        rows = sorted(self.rows, key=lambda row: row["key"])
        deadline = time.monotonic() + seconds
        retained, comparisons = [], 0
        incomplete = False
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                if comparisons >= max_comparisons or time.monotonic() >= deadline:
                    incomplete = True
                    break
                comparisons += 1
                af, bf = a["features"], b["features"]
                if af["semantics_sha256"] != bf["semantics_sha256"]:
                    continue
                keys = af["subtrees"].keys() | bf["subtrees"].keys()
                overlap = sum(min(af["subtrees"].get(k, 0), bf["subtrees"].get(k, 0)) for k in keys)
                total = sum(max(af["subtrees"].get(k, 0), bf["subtrees"].get(k, 0)) for k in keys)
                if overlap:
                    retained.append((1 - overlap / total, (a["key"], b["key"])))
                    retained = heapq.nsmallest(max_pairs, retained)
            if incomplete:
                break
        return tuple(sorted(retained)), {
            "comparisons": comparisons,
            "status": "UNKNOWN" if incomplete else "COMPLETE",
            "features": "typed_subtree_proposals_only",
        }


def legacy_features(source):
    from oeis_learn.sandbox.wat_ast import parse
    from oeis_learn.experiments.profiles import profile_digests

    program = parse(source)
    names = {}
    operators = Counter()
    subtrees = Counter()

    def walk(node):
        operators[node.op] += 1
        arg = node.arg
        if node.op.startswith("local.") and arg != "$n":
            if arg not in names:
                names[arg] = len(names)
            arg = ("local", names[arg])
        elif node.op.endswith("const"):
            arg = ("parameter",)
        value = (
            node.op,
            arg,
            tuple(walk(n) for n in node.body),
            tuple(walk(n) for n in node.otherwise),
            node.results,
        )
        subtrees[digest(value)] += 1
        return value

    value = tuple(walk(n) for n in program.instructions)
    return {
        "version": "native-body-features/v1",
        "source_sha256": digest(program.canonical_source),
        "semantics_sha256": digest(
            [
                profile_digests(),
                __import__(
                    "oeis_learn.sandbox.pipeline", fromlist=["runtime_identity"]
                ).runtime_identity(),
            ]
        ),
        "structure_sha256": digest(value),
        "operators": dict(sorted(operators.items())),
        "subtrees": dict(sorted(subtrees.items())),
        "equivalence_status": "not_claimed",
    }
