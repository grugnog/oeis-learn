"""Immutable structure/parameter/composition exclusions shared by admission paths."""

from dataclasses import dataclass

from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.sandbox.wat_ast import parse


def structural_signature(source):
    program = parse(source)
    names = {"$n": "index"}

    def nodes(items):
        result = []
        for node in items:
            arg = node.arg
            if node.op.startswith("local."):
                if arg not in names:
                    typ = "i64" if arg.startswith("$v") else "i32"
                    names[arg] = f"{typ}:{sum(v.startswith(typ) for v in names.values())}"
                arg = names[arg]
            elif node.op.endswith(".const") or node.op == "i256.zero":
                arg = "typed_literal"
            result.append(
                [
                    "i256.const" if node.op == "i256.zero" else node.op,
                    arg,
                    list(node.results),
                    node.has_else,
                    nodes(node.body),
                    nodes(node.otherwise),
                ]
            )
        return result

    return digest(["dependency-structure/v1", nodes(program.instructions)])


def split_for_structure(signature, seed=20260913):
    bucket = int(digest(["structural-split/v1", seed, signature])[7:23], 16) % 100
    return "train" if bucket < 80 else "development" if bucket < 90 else "final"


def features(source):
    program = parse(source)
    operators, constants = set(), []

    def visit(nodes):
        for node in nodes:
            operators.add(node.op)
            if node.op.endswith(".const"):
                constants.append(node.arg)
            visit(node.body)
            visit(node.otherwise)

    visit(program.instructions)
    return frozenset(operators), tuple(constants)


@dataclass(frozen=True)
class StructuralHoldouts:
    seed: int = 20260913
    reserved_structures: frozenset[str] = frozenset()
    reserved_compositions: tuple[frozenset[str], ...] = (frozenset({"i64.xor", "i64.mul"}),)
    reserved_parameter_bits: int = 128
    reserved_prefixes: frozenset[str] = frozenset()

    def __post_init__(self):
        from oeis_learn.experiments.models import check_digest

        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**64
            or type(self.reserved_parameter_bits) is not int
            or not 1 <= self.reserved_parameter_bits <= 256
        ):
            raise ValueError("bounded exact holdout settings required")
        for key in self.reserved_structures | self.reserved_prefixes:
            check_digest(key, "reserved holdout identity")
        if (
            not isinstance(self.reserved_structures, frozenset)
            or not isinstance(self.reserved_prefixes, frozenset)
            or any(not isinstance(v, frozenset) or not v for v in self.reserved_compositions)
        ):
            raise ValueError("immutable canonical holdout sets required")

    def decision(self, source, conditioning):
        if not 1 <= len(conditioning) <= 20 or any(type(v) is not int for v in conditioning):
            raise ValueError("exact visible conditioning required")
        structure = structural_signature(source)
        operators, constants = features(source)
        if (
            structure in self.reserved_structures
            or split_for_structure(structure, self.seed) != "train"
        ):
            return False, "structural_holdout"
        if any(abs(c).bit_length() >= self.reserved_parameter_bits for c in constants):
            return False, "parameter_holdout"
        if any(group <= operators for group in self.reserved_compositions):
            return False, "composition_holdout"
        if digest([str(v) for v in conditioning]) in self.reserved_prefixes:
            return False, "oeis_prefix_holdout"
        return True, "eligible"

    def identity(self):
        return digest(
            {
                "version": "expanded-holdouts/v1",
                "seed": self.seed,
                "structures": sorted(self.reserved_structures),
                "compositions": [sorted(s) for s in self.reserved_compositions],
                "parameter_bits": self.reserved_parameter_bits,
                "prefixes": sorted(self.reserved_prefixes),
            }
        )


def structural_report(sources, seed=20260913):
    groups = {}
    for source in sources:
        signature = structural_signature(source)
        groups.setdefault(signature, []).append(digest(source))
    report = {
        "version": "structural-split/v1",
        "seed": seed,
        "groups": {
            key: {"split": split_for_structure(key, seed), "sources": sorted(set(values))}
            for key, values in sorted(groups.items())
        },
    }
    report["identity_sha256"] = digest(report)
    return report
