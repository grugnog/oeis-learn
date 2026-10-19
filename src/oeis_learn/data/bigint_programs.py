"""Counter-based connected typed generic A–D families. No target metadata inputs."""

from dataclasses import asdict
import random
from oeis_learn.synthesis.contracts import identity, ResourceProfile, canonical_integer
from oeis_learn.synthesis.reference_parser import parse
from oeis_learn.synthesis.reference import evaluate

PRIORS = dict(
    version="bigint-connected/v1",
    variants=16,
    stages=("A", "B", "C", "D"),
    parameter_range=(-16, 16),
    depth=(1, 3),
    loop="actual_nonnegative_index",
    simplifier="none",
)


def render(node):
    return node if isinstance(node, str) else "(" + " ".join(render(n) for n in node) + ")"


def canonical_family(source):
    p = parse(source, allow_parameters=True)
    names = {}
    counts = {"bigint": 0, "control": 0, "bool": 0}
    prefix = {"bigint": "b", "control": "c", "bool": "q"}
    for name, kind in p.declarations:
        names[name] = f"${prefix[kind]}{counts[kind]}"
        counts[kind] += 1
    params = {}

    def visit(node):
        if isinstance(node, str):
            return names.get(node, node)
        if node[0] == "param.get":
            params.setdefault(node[1], f"p{len(params)}")
            return ("param.get", params[node[1]])
        return tuple(visit(n) for n in node)

    tree = (
        "program",
        *(("local", names[n], k) for n, k in p.declarations),
        *(visit(s) for s in p.statements),
        ("return", visit(p.result)),
    )
    return identity({"version": "typed-family/v1", "source": render(tree)})


class BigintSampler:
    def __init__(self, seed=20261016):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("invalid seed")
        self.seed = seed
        self.identity = identity({"priors": PRIORS, "seed": seed})

    def sample(self, counter):
        if type(counter) is not int or not 0 <= counter < 1_000_000:
            raise ValueError("counter limit")
        family, variant = divmod(counter, 16)
        stage = PRIORS["stages"][family % 4]
        rng = random.Random(int(identity([self.identity, family])[7:23], 16))
        coefficients = random.Random(int(identity([self.identity, family, variant])[7:23], 16))
        assignments = {}

        def parameter():
            if len(assignments) >= 8:
                return ("param.get", rng.choice(tuple(assignments)))
            name = f"p{len(assignments)}"
            assignments[name] = str(coefficients.randint(-16, 16))
            return ("param.get", name)

        def expression(depth, leaf):
            if depth == 0:
                return leaf
            op = rng.choice(("add", "sub", "mul"))
            return (op, expression(depth - 1, leaf), parameter() if rng.randrange(2) else "$index")

        declarations = []
        statements = []
        if stage == "A":
            result = expression(rng.randint(1, 3), "$index")
        elif stage == "B":
            result = (
                rng.choice(("add", "sub", "mul")),
                expression(2, "$index"),
                expression(2, parameter()),
            )
        elif stage == "C":
            declarations = [("local", "$b0", "bigint"), ("local", "$c0", "control")]
            carried = (
                rng.choice(("add", "mul")),
                ("get", "$b0"),
                ("add", ("to_bigint", ("increment", ("get", "$c0"))), parameter()),
            )
            statements = [
                ("set", "$b0", parameter()),
                ("loop", "$c0", ("to_control", "$index"), ("set", "$b0", carried)),
            ]
            result = ("add", ("get", "$b0"), expression(1, "$index"))
        else:
            declarations = [("local", "$b0", "bigint")]
            guard = (rng.choice(("lt", "le", "eq")), "$index", ("int", str(rng.randint(1, 60))))
            statements = [
                (
                    "if",
                    guard,
                    ("then", ("set", "$b0", expression(2, "$index"))),
                    ("else", ("set", "$b0", expression(3, "$index"))),
                )
            ]
            result = ("get", "$b0")
        tree = ("program", *declarations, *statements, ("return", result))
        # Renumber in actual source order, including shared occurrences.
        renaming = {}

        def rename(n):
            if isinstance(n, str):
                return n
            if n[0] == "param.get":
                renaming.setdefault(n[1], f"p{len(renaming)}")
                return ("param.get", renaming[n[1]])
            return tuple(rename(x) for x in n)

        tree = rename(tree)
        assignments = {renaming[k]: v for k, v in assignments.items()}

        def concrete(n):
            if isinstance(n, str):
                return n
            if n[0] == "param.get":
                return ("int", assignments[n[1]])
            return tuple(concrete(x) for x in n)

        source = render(concrete(tree))
        parameter_source = render(tree)
        parse(source)
        parse(parameter_source, allow_parameters=True)
        return dict(
            id=identity([self.identity, counter]),
            generator=self.identity,
            stage=stage,
            family=canonical_family(parameter_source),
            variant=variant,
            assignment=assignments,
            source=source,
            parameter_source=parameter_source,
            origin="generic_random",
        )


def materialize(row, limits=ResourceProfile()):
    """Independent exact 100-index oracle; production cross-check is qualification."""
    indices = tuple(map(str, range(100)))
    result = evaluate(row["source"], indices, limits)
    if result.outcome != "ok":
        raise ValueError(f"excluded:{result.reason}")
    # Inputs to the learning representation have their own stricter bound.
    if any(
        abs(canonical_integer(v, bits=limits.intermediate_bits)).bit_length() > limits.input_bits
        for v in result.values[:20]
    ):
        raise ValueError("excluded:input_integer_limit")
    return dict(
        row,
        indices=list(indices),
        values=list(result.values),
        oracle="independent-source-reference/v1",
        resource=identity(asdict(limits)),
    )
