"""Frozen counter-keyed typed AST law, independent of held-out outcomes."""

import random
import re

from oeis_learn.data.bigint_programs import canonical_family, render

from .contracts import digest
from .metrics import _seed


def default_policy(*, seed):
    return {
        "version": "progressive-typed-ast/v1",
        "seed": _seed(seed),
        "depth": 3,
        "operators": ["add", "sub", "mul", "select", "loop"],
        "max_loop": 4,
        "max_draws": 65536,
        "max_work_units": 131072,
        "shard_rows": 4096,
        "shard_bytes": 256 << 20,
        "forbidden_structures": [],
        "forbidden_behaviors": [],
        "forbidden_prefixes": [],
    }


def validate_policy(p):
    if not isinstance(p, dict) or set(p) != set(default_policy(seed=0)):
        raise ValueError("closed frozen generator law required")
    if p["version"] != "progressive-typed-ast/v1" or p["operators"] != [
        "add",
        "sub",
        "mul",
        "select",
        "loop",
    ]:
        raise ValueError("registered typed operator law required")
    _seed(p["seed"])
    for k, cap in [
        ("depth", 5),
        ("max_loop", 8),
        ("max_draws", 65536),
        ("max_work_units", 131072),
        ("shard_rows", 4096),
        ("shard_bytes", 256 << 20),
    ]:
        if type(p[k]) is not int or not 1 <= p[k] <= cap:
            raise ValueError("finite frozen generation bound required")
    for k in ("forbidden_structures", "forbidden_behaviors", "forbidden_prefixes"):
        if (
            not isinstance(p[k], list)
            or len(set(p[k])) != len(p[k])
            or any(
                not isinstance(h, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", h) is None
                for h in p[k]
            )
        ):
            raise ValueError("frozen exclusion identities required")
    return p


def structure_id(source):
    # Literal magnitudes never count as new structural support. Keep all AST
    # operators, operand order, control flow and normalized local identities.
    return canonical_family(re.sub(r"\(int -?(?:0|[1-9][0-9]*)\)", "(int 0)", source))


def draw(policy, ordinal):
    p = validate_policy(policy)
    if type(ordinal) is not int or not 0 <= ordinal < p["max_draws"]:
        raise ValueError("counter range exhausted")
    law = {k: p[k] for k in ("version", "seed", "depth", "operators", "max_loop")}
    rng = random.Random(int(digest([law, ordinal])[7:], 16))

    def expr(depth):
        if depth == 0 or rng.random() < 0.25:
            return "$index" if rng.random() < 0.6 else ("int", str(rng.randint(-8, 8)))
        op = rng.choice(("add", "sub", "mul", "select"))
        if op == "select":
            return (
                "select",
                ("lt", "$index", ("int", str(rng.randint(1, 99)))),
                expr(depth - 1),
                expr(depth - 1),
            )
        return (op, expr(depth - 1), expr(depth - 1))

    body = expr(rng.randint(1, p["depth"]))
    if rng.random() < 0.4:
        tree = (
            "program",
            ("local", "$b0", "bigint"),
            ("local", "$c0", "control"),
            ("set", "$b0", ("int", str(rng.randint(-8, 8)))),
            (
                "loop",
                "$c0",
                ("to_control", ("int", str(rng.randint(1, p["max_loop"])))),
                ("set", "$b0", ("add", ("get", "$b0"), body)),
            ),
            ("return", ("get", "$b0")),
        )
    else:
        tree = ("program", ("return", body))
    source = render(tree)
    return {
        "ordinal": ordinal,
        "source": source,
        "structure": structure_id(source),
        "source_policy": digest(p),
        "draw_identity": digest([digest(p), ordinal]),
        "detector_limit": "structural normalization and finite 100-term signatures do not prove semantic independence",
    }
