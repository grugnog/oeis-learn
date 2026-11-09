"""Typed pure-expression components with hygienic, fully charged expansion.

Semantic validation is independent of scientific adoption. Transparent expansion
adds no runtime calls or decoder tokens; it retains checked bigint-body semantics.
"""

import copy
import re
from pathlib import Path

from .contracts import digest, record, validate

PROFILE = "bigint-pure-components/v1"
BOUNDS = dict(
    maximum_components=64,
    dependency_depth=4,
    maximum_arguments=8,
    expanded_nodes=4096,
    expanded_bytes=256 << 10,
)
TYPES = {"bigint", "bool", "control"}
SIGNATURES = {
    "add": (("bigint", "bigint"), "bigint"),
    "sub": (("bigint", "bigint"), "bigint"),
    "mul": (("bigint", "bigint"), "bigint"),
    "eq": (("bigint", "bigint"), "bool"),
    "lt": (("bigint", "bigint"), "bool"),
    "le": (("bigint", "bigint"), "bool"),
    "to_control": (("bigint",), "control"),
    "to_bigint": (("control",), "bigint"),
    "increment": (("control",), "control"),
    "control.lt": (("control", "control"), "bool"),
    "true": ((), "bool"),
    "false": ((), "bool"),
}
NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def normalize(definition):
    required = {"id", "arguments", "returns", "body", "dependencies", "origins", "assumptions"}
    if (
        not isinstance(definition, dict)
        or set(definition) != required
        or not isinstance(definition["id"], str)
        or not NAME.fullmatch(definition["id"])
    ):
        raise ValueError("closed typed component definition required")
    if (
        definition["returns"] not in TYPES
        or not isinstance(definition["arguments"], list)
        or len(definition["arguments"]) > 8
    ):
        raise ValueError("bounded typed arguments required")
    names = {}
    for i, arg in enumerate(definition["arguments"]):
        if (
            set(arg) != {"name", "type"}
            or not isinstance(arg["name"], str)
            or not NAME.fullmatch(arg["name"])
            or arg["type"] not in TYPES
            or arg["name"] in names
        ):
            raise ValueError("unique typed hole names required")
        names[arg["name"]] = f"h{i}"
    visits = 0

    def rename(node, depth=0):
        nonlocal visits
        visits += 1
        if depth > 32 or visits > 4096:
            raise ValueError("component AST bound exceeded")
        if isinstance(node, str):
            return node
        if not isinstance(node, (tuple, list)) or not node or len(node) > 10:
            raise ValueError("pure expression AST required")
        if node[0] == "arg":
            if len(node) != 2 or node[1] not in names:
                raise ValueError("unbound component hole")
            return ["arg", names[node[1]]]
        return [rename(x, depth + 1) for x in node]

    result = copy.deepcopy(definition)
    result["body"] = rename(definition["body"])
    result["arguments"] = [
        dict(name=f"h{i}", type=a["type"]) for i, a in enumerate(definition["arguments"])
    ]
    return result


def _dependencies(node):
    found = set()

    def visit(n):
        if isinstance(n, list):
            if n and n[0] == "call":
                if len(n) < 2 or not isinstance(n[1], str):
                    raise ValueError("component call identity required")
                found.add(n[1])
            for c in n:
                visit(c)

    visit(node)
    return sorted(found)


def _table(components, bounds):
    if not isinstance(components, list) or len(components) > bounds["maximum_components"]:
        raise ValueError("component count bound exceeded")
    table = {c["id"]: c for c in components}
    if len(table) != len(components):
        raise ValueError("duplicate component identity")
    for c in components:
        if c["dependencies"] != _dependencies(c["body"]):
            raise ValueError("exact component dependencies required")
        if len(c["arguments"]) > bounds["maximum_arguments"]:
            raise ValueError("argument bound exceeded")
        if (
            len({o["structure"] for o in c["origins"]}) < 3
            or len({o["group"] for o in c["origins"]}) < 2
            or any(o["role"] != "training" for o in c["origins"])
        ):
            raise ValueError("three verified training structures and two groups required")
    depths = {}

    def depth(name, stack=()):
        if name in stack:
            raise ValueError("cyclic component dependency")
        if name not in table:
            raise ValueError("unknown component dependency")
        if name not in depths:
            depths[name] = 1 + max(
                (depth(d, (*stack, name)) for d in table[name]["dependencies"]), default=0
            )
        if depths[name] > bounds["dependency_depth"]:
            raise ValueError("component dependency depth exceeded")
        return depths[name]

    for name in table:
        depth(name)
    return table


class _Expansion:
    def __init__(self, table, bounds, limits):
        self.table, self.bounds, self.limits = table, bounds, limits
        self.work = self.calls = 0
        self.cap = min(bounds["expanded_nodes"], limits.ast_nodes)

    def tick(self, n=1):
        self.work += n
        if self.work > 4 * self.cap:
            raise ValueError("expanded AST work budget exceeded")

    def clone(self, node, depth=0):
        self.tick()
        if depth > self.limits.nesting:
            raise ValueError("expanded nesting limit exceeded")
        return [self.clone(c, depth + 1) for c in node] if isinstance(node, list) else node

    def expand(self, node, env=None, stack=(), depth=0):
        env = {} if env is None else env
        self.tick()
        if depth > self.limits.nesting:
            raise ValueError("expanded nesting limit exceeded")
        if node == "$index":
            return "$index", "bigint"
        if not isinstance(node, (tuple, list)) or not node or not isinstance(node[0], str):
            raise ValueError("pure typed expression required; mutable captures are excluded")
        op, *args = node
        if op == "arg":
            if len(args) != 1 or args[0] not in env:
                raise ValueError("unbound typed hole")
            value, kind = env[args[0]]
            return self.clone(value), kind  # caller operands are never recaptured by the callee
        if op == "call":
            if not args or args[0] not in self.table or args[0] in stack:
                raise ValueError("unknown/cyclic component call")
            c = self.table[args[0]]
            if len(stack) >= self.bounds["dependency_depth"] or len(args) - 1 != len(
                c["arguments"]
            ):
                raise ValueError("component call depth/argument limit")
            actual = [self.expand(a, env, stack, depth + 1) for a in args[1:]]
            if [kind for _, kind in actual] != [a["type"] for a in c["arguments"]]:
                raise ValueError("component argument type mismatch")
            self.calls += 1
            bound = {a["name"]: v for a, v in zip(c["arguments"], actual)}
            value, kind = self.expand(c["body"], bound, (*stack, c["id"]), depth)
            if kind != c["returns"]:
                raise ValueError("component result type mismatch")
            return value, kind
        if op in {"int", "control"}:
            from oeis_learn.synthesis.contracts import canonical_integer

            if len(args) != 1 or not isinstance(args[0], str):
                raise ValueError("exact literal required")
            value = canonical_integer(args[0], bits=self.limits.literal_bits if op == "int" else 31)
            if op == "control" and not 0 <= value <= self.limits.control_max:
                raise ValueError("control literal bound")
            self.tick(2)
            return [op, args[0]], "bigint" if op == "int" else "control"
        expanded = [self.expand(a, env, stack, depth + 1) for a in args]
        types = tuple(k for _, k in expanded)
        if op == "select" and len(types) == 3 and types[0] == "bool" and types[1] == types[2]:
            kind = types[1]
        elif op in SIGNATURES and types == SIGNATURES[op][0]:
            kind = SIGNATURES[op][1]
        else:
            raise ValueError("unsupported pure expression or type mismatch: " + op)
        self.tick()
        return [op, *[n for n, _ in expanded]], kind


def _render(node):
    return node if isinstance(node, str) else "(" + " ".join(map(_render, node)) + ")"


def _count(node):
    return 1 + sum(_count(n) for n in node) if isinstance(node, list) else 1


def _source(ast, kind):
    result = (
        ast
        if kind == "bigint"
        else ["to_bigint", ast]
        if kind == "control"
        else ["select", ast, ["int", "1"], ["int", "0"]]
    )
    return "(program (return " + _render(result) + "))"


def _bounds(bounds):
    bounds = BOUNDS if bounds is None else bounds
    if set(bounds) != set(BOUNDS) or any(
        type(bounds[k]) is not int or not 1 <= bounds[k] <= BOUNDS[k] for k in BOUNDS
    ):
        raise ValueError("positive frozen component bounds required")
    return dict(bounds)


def _templates(components, bounds, limits):
    table = _table(components, bounds)
    result = []
    from oeis_learn.synthesis.reference_parser import parse

    for c in components:
        env = {a["name"]: (["arg", a["name"]], a["type"]) for a in c["arguments"]}
        worker = _Expansion(table, bounds, limits)
        ast, kind = worker.expand(c["body"], env, stack=(c["id"],))
        if kind != c["returns"]:
            raise ValueError("component declared return type mismatch")
        # Independently parse a typed representative of every fully expanded
        # template. This establishes typing, never an all-input equality proof.
        samples = {
            a["name"]: (
                ["int", "0"]
                if a["type"] == "bigint"
                else ["control", "0"]
                if a["type"] == "control"
                else ["false"],
                a["type"],
            )
            for a in c["arguments"]
        }
        representative, _ = _Expansion(table, bounds, limits).expand(
            c["body"], samples, stack=(c["id"],)
        )
        parse(_source(representative, kind), limits)
        if _count(ast) > min(bounds["expanded_nodes"], limits.ast_nodes) or len(
            _source(ast, kind).encode()
        ) > min(bounds["expanded_bytes"], limits.source_bytes):
            raise ValueError("expanded template node/byte bound exceeded")
        result.append(dict(c, expanded_ast=ast))
    return result


def make_library(
    components,
    *,
    purpose,
    track,
    previous=None,
    bounds=None,
    verification=None,
    created_at="2026-11-08T00:00:00Z",
):
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.synthesis.admission import reference_identity
    import oeis_learn.synthesis.reference_parser as parser

    bounds = _bounds(bounds)
    normalized = [normalize(c) for c in components]
    normalized = _templates(normalized, bounds, ResourceProfile())
    verification = verification or dict(
        status="DIAGNOSTIC_ONLY",
        native=None,
        reference=reference_identity(),
        parser=digest(Path(parser.__file__).read_text()),
        proofs=[],
    )
    if (
        verification["status"] == "NATIVE_VERIFIED"
        and (verification["native"] is None or (normalized and not verification["proofs"]))
    ) or (verification["status"] == "DIAGNOSTIC_ONLY" and verification["native"] is not None):
        raise ValueError("original native producer and full expansion proofs required")
    if purpose == "research" and verification["status"] != "NATIVE_VERIFIED":
        raise ValueError("native component publication required for research")
    return record(
        "library",
        dict(
            profile=PROFILE,
            components=normalized,
            bounds=bounds,
            previous=previous,
            verification=verification,
            semantics="hygienic_operand_order_expansion",
            scientific_status="UNMEASURED",
        ),
        created_at=created_at,
        parents=[previous] if previous else [],
        purpose=purpose,
        track=track,
    )


def validate_library(library, *, limits=None):
    from oeis_learn.synthesis.contracts import ResourceProfile

    value = validate("library", library)
    p = value["payload"]
    bounds = _bounds(p["bounds"])
    raw = [{k: v for k, v in c.items() if k != "expanded_ast"} for c in p["components"]]
    normalized = [normalize(c) for c in raw]
    if (
        raw != normalized
        or _templates(normalized, bounds, limits or ResourceProfile()) != p["components"]
    ):
        raise ValueError("expanded component template binding differs")
    verification = p["verification"]
    if (
        verification["status"] == "NATIVE_VERIFIED"
        and (verification["native"] is None or (p["components"] and not verification["proofs"]))
    ) or (verification["status"] == "DIAGNOSTIC_ONLY" and verification["native"] is not None):
        raise ValueError("original native component proof binding missing")
    if value["purpose"] == "research" and p["verification"]["status"] != "NATIVE_VERIFIED":
        raise ValueError("research library lacks original native verification")
    return value


def expand(library, expression, *, limits=None):
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.synthesis.reference_parser import parse

    limits = limits or ResourceProfile()
    p = validate_library(library)["payload"]
    worker = _Expansion(_table(p["components"], p["bounds"]), p["bounds"], limits)
    ast, kind = worker.expand(expression)
    source = _source(ast, kind)
    parsed = parse(source, limits)

    def json_ast(node):
        return [json_ast(c) for c in node] if isinstance(node, tuple) else node

    nodes = _count(["program", ["return", json_ast(parsed.result)]])
    if nodes > min(p["bounds"]["expanded_nodes"], limits.ast_nodes) or len(source.encode()) > min(
        p["bounds"]["expanded_bytes"], limits.source_bytes
    ):
        raise ValueError("expanded candidate node/byte bound exceeded")
    parse(source, limits)
    return dict(
        source=source,
        expanded_ast=ast,
        expanded_nodes=nodes,
        expanded_bytes=len(source.encode()),
        calls=worker.calls,
        execution_units=nodes,
        expansion_units=worker.work,
        scientific_status="UNMEASURED",
    )
