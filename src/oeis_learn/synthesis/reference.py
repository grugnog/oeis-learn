"""Independent bounded Python-int execution; never searches/fits parameters."""

from dataclasses import dataclass
import time

from .contracts import ResourceProfile, canonical_integer, decimal_integer, ContractError
from .reference_parser import parse, SourceError


@dataclass(frozen=True)
class ReferenceResult:
    outcome: str
    reason: str | None
    values: tuple[str, ...]
    steps: int
    coverage: int


class _Stop(Exception):
    pass


def evaluate(source, indices, limits=ResourceProfile(), *, deadline=None):
    start = time.monotonic()
    deadline = min(
        start + limits.evaluation_seconds, deadline if deadline is not None else float("inf")
    )
    total_steps, values = 0, []
    try:
        if time.monotonic() >= deadline:
            raise _Stop("deadline")
        program = parse(source, limits)
        if not 1 <= len(indices) <= 100:
            raise _Stop("invalid_source")
        for text in indices:
            try:
                index = canonical_integer(text, bits=limits.input_bits)
            except ContractError as exc:
                raise _Stop("integer_limit") from exc
            env = {name: False if kind == "bool" else 0 for name, kind in program.declarations}
            term_steps, loops = 0, 0
            term_deadline = min(deadline, time.monotonic() + limits.term_seconds)

            def tick():
                nonlocal total_steps, term_steps
                total_steps += 1
                term_steps += 1
                if (
                    term_steps > limits.reference_steps
                    or total_steps > limits.aggregate_reference_steps
                ):
                    raise _Stop("reference_limit")
                if time.monotonic() >= term_deadline:
                    raise _Stop("deadline")

            def bounded(n):
                if abs(n).bit_length() > limits.intermediate_bits:
                    raise _Stop("integer_limit")
                return n

            def expr(node):
                tick()
                if node == "$index":
                    return index
                op, args = node[0], node[1:]
                if op == "int":
                    return canonical_integer(args[0], bits=limits.literal_bits)
                if op == "control":
                    return int(args[0])
                if op == "get":
                    return env[args[0]]
                xs = [expr(a) for a in args]  # select is explicitly eager
                if op == "add":
                    return bounded(xs[0] + xs[1])
                if op == "sub":
                    return bounded(xs[0] - xs[1])
                if op == "mul":
                    return bounded(xs[0] * xs[1])
                if op == "eq":
                    return xs[0] == xs[1]
                if op in ("lt", "control.lt"):
                    return xs[0] < xs[1]
                if op == "le":
                    return xs[0] <= xs[1]
                if op == "true":
                    return True
                if op == "false":
                    return False
                if op == "to_bigint":
                    return bounded(xs[0])
                if op == "select":
                    return xs[1] if xs[0] else xs[2]
                if op in ("to_control", "increment"):
                    n = xs[0] + (op == "increment")
                    if not 0 <= n <= limits.control_max:
                        raise _Stop("control_range")
                    return n
                raise _Stop("invalid_source")

            def statement(node):
                nonlocal loops
                tick()
                op, args = node[0], node[1:]
                if op == "set":
                    env[args[0]] = expr(args[1])
                elif op == "if":
                    branch = args[1] if expr(args[0]) else args[2]
                    for child in branch[1:]:
                        statement(child)
                elif op == "loop":
                    count = expr(args[1])
                    if count > limits.loop_iterations:
                        raise _Stop("loop_limit")
                    env[args[0]] = 0
                    for i in range(count):
                        tick()
                        loops += 1
                        if loops > limits.aggregate_loop_iterations:
                            raise _Stop("loop_limit")
                        env[args[0]] = i
                        for child in args[2:]:
                            statement(child)
                    env[args[0]] = count
                else:
                    raise _Stop("invalid_source")

            for stmt in program.statements:
                statement(stmt)
            value = expr(program.result)
            tick()  # final bounded serialization
            if max(1, (abs(value).bit_length() + 7) // 8) > limits.output_bytes:
                raise _Stop("output_limit")
            values.append(decimal_integer(value))
        return ReferenceResult("ok", None, tuple(values), total_steps, len(values))
    except (SourceError, ContractError):
        return ReferenceResult("failure", "invalid_source", (), total_steps, len(values))
    except _Stop as exc:
        return ReferenceResult("failure", str(exc), (), total_steps, len(values))


def compare_final(
    source, indices, production_values, expected_values, limits=ResourceProfile(), *, deadline=None
):
    """Independent visible comparison; failure/partial evidence never agrees.

    This is an arithmetic check, not public admission or a fitting operation.
    The later admission policy must also validate all source/profile identities.
    """
    if not (len(indices) == len(production_values) == len(expected_values)):
        return "verification_failure", None
    try:
        expected = [canonical_integer(n, bits=limits.input_bits) for n in expected_values]
        production = [
            canonical_integer(n, bits=limits.intermediate_bits) for n in production_values
        ]
    except ContractError:
        return "verification_failure", None
    result = evaluate(source, indices, limits, deadline=deadline)
    if result.outcome != "ok":
        return "resource_exhausted" if result.reason not in (
            "invalid_source",
        ) else "verification_failure", result
    reference = [canonical_integer(n, bits=limits.intermediate_bits) for n in result.values]
    if reference != expected or production != expected:
        return "verification_failure", result
    return "independent_agreement", result


def evaluate_retained(source, indices, limits=ResourceProfile(), *, deadline=None):
    """Independent checked add/sub semantics with every intermediate in [-2^255,2^255)."""
    from .reference_parser import parse_retained

    started = time.monotonic()
    steps = 0
    deadline = min(
        started + limits.evaluation_seconds, float("inf") if deadline is None else deadline
    )
    try:
        p = parse_retained(source, limits)
        if not 1 <= len(indices) <= 100:
            raise _Stop("invalid_source")
        values = []
        for index in indices:
            n = canonical_integer(index, bits=32)
            if not -(1 << 31) <= n < (1 << 31):
                raise _Stop("control_range")
            term_steps = 0
            term_deadline = min(deadline, time.monotonic() + limits.term_seconds)

            def expr(node):
                nonlocal steps, term_steps
                steps += 1
                term_steps += 1
                if steps > limits.aggregate_reference_steps or term_steps > limits.reference_steps:
                    raise _Stop("reference_limit")
                if time.monotonic() >= term_deadline:
                    raise _Stop("deadline")
                if node[0] == "int":
                    value = canonical_integer(node[1], bits=256)
                else:
                    a, b = expr(node[1]), expr(node[2])
                    value = a + b if node[0] == "add" else a - b
                if not -(1 << 255) <= value < (1 << 255):
                    raise _Stop("numeric_limit")
                return value

            values.append(decimal_integer(expr(p.result)))
        return ReferenceResult("ok", None, tuple(values), steps, len(values))
    except (ValueError, _Stop) as exc:
        return ReferenceResult("failure", str(exc), (), steps, 0)
