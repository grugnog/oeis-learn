"""Bounded symbolic proof for pure bigint-body/v1 polynomials over Z.

The independent reference reader supplies syntax/types only. No execution,
production AST/lowering, fitting, sampling, legacy proof or CAS is an authority.
Coefficient convolution proves a program identity, not an unknown sequence law.
Optional interval arithmetic checks numeric bounds at every original subtree;
it proves neither unbounded machine safety nor fuel/memory/deadline sufficiency.
"""

from dataclasses import asdict, dataclass
import hashlib
from importlib.resources import files
import time

from .contracts import _check, digest, schema, seal


@dataclass(frozen=True)
class Limits:
    source_bytes: int = 65536
    nodes: int = 4096
    nesting: int = 32
    degree: int = 128
    coefficient_bits: int = 4096
    work: int = 131072

    def __post_init__(self):
        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= field.default:
                raise ValueError("invalid or enlarged polynomial checker limit: " + name)


def checker_identity():
    """Bind the checker, independent grammar and declared resource semantics."""
    paths = (
        ("oeis_learn.generalization.progressive", "polynomial_certificate.py"),
        ("oeis_learn.synthesis", "reference_parser.py"),
        ("oeis_learn.synthesis", "contracts.py"),
        ("oeis_learn.generalization", "schemas/progressive-polynomial-certificate.schema.json"),
    )
    return digest(
        {
            package + "/" + path: "sha256:"
            + hashlib.sha256(files(package).joinpath(path).read_bytes()).hexdigest()
            for package, path in paths
        }
    )


class _Work:
    def __init__(self, limits, budget):
        self.limits = limits
        self.used = self.nodes = 0
        self.deadline = (
            min(time.monotonic() + 2, budget.deadline) if budget else time.monotonic() + 2
        )
        if budget:
            # One bounded symbolic attempt; failures retain their original debit.
            budget.consume(attempts=1, rejections=1)
        self.spend(0)

    def spend(self, count=1):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("polynomial checker deadline exhausted")
        if count > self.limits.work - self.used:
            raise ValueError("polynomial work budget exhausted")
        self.used += count

    def coefficient(self, value):
        if abs(value).bit_length() > self.limits.coefficient_bits:
            raise ValueError("polynomial coefficient bit limit")
        return value

    def decimal(self, text, bits):
        # Shape/grammar validates canonical syntax; bound before conversion.
        if len(text) > bits * 30103 // 100000 + 2:
            raise ValueError("integer range/length limit")
        self.spend(len(text))
        value = int(text)
        if abs(value).bit_length() > bits:
            raise ValueError("integer range/bit limit")
        return value


def _trim(values):
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return values


def _normalize(node, work):
    work.spend()
    work.nodes += 1
    if work.nodes > work.limits.nodes:
        raise ValueError("polynomial node limit")
    if node == "$index":
        return [0, 1]
    op = node[0]
    if op == "int":
        return [work.decimal(node[1], work.limits.coefficient_bits)]
    if op not in {"add", "sub", "mul"}:
        raise ValueError("unsupported non-polynomial expression")
    # Never eliminate an operand before checking it (including 0 * expression).
    left, right = (_normalize(child, work) for child in node[1:])
    if op == "mul":
        degree = len(left) + len(right) - 2
        if degree > work.limits.degree:
            raise ValueError("polynomial degree limit")
        out = [0] * (degree + 1)
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                work.spend()
                product = work.coefficient(a * b)
                out[i + j] = work.coefficient(out[i + j] + product)
    else:
        out = [0] * max(len(left), len(right))
        for i in range(len(out)):
            work.spend()
            a = left[i] if i < len(left) else 0
            b = right[i] if i < len(right) else 0
            out[i] = work.coefficient(a + b if op == "add" else a - b)
    return _trim(out)


def _interval(node, domain, work, numeric):
    work.spend()
    if node == "$index":
        out = domain
    elif node[0] == "int":
        n = work.decimal(node[1], numeric["literal_bits"])
        out = (n, n)
    else:
        a, b = _interval(node[1], domain, work, numeric)
        c, d = _interval(node[2], domain, work, numeric)
        if node[0] == "add":
            out = (a + c, b + d)
        elif node[0] == "sub":
            out = (a - d, b - c)
        else:
            work.spend(4)
            products = (a * c, a * d, b * c, b * d)
            out = (min(products), max(products))
    if any(abs(n).bit_length() > numeric["intermediate_bits"] for n in out):
        raise ValueError("finite numeric range safety not established at an intermediate")
    return out


def check(source, certificate, *, limits=Limits(), budget=None):
    """Replay exact coefficients (ascending degree), failing closed on any limit.

    Only empty assumptions and one pure return are supported. ``machine_domain``
    is null or an inclusive {lower, upper} interval of canonical integer strings.
    An unsuccessful interval argument raises; a separate certificate with null
    domain can still establish the mathematical identity with safety unverified.
    """
    if type(limits) is not Limits:
        raise ValueError("registered polynomial limits required")
    work = _Work(limits, budget)
    if type(source) is not str or len(source) > limits.source_bytes:
        raise ValueError("polynomial source limit")
    if len(source.encode("utf-8")) > limits.source_bytes:
        raise ValueError("polynomial source byte limit")
    doc = schema("polynomial-certificate")
    _check(doc, certificate, doc)
    if certificate["source"] != digest(source):
        raise ValueError("polynomial certificate original source binding differs")
    coefficients = certificate["coefficients"]
    if len(coefficients) > limits.degree + 1:
        raise ValueError("polynomial degree limit")
    expected = [work.decimal(c, limits.coefficient_bits) for c in coefficients]
    if len(expected) > 1 and expected[-1] == 0:
        raise ValueError("canonical polynomial cannot have trailing zero coefficients")

    from oeis_learn.synthesis.reference_parser import parse
    from oeis_learn.synthesis.contracts import ResourceProfile

    profile = ResourceProfile(
        source_bytes=limits.source_bytes, ast_nodes=limits.nodes, nesting=limits.nesting
    )
    program = parse(source, profile)
    if program.declarations or program.statements:
        raise ValueError("unsupported polynomial statements/locals/loops")
    actual = _normalize(program.result, work)
    if actual != expected:
        raise ValueError("polynomial coefficients do not prove the bound source")
    numeric = {
        k: getattr(profile, k)
        for k in ("input_bits", "literal_bits", "intermediate_bits", "output_bytes")
    }
    domain = certificate["machine_domain"]
    if domain is not None:
        lower, upper = (work.decimal(domain[k], numeric["input_bits"]) for k in ("lower", "upper"))
        if lower > upper:
            raise ValueError("ordered nonempty finite machine range required")
        interval = _interval(program.result, (lower, upper), work, numeric)
        if any(max(1, (abs(n).bit_length() + 7) // 8) > numeric["output_bytes"] for n in interval):
            raise ValueError("finite output range safety not established")
    checker = checker_identity()
    work.spend(0)
    result = seal(
        dict(
            version="progressive-polynomial-check/v1",
            checker=checker,
            source=digest(source),
            certificate=digest(certificate),
            statement="proved",
            scope="program_equals_polynomial_only",
            arithmetic="mathematical_integers",
            domain="all_integer_indices",
            coefficients=list(coefficients),
            assumptions=[],
            machine_domain=dict(domain) if domain is not None else None,
            machine_safety="unverified" if domain is None else "finite_numeric_range_verified",
            numeric_limits=numeric,
            resource_safety="unverified",
            sequence_law="unproved",
            limits=asdict(limits),
            work=dict(nodes=work.nodes, operations=work.used),
        )
    )
    work.spend(0)
    return result
