"""Sealed class-specific statements, distinct from finite output matches."""

from pathlib import Path
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.sandbox.logical_ir import parse_logical, profile_identity

VERSION = "program-certificate/v1"
KINDS = (
    "polynomial_program",
    "linear_recurrence",
    "finite_state",
    "loop_invariant",
    "bounded_machine",
)
MATH_ASSUMPTION = "mathematical integer IR; checked-machine safe range is not established"


def compiler_identity():
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            p: digest((root / p).read_text())
            for p in (
                "sandbox/logical_ir.py",
                "sandbox/lowering.py",
                "sandbox/wide_helpers.py",
                "sandbox/preamble.wat",
            )
        }
    )


def validate_statement(statement):
    from oeis_learn.discovery.pslq_solver import exact_integer

    if (
        set(statement) != {"programs", "coefficients", "domain", "semantics", "assumptions"}
        or not 1 <= len(statement["programs"]) <= 8
        or len(statement["coefficients"]) != len(statement["programs"])
    ):
        raise ValueError("closed bounded certificate statement required")
    domain = statement["domain"]
    if (
        set(domain) != {"start", "end", "excluded_indices"}
        or type(domain["start"]) is not int
        or (
            domain["end"] is not None
            and (type(domain["end"]) is not int or domain["end"] < domain["start"])
        )
        or len(domain["excluded_indices"]) > 1024
        or any(type(x) is not int for x in domain["excluded_indices"])
        or sorted(set(domain["excluded_indices"])) != domain["excluded_indices"]
    ):
        raise ValueError("nonempty exact domain required")
    if domain["end"] is not None and domain["end"] - domain["start"] + 1 <= len(
        [x for x in domain["excluded_indices"] if domain["start"] <= x <= domain["end"]]
    ):
        raise ValueError("empty domain")
    if statement["semantics"] not in ("mathematical_integer_ir", "checked_logical_ir"):
        raise ValueError("explicit theorem semantics required")
    expected = [MATH_ASSUMPTION] if statement["semantics"] == "mathematical_integer_ir" else []
    if statement["assumptions"] != expected:
        raise ValueError("unverified/changed certificate assumptions")
    for p in statement["programs"]:
        if set(p) != {"source", "ir_sha256", "profile_sha256"}:
            raise ValueError("bound program record required")
        program = parse_logical(p["source"])
        if (
            p["source"] != program.canonical_source
            or p["ir_sha256"] != program.identity
            or p["profile_sha256"] != digest(profile_identity(program.profile))
        ):
            raise ValueError("certificate source/IR/semantics mutation")
    coefficients = [exact_integer(c) for c in statement["coefficients"]]
    if not all(coefficients):
        raise ValueError("nonzero supported relation required")
    return [p["source"] for p in statement["programs"]], coefficients, domain


def derive(kind, statement, template):
    programs, coeffs, domain = validate_statement(statement)
    if (
        kind in ("polynomial_program", "linear_recurrence")
        and statement["semantics"] != "mathematical_integer_ir"
    ):
        raise ValueError("machine safety cannot follow from integer identity")
    if (
        kind in ("finite_state", "loop_invariant", "bounded_machine")
        and statement["semantics"] != "checked_logical_ir"
    ):
        raise ValueError("checked logical semantics required")
    if kind == "polynomial_program":
        from oeis_learn.discovery.polynomial_certificate import check_polynomial

        return check_polynomial(programs, coeffs, domain)
    if kind == "linear_recurrence":
        from oeis_learn.discovery.recurrence_certificate import check_recurrence

        return check_recurrence(programs, coeffs, domain)
    if kind == "finite_state":
        from oeis_learn.discovery.state_certificate import check_state

        return check_state(programs, coeffs, domain)
    if kind == "loop_invariant":
        from oeis_learn.discovery.loop_certificate import check_loop

        if len(programs) != 1 or coeffs != [1]:
            raise ValueError("one restricted loop property required")
        return check_loop(programs[0], domain, template)
    if kind == "bounded_machine":
        from oeis_learn.sandbox.streaming import LogicalRuntime, LogicalLimits
        from oeis_learn.sandbox.pipeline import decode_limbs

        if (
            domain["end"] is None
            or domain["end"] - domain["start"] >= 256
            or not -(1 << 31) <= domain["start"] <= domain["end"] < (1 << 31)
        ):
            raise ValueError("finite machine enumeration domain cap256")
        outputs = []
        fuel = steps = 0
        for source in programs:
            runtime = LogicalRuntime(source, LogicalLimits())
            if runtime.program.profile != "pure":
                raise NotImplementedError("bounded pure machine properties")
            row = []
            for n in range(domain["start"], domain["end"] + 1):
                if n in domain["excluded_indices"]:
                    continue
                runtime.fresh()
                value = decode_limbs(runtime.call("compute", n))
                expected = runtime.reference_call("compute", n)
                if value != expected:
                    raise ValueError("compiled/reference disagreement")
                row.append(str(value))
            outputs.append(row)
            fuel += runtime.fuel
            steps += runtime.steps
        return all(
            sum(c * int(outputs[j][i]) for j, c in enumerate(coeffs)) == 0
            for i in range(len(outputs[0]))
        ), {
            "outputs": outputs,
            "indices": [
                n
                for n in range(domain["start"], domain["end"] + 1)
                if n not in domain["excluded_indices"]
            ],
            "scope": "bounded_machine_only",
            "fuel": fuel,
            "reference_steps": steps,
        }
    raise NotImplementedError("unsupported certificate family")


def make_certificate(kind, sources, coefficients, *, domain=None, template=None):
    from oeis_learn.sandbox.contained import run_contained
    from oeis_learn.experiments.artifacts import canonical_bytes

    if len(canonical_bytes([sources, coefficients, domain, template])) > 512 << 10:
        raise ValueError("certificate request cap512KiB")
    status, result = run_contained(
        _make_certificate,
        (kind, sources, coefficients, domain, template),
        timeout_ms=10000,
        proof=True,
    )
    if status != "ok":
        raise ValueError("certificate obligations unproved: " + str(result))
    return result


def _make_certificate(kind, sources, coefficients, domain, template):
    from oeis_learn.sandbox.lowering import lower_logical
    from oeis_learn.discovery.compiler_conformance import finite_compiled_conformance

    if kind not in KINDS:
        raise ValueError("unknown certificate family")
    programs = [parse_logical(s) for s in sources]
    mathematical = kind in ("polynomial_program", "linear_recurrence")
    statement = {
        "programs": [
            {
                "source": p.canonical_source,
                "ir_sha256": p.identity,
                "profile_sha256": digest(profile_identity(p.profile)),
            }
            for p in programs
        ],
        "coefficients": [str(c) for c in coefficients],
        "domain": domain
        or {
            "start": 0,
            "end": (1 << 31) - 2 if kind == "finite_state" else None,
            "excluded_indices": [],
        },
        "semantics": "mathematical_integer_ir" if mathematical else "checked_logical_ir",
        "assumptions": [MATH_ASSUMPTION] if mathematical else [],
    }
    okay, payload = derive(kind, statement, template)
    if not okay:
        raise ValueError("certificate obligations unproved")
    certificate = {
        "version": VERSION,
        "kind": kind,
        "statement": statement,
        "statement_sha256": digest(statement),
        "template": template,
        "payload": payload,
        "compiler": {
            "implementation_sha256": compiler_identity(),
            "compiled_sha256": [digest(lower_logical(p)) for p in programs],
            "linkage": "finite_conformance_only",
            "wasm_theorem": False,
            "finite_conformance": finite_compiled_conformance(
                [p.canonical_source for p in programs], statement["domain"]
            ),
        },
    }
    certificate["certificate_sha256"] = digest(certificate)
    return certificate
