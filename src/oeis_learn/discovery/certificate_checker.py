"""Independent bounded fresh-process replay. No model or external knowledge APIs."""

from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.discovery.certificates import (
    VERSION,
    KINDS,
    compiler_identity,
    validate_statement,
    derive,
)


def _check(certificate):
    from oeis_learn.sandbox.lowering import lower_logical
    from oeis_learn.sandbox.wat_ast import ExecutionFailure

    try:
        if (
            set(certificate)
            != {
                "version",
                "kind",
                "statement",
                "statement_sha256",
                "template",
                "payload",
                "compiler",
                "certificate_sha256",
            }
            or certificate["version"] != VERSION
            or certificate["kind"] not in KINDS
            or certificate["certificate_sha256"] != digest(certificate, "certificate_sha256")
            or certificate["statement_sha256"] != digest(certificate["statement"])
        ):
            raise ValueError("certificate/statement identity mutation")
        sources, _, _ = validate_statement(certificate["statement"])
        compiler = certificate["compiler"]
        if compiler != {
            "implementation_sha256": compiler_identity(),
            "compiled_sha256": [digest(lower_logical(s)) for s in sources],
            "linkage": "finite_conformance_only",
            "wasm_theorem": False,
        }:
            raise ValueError("compiler binding/linkage mutation")
        if certificate["kind"] != "loop_invariant" and certificate["template"] is not None:
            raise ValueError("inactive certificate template")
        okay, payload = derive(
            certificate["kind"], certificate["statement"], certificate["template"]
        )
        if not okay or certificate["payload"] != payload:
            raise ValueError("class-specific certificate obligations/evidence mismatch")
        return {
            "result": "verified",
            "kind": certificate["kind"],
            "certificate_sha256": certificate["certificate_sha256"],
            "statement_sha256": certificate["statement_sha256"],
            "semantics": certificate["statement"]["semantics"],
            "assumptions": certificate["statement"]["assumptions"],
            "compiler_linkage": "finite_conformance_only",
            "wasm_theorem": False,
            "checker": "program-certificate-checker/v1",
        }
    except NotImplementedError as exc:
        return {"result": "unsupported", "reason": str(exc)}
    except TimeoutError as exc:
        return {"result": "unknown", "reason": str(exc)}
    except (ValueError, KeyError, TypeError, ExecutionFailure) as exc:
        return {"result": "unknown", "reason": str(exc)}


def check_certificate(certificate, *, timeout_ms=10000):
    from oeis_learn.sandbox.contained import run_contained
    from oeis_learn.experiments.artifacts import canonical_bytes

    if len(canonical_bytes(certificate)) > 512 << 10:
        raise ValueError("certificate request cap512KiB")
    status, result = run_contained(_check, (certificate,), timeout_ms=timeout_ms, proof=True)
    if status == "ok":
        return result
    return {"result": "timeout" if status == "timeout" else "unknown", "reason": result}
