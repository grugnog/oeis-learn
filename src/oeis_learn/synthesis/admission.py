"""Independent final admission. No search, constant proposal, or solver callbacks."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import time

from .contracts import ResourceProfile, canonical_integer, identity
from .reference import compare_final


def source_identity(source):
    return "sha256:" + sha256(source.encode()).hexdigest()


def reference_identity():
    root = Path(__file__).parent
    return identity(
        {
            name: source_identity((root / name).read_text())
            for name in ("reference.py", "reference_parser.py", "contracts.py", "admission.py")
        }
    )


@dataclass(frozen=True)
class VisiblePrompt:
    indices: tuple[str, ...]
    values: tuple[str, ...]

    def __post_init__(self):
        if type(self.indices) is not tuple or type(self.values) is not tuple:
            raise ValueError("prompt must be immutable tuples")
        if not 1 <= len(self.indices) <= 20 or len(self.indices) != len(self.values):
            raise ValueError("visible prompt count mismatch")
        ns = [canonical_integer(n) for n in self.indices]
        for n in self.values:
            canonical_integer(n)
        if any(a >= b for a, b in zip(ns, ns[1:])):
            raise ValueError("actual indices must strictly increase; gaps stay explicit")

    @property
    def sha256(self):
        return identity({"indices": self.indices, "values": self.values})


@dataclass(frozen=True)
class Admission:
    outcome: str
    reason: str | None = None
    verification: dict | None = None


def admit(result, prompt, *, runtime_sha256, limits=ResourceProfile(), deadline=None):
    """Caller supplies the independently trusted expected runtime artifact identity.

    An untrusted native status/hash alone never grants independent agreement.
    Phase 3 binds request/generation/contract identity before calling this policy.
    """
    deadline = min(time.monotonic() + 2, float("inf") if deadline is None else deadline)
    required = {
        "outcome",
        "source",
        "source_sha256",
        "original_source_sha256",
        "transformed_source",
        "occurrences",
        "assignment",
        "indices",
        "values",
        "prompt_sha256",
        "resource_sha256",
        "runtime_sha256",
        "reference_sha256",
        "language_profile",
        "coverage",
        "coefficient_bits",
        "uniqueness",
    }
    if not isinstance(result, dict) or set(result) != required:
        return Admission("verification_failure", "malformed provisional record")
    source = result["source"]
    if not isinstance(source, str) or len(source.encode()) > limits.source_bytes:
        return Admission("verification_failure", "invalid final source")
    if (
        result["outcome"] != "assignment_found"
        or result["source_sha256"] != source_identity(source)
        or result["prompt_sha256"] != prompt.sha256
        or result["resource_sha256"] != limits.identity
        or result["runtime_sha256"] != runtime_sha256
        or result["reference_sha256"] != reference_identity()
        or result["language_profile"] not in {"bigint-body/v1", "wat_i256_checked_v1"}
        or result["indices"] != list(prompt.indices)
        or type(result["coverage"]) is not int
        or result["coverage"] != len(prompt.indices)
        or result["uniqueness"] not in {"unique_in_scope", "multiple", "unchecked", "incomplete"}
        or type(result["coefficient_bits"]) is not int
        or not 1 <= result["coefficient_bits"] <= 4096
    ):
        return Admission("verification_failure", "identity, scope, or coverage mismatch")
    try:
        assignment = result["assignment"]
        if (
            not isinstance(assignment, dict)
            or set(assignment) != {f"p{i}" for i in range(len(assignment))}
            or len(assignment) > 8
        ):
            raise ValueError("invalid assignment")
        for n in assignment.values():
            canonical_integer(n)
        # Independently bind final source to the transformed source and assignment.
        from .reference_parser import parse

        transformed = parse(result["transformed_source"], limits, allow_parameters=True)

        def substitute(node):
            if isinstance(node, str):
                return node
            if node[0] == "param.get":
                return ("int", assignment[node[1]])
            return tuple(substitute(x) for x in node)

        if result["language_profile"] == "wat_i256_checked_v1":
            from .reference_parser import parse_retained

            final = parse_retained(source, limits)
        else:
            final = parse(source, limits)
        if (
            final.declarations != transformed.declarations
            or final.statements != tuple(substitute(n) for n in transformed.statements)
            or final.result != substitute(transformed.result)
        ):
            raise ValueError("final substitution mismatch")
        if result["language_profile"] == "wat_i256_checked_v1":
            from .reference import evaluate_retained

            reference = evaluate_retained(source, prompt.indices, limits, deadline=deadline)
            status = (
                "resource_exhausted"
                if reference.outcome != "ok"
                else "independent_agreement"
                if list(reference.values) == list(prompt.values) == result["values"]
                else "verification_failure"
            )
        else:
            status, reference = compare_final(
                source, prompt.indices, result["values"], prompt.values, limits, deadline=deadline
            )
    except (ValueError, KeyError, TypeError):
        return Admission("verification_failure", "malformed assignment or final substitution")
    if status != "independent_agreement":
        return Admission(status, reference.reason if reference else "visible disagreement")
    return Admission(
        "verified_solution",
        verification={
            "source_sha256": result["source_sha256"],
            "prompt_sha256": prompt.sha256,
            "runtime_sha256": runtime_sha256,
            "reference_sha256": reference_identity(),
            "resource_sha256": limits.identity,
            "indices": list(prompt.indices),
            "values": list(reference.values),
            "reference_steps": reference.steps,
            "scope": "visible_only",
            "admitted": True,
        },
    )
