"""Scoped evidence envelope/cache identity policies; solving stays in Rust."""
from .contracts import identity


def cache_key(*, source, indices, values, language, resources, solver, runtime, reference, policy):
    return identity(dict(source=source, indices=indices, values=values, language=language,
                         resources=resources, solver=solver, runtime=runtime, reference=reference, policy=policy))


def reusable(result, *, requested_bound, independent_evidence_validated=False):
    if result.get("outcome") == "verified_solution":
        return independent_evidence_validated is True
    if result.get("outcome") == "unsat_in_scope":
        return (result.get("replay_status") == "replayed_unsat"
                and type(result.get("coefficient_bits")) is int
                and result["coefficient_bits"] >= requested_bound)
    return False  # UNKNOWN/timeouts never become permanent negative cache hits.


def classify_scope(result, *, maximum_bound=4096):
    bounds = result.get("completed_bounds", [])
    if any(type(x) is not int or x not in (16,64,256,1024,4096) for x in bounds):
        raise ValueError("invalid completed scopes")
    if result.get("outcome") == "unsat_in_scope" and maximum_bound not in bounds:
        return "incomplete_scope"
    return result.get("outcome") if result.get("outcome") in {
        "assignment_found", "unsat_in_scope", "unknown", "resource_exhausted", "unsupported",
        "verification_failure", "invalid_source"} else "verification_failure"
