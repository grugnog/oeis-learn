"""Phase 7 capability gate: incomplete evidence never enables qualified services."""
from pathlib import Path
from oeis_learn.experiments.artifacts import compute_canonical_digest

REQUIRED_CHECKS = (
    "grounding", "symbolic", "optimizer", "native_parity",
    "qualified_contracts", "recurrence_dataflow",
)

def implementation_identity():
    from oeis_learn.sandbox.pipeline import digest_bytes
    root = Path(__file__).resolve().parents[1]
    return compute_canonical_digest({
        str(path.relative_to(root)): digest_bytes(path.read_bytes())
        for path in sorted(root.rglob("*.py"))
    })

def require_g7(report):
    if not isinstance(report, dict) or report.get("gate") != "G7":
        raise ValueError("Concrete G7 evidence is required")
    if report.get("status") != "PASS":
        raise ValueError("G7 remains partial; qualified services are disabled")
    if report.get("report_sha256") != compute_canonical_digest(report, "report_sha256"):
        raise ValueError("G7 report identity mismatch")
    if report.get("implementation_sha256") != implementation_identity():
        raise ValueError("G7 evidence does not match current implementation")
    # Do not accept self-declared PASS flags as replayable conformance evidence.
    raise ValueError(
        "G7 activation remains unavailable pending native conformance, "
        "qualified evidence replay and complete activation integration"
    )
