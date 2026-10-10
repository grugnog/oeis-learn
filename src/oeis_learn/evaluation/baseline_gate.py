"""Replay G0–G6 run-bound evidence; diagnostic completion alone never qualifies."""

from pathlib import Path
import math
import xml.etree.ElementTree as ET
from oeis_learn.experiments.artifacts import compute_file_hash, compute_canonical_digest as digest
from oeis_learn.evaluation.foundation_cohort import read_ref
from oeis_learn.evaluation.experiment_runner import implementation_identity
from oeis_learn.experiments.profiles import profile_digests

SUITES = {
    "G1": {"test_foundation_reference", "test_foundation_execution", "test_foundation_conformance"},
    "G2": {
        "test_foundation_cohort",
        "test_foundation_evaluation",
        "test_foundation_finalization",
        "test_foundation_roles",
    },
    "G3": {
        "test_foundation_codec",
        "test_foundation_sampler",
        "test_foundation_admission",
        "test_foundation_pool",
    },
    "G4": {"test_foundation_resume"},
    "G5": {"test_foundation_resources", "test_foundation_roles", "test_foundation_gpu"},
    "G6": {"test_foundation_artifacts", "test_foundation_proof_boundary"},
}


def require_junit(root, reference, gate):
    from oeis_learn.experiments.artifacts import ArtifactPath

    if set(reference) != {"path", "sha256"}:
        raise ValueError("hashed owning-suite reference required")
    file = ArtifactPath(Path(root), reference["path"]).as_path()
    if compute_file_hash(file) != reference["sha256"]:
        raise ValueError("owning-suite content mismatch")
    cases = ET.parse(file).findall(".//testcase")
    cases = [
        c
        for c in cases
        if not (
            gate == "G5"
            and c.attrib.get("name") == "test_unavailable_gpu_never_falls_back"
            and c.find("skipped") is not None
        )
    ]
    if not cases or any(
        any(c.find(t) is not None for t in ("failure", "error", "skipped")) for c in cases
    ):
        raise ValueError("owning suite failed, skipped or empty")
    names = {c.attrib.get("classname", "").split(".")[-1] for c in cases}
    if not SUITES[gate] <= names:
        raise ValueError("owning-suite coverage incomplete")
    if gate in ("G2", "G5") and not any(
        c.attrib.get("name") == "test_actual_docker_role_views_and_reclamation" for c in cases
    ):
        raise ValueError("actual Docker containment execution required")
    if gate == "G5" and not any(
        c.attrib.get("name") == "test_actual_hip_three_updates" for c in cases
    ):
        raise ValueError("actual HIP strict updates required")
    return len(cases)


def validate_preflight(report):
    if (
        report.get("status") != "ready"
        or report.get("gate") != "passed"
        or report.get("visible_terms") != 20
        or report.get("body_tokens") != 1024
        or report.get("updates") != 3
        or report.get("no_cpu_fallback") is not True
        or not report.get("hip_version")
        or not str(report.get("device", "")).startswith("cuda")
        or not 0 < report.get("elapsed_total_s", float("inf")) <= 600
        or report.get("peak_memory_bytes_after", 0) <= 0
    ):
        raise ValueError("complete actual-device1024-token preflight required")
    if len(report.get("steps", [])) != 3:
        raise ValueError("three measured preflight updates required")
    for i, row in enumerate(report["steps"]):
        if (
            row["update"] != i + 1
            or row["visible_terms"] != 20
            or row["body_tokens"] != 1024
            or row["dtype"] != "float32"
            or row["loss_finite"] is not True
            or row["grad_finite"] is not True
            or row["params_changed"] <= 0
            or row["max_abs_param_change"] <= 0
            or not str(row["device"]).startswith("cuda")
            or not math.isfinite(row["loss"])
        ):
            raise ValueError("invalid actual device update")


def require_baseline(bundle, checkpoint, *, evidence_root):
    expected = dict(
        checkpoint_sha256=checkpoint.manifest["blob_sha256"],
        contract_sha256=digest(checkpoint.contract),
        run_id=checkpoint.contract["run_id"],
        code_sha256=implementation_identity(),
        profiles=profile_digests(),
        runtime_sha256=checkpoint.manifest["runtime_sha256"],
    )
    if (
        set(bundle) != {"version", "binding", "checks", "report_sha256"}
        or bundle["version"] != "baseline-gates/v1"
        or bundle["binding"] != expected
        or bundle["report_sha256"] != digest(bundle, "report_sha256")
        or set(bundle["checks"]) != {f"G{i}" for i in range(7)}
    ):
        raise ValueError("complete current run/checkpoint-bound G0–G6 evidence required")
    checks, root = bundle["checks"], Path(evidence_root)
    if set(checks["G0"]) != {"preflight", "adoption"}:
        raise ValueError("G0 references incomplete")
    preflight = read_ref(root, checks["G0"]["preflight"])
    validate_preflight(preflight)
    adoption = read_ref(root, checks["G0"]["adoption"])
    if (
        set(adoption) != {"version", "constitution_sha256", "adoption_sha256"}
        or adoption["version"] != "adoption-binding/v1"
    ):
        raise ValueError("explicit adopted constitution binding required")
    repo = Path(__file__).resolve().parents[3]
    # Installed wheels embed the same governance bytes at build time.
    constitution = repo / ".specify/memory/constitution.md"
    record = repo / "specs/007-experiment-foundation/validation/adoption.md"
    if not constitution.exists():
        constitution = Path(__file__).parent / "governance/constitution.md"
        record = Path(__file__).parent / "governance/adoption.md"
    if adoption["constitution_sha256"] != compute_file_hash(constitution) or adoption[
        "adoption_sha256"
    ] != compute_file_hash(record):
        raise ValueError("adopted governance identity mismatch")
    gates = {"G0": dict(status="PASS", preflight_sha256=checks["G0"]["preflight"]["sha256"])}
    for gate in SUITES:
        if set(checks[gate]) != (
            {"junit", "run_updates"}
            if gate == "G5"
            else {"junit", "containment"}
            if gate == "G2"
            else {"junit", "native_parity"}
            if gate == "G1"
            else {"junit"}
        ):
            raise ValueError("closed owning-suite evidence required")
        gates[gate] = dict(status="PASS", cases=require_junit(root, checks[gate]["junit"], gate))
    from oeis_learn.sandbox.native_conformance import check_native_transcript

    check_native_transcript(read_ref(root, checks["G1"]["native_parity"]))
    require_containment(read_ref(root, checks["G2"]["containment"]), checkpoint)
    # Owning HIP test cannot be rebound to a CPU checkpoint with a completed label.
    if (
        checkpoint.manifest["completed_update"] < 3
        or not checkpoint.runtime
        or checkpoint.runtime["device"] != "cuda"
    ):
        raise ValueError("current checkpoint is not a completed strict GPU run")
    updates = read_ref(root, checks["G5"]["run_updates"])
    if (
        set(updates) != {"checkpoint_sha256", "run_id", "events"}
        or updates["checkpoint_sha256"] != expected["checkpoint_sha256"]
        or updates["run_id"] != expected["run_id"]
    ):
        raise ValueError("GPU updates are not bound to this checkpoint/run")
    records = [r for r in updates["events"] if r.get("kind") == "update"]
    if len(records) != checkpoint.manifest["completed_update"] or not all(
        r["parameters_changed"] is True
        and r["parameter_device"].startswith("cuda")
        and r["loss"]["state"] == "measured"
        and math.isfinite(r["loss"]["value"])
        and r["gradient_norm"]["state"] == "measured"
        and math.isfinite(r["gradient_norm"]["value"])
        and r["resources"]["gpu_peak_allocated"]["value"] > 0
        for r in records
    ):
        raise ValueError("current strict run lacks finite measured GPU updates")
    return gates


def assemble_baseline_bundle(checkpoint_path, run_root, preflight, junit, output):
    """Bind existing executed reports and actual run events, then replay every gate."""
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
    from oeis_learn.tracking.foundation_metrics import EventLog
    from oeis_learn.evaluation.foundation_cohort import write_json
    from oeis_learn.experiments.artifacts import ArtifactPath, atomic_write

    checkpoint = load_foundation_checkpoint(checkpoint_path, device="cpu", include_payload=True)
    root = Path(output)
    checks = {gate: {} for gate in ("G0", *SUITES)}

    def copy(name, source):
        target = ArtifactPath(root, name)
        atomic_write(target, Path(source).read_bytes())
        return dict(path=name, sha256=compute_file_hash(target.as_path()))

    checks["G0"]["preflight"] = copy("preflight.json", preflight)
    repo = Path(__file__).resolve().parents[3]
    constitution = repo / ".specify/memory/constitution.md"
    record = repo / "specs/007-experiment-foundation/validation/adoption.md"
    if not constitution.exists():
        constitution = Path(__file__).parent / "governance/constitution.md"
        record = Path(__file__).parent / "governance/adoption.md"
    adoption = dict(
        version="adoption-binding/v1",
        constitution_sha256=compute_file_hash(constitution),
        adoption_sha256=compute_file_hash(record),
    )
    checks["G0"]["adoption"] = write_json(root, "adoption.json", adoption)
    for gate in SUITES:
        checks[gate]["junit"] = copy(gate + ".xml", junit[gate])
    checks["G1"]["native_parity"] = copy("native-parity.json", junit["native_parity"])
    checks["G2"]["containment"] = copy("containment.json", junit["containment"])
    checks["G5"]["run_updates"] = write_json(
        root,
        "run-updates.json",
        dict(
            checkpoint_sha256=checkpoint.manifest["blob_sha256"],
            run_id=checkpoint.contract["run_id"],
            events=list(EventLog(Path(run_root) / "events").events.values()),
        ),
    )
    bundle = dict(
        version="baseline-gates/v1",
        binding=dict(
            checkpoint_sha256=checkpoint.manifest["blob_sha256"],
            contract_sha256=digest(checkpoint.contract),
            run_id=checkpoint.contract["run_id"],
            code_sha256=implementation_identity(),
            profiles=profile_digests(),
            runtime_sha256=checkpoint.manifest["runtime_sha256"],
        ),
        checks=checks,
    )
    bundle["report_sha256"] = digest(bundle)
    require_baseline(bundle, checkpoint, evidence_root=root)
    write_json(root, "baseline.json", bundle)
    return bundle


def require_containment(report, checkpoint):
    if (
        report.get("version") != "coordinated-smoke/v1"
        or report.get("report_sha256") != digest(report, "report_sha256")
        or report.get("diagnostic_small_host") is not False
        or report.get("final_truth_available") is not False
        or report.get("main_memory_bytes") != 88 << 30
        or report.get("worker_memory_bytes") != 1 << 30
        or type(report.get("workers")) is not int
        or not 1 <= report["workers"] <= 8
        or report.get("aggregate_memory_bytes") != (88 + report["workers"]) << 30
        or report.get("host_floor_bytes") != 16 << 30
    ):
        raise ValueError("actual strict workstation containment evidence required")
    result = report["result"]
    if (
        result["status"] != "complete"
        or result["device"] != "cuda"
        or result["completed_updates"] != 3
        or result["checkpoint_sha256"] != checkpoint.manifest["blob_sha256"]
        or result["run_id"] != checkpoint.contract["run_id"]
    ):
        raise ValueError("containment is not bound to current GPU checkpoint")
    events = report["roles"]
    starts = [r for r in events if "operation" in r]
    stops = [r for r in events if "reclaimed" in r]
    samples = [r for r in events if "host" in r]
    if (
        len(starts) != 1
        or starts[0]["operation"] != "smoke"
        or starts[0]["image"] != report["image"]
        or len(stops) != 1
        or stops[0]["reclaimed"] is not True
        or not samples
        or any(
            r["host_floor"] != 16 << 30 or r["host"]["host_available"]["state"] != "measured"
            for r in samples
        )
    ):
        raise ValueError("executed role lifecycle/reserve monitoring missing")
    return True
