"""G7 qualification: owning tests, portable replay and mandatory native parity.

The qualification harness is diagnostic and can run before activation. Ordinary
activation requires complete evidence bound to the current implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from oeis_learn.experiments.artifacts import compute_canonical_digest, canonical_bytes, load_json
from oeis_learn.sandbox.pipeline import digest_bytes, runtime_identity

ROOT = Path(__file__).resolve().parents[3]
SUITES = {
    "grounding": [
        "tests/unit/test_foundation_grounding.py",
        "tests/unit/test_constant_solver.py",
        "tests/unit/test_smt_constant_solver.py",
        "tests/unit/test_dixon_diophantine_solver.py",
        "tests/unit/test_mersenne61_filter.py",
        "tests/contract/test_constant_solver_contract.py",
        "tests/contract/test_symbolic_grounding_contract.py",
        "tests/integration/test_symbolic_grounding_pipeline.py",
    ],
    "symbolic": ["tests/unit/test_foundation_symbolic.py", "tests/unit/test_symbolic_prover.py"],
    "optimizer": ["tests/integration/test_foundation_native_optimizer.py"],
    "qualified_contracts": ["tests/contract/test_foundation_qualified.py"],
    "recurrence_dataflow": ["tests/integration/test_recurrence_grounding.py"],
    "containment_activation": ["tests/integration/test_foundation_repair_gate.py"],
    "native_entrypoints": [
        "tests/integration/test_native_phase7.py",
        "tests/contract/test_ffi_contract.py",
    ],
}
REQUIRED_CHECKS = (*SUITES, "native_parity")


def implementation_identity():
    paths = list((ROOT / "src/oeis_learn").rglob("*.py"))
    paths += list((ROOT / "src/oeis_learn").rglob("*.wat"))
    paths += list((ROOT / "src/oeis_learn").rglob("*.json"))
    paths += list((ROOT / "crates/oeis_wasm_evaluator/src").rglob("*.rs"))
    paths += [
        ROOT / p
        for p in (
            "crates/oeis_wasm_evaluator/build.rs",
            "crates/oeis_wasm_evaluator/Cargo.toml",
            "Dockerfile",
            "uv.lock",
            "pyproject.toml",
            "docker/dev/install.sh",
            "docker/dev/build-native.sh",
        )
    ]
    paths += [ROOT / p for suite in SUITES.values() for p in suite]
    return compute_canonical_digest(
        {str(p.relative_to(ROOT)): digest_bytes(p.read_bytes()) for p in sorted(set(paths))}
    )


def _junit(path, suite):
    tree = ET.parse(path)
    cases = tree.findall(".//testcase")
    if not cases or any(
        c.find("failure") is not None
        or c.find("error") is not None
        or c.find("skipped") is not None
        for c in cases
    ):
        raise ValueError("G7 owning suite failed, skipped or empty")
    expected = {Path(p).stem for p in SUITES[suite]}
    found = {c.attrib.get("classname", "").split(".")[-1] for c in cases}
    if not expected <= found:
        raise ValueError("incomplete G7 owning-module coverage")
    return len(cases)


def _artifact(root, entry):
    if set(entry) != {"path", "sha256"} or not isinstance(entry["path"], str):
        raise ValueError("invalid evidence reference")
    root = Path(root).resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("evidence path escape/missing")
    if digest_bytes(path.read_bytes()) != entry["sha256"]:
        raise ValueError("evidence content mismatch")
    return path


def require_g7(report, *, evidence_root=None):
    if not isinstance(report, dict) or set(report) != {
        "gate",
        "version",
        "status",
        "report_sha256",
        "implementation_sha256",
        "runtime",
        "checks",
        "replay_records",
    }:
        raise ValueError("Concrete complete G7 evidence is required")
    if report["gate"] != "G7" or report["version"] != "g7/v1" or report["status"] != "PASS":
        raise ValueError("G7 remains partial")
    if report["report_sha256"] != compute_canonical_digest(report, "report_sha256"):
        raise ValueError("G7 report identity mismatch")
    if report["implementation_sha256"] != implementation_identity():
        raise ValueError("G7 implementation mismatch")
    if report["runtime"] != {"python": runtime_identity(), "native": runtime_identity("rust")}:
        raise ValueError("G7 runtime mismatch")
    if evidence_root is None or set(report["checks"]) != set(REQUIRED_CHECKS):
        raise ValueError("complete G7 evidence directory required")
    for suite in SUITES:
        _junit(_artifact(evidence_root, report["checks"][suite]), suite)
    from oeis_learn.sandbox.native_conformance import check_native_transcript

    check_native_transcript(
        load_json(_artifact(evidence_root, report["checks"]["native_parity"]).read_bytes())
    )
    from oeis_learn.experiments.qualified_models import replay_record

    records = report["replay_records"]
    if len(records) != 3 or {r["kind"] for r in records} != {"grounding_result", "proof_evidence"}:
        raise ValueError("G7 requires solution, scoped UNSAT and formula replay")
    if {r.get("outcome") for r in records if r["kind"] == "grounding_result"} != {
        "verified_solution",
        "proved_unsat_in_scope",
    }:
        raise ValueError("incomplete grounding replay")
    if any(replay_record(r)["status"] != "verified" for r in records):
        raise ValueError("G7 portable replay failed")
    return report["report_sha256"]


def run_qualification(output):
    """Run fixed software acceptance. Missing native execution is a hard failure."""
    from oeis_learn.sandbox.native_conformance import run_native_conformance
    from oeis_learn.decoder.grounding import ground
    from oeis_learn.discovery.symbolic_prover import SymbolicProver
    from oeis_learn.discovery.formula_checker import qualified_proof

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "g7.json").exists():
        raise ValueError("qualification evidence is immutable; use a new directory")
    # Import and source verification before spending conformance time.
    native = runtime_identity("rust")
    report = {
        "gate": "G7",
        "version": "g7/v1",
        "status": "PARTIAL",
        "implementation_sha256": implementation_identity(),
        "runtime": {"python": runtime_identity(), "native": native},
        "checks": {},
        "replay_records": [],
    }

    def reference(path):
        return {"path": str(path.relative_to(output)), "sha256": digest_bytes(path.read_bytes())}

    try:
        for name, paths in SUITES.items():
            path = output / (name + ".xml")
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *paths, "--junitxml=" + str(path.resolve())],
                cwd=ROOT,
                timeout=600,
            )
            if result.returncode != 0:
                raise ValueError(f"{name}: owning regression failed")
            _junit(path, name)
            report["checks"][name] = reference(path)
        run_native_conformance(output / "native")
        report["checks"]["native_parity"] = reference(output / "native/parity.json")
        report["replay_records"] = [
            ground("i256.const_?", [7] * 20).to_qualified(),
            ground("i256.const 1", [2] * 20).to_qualified(),
            qualified_proof(SymbolicProver()._run(["n", "n"], [1, -1])[1]),
        ]
        report["status"] = "PASS"
        report["report_sha256"] = compute_canonical_digest(report, "report_sha256")
        require_g7(report, evidence_root=output)
        _exercise_activation(report, output)
    except BaseException:
        report["status"] = "PARTIAL"
        report["report_sha256"] = compute_canonical_digest(report, "report_sha256")
        (output / "g7.json").write_bytes(canonical_bytes(report))
        raise
    (output / "g7.json").write_bytes(canonical_bytes(report))
    return report


def _exercise_activation(report, output):
    """Exercise ordinary dispatch using the actual complete qualification report."""
    from copy import deepcopy
    from oeis_learn.experiments.config import activate_qualified_contract, load_config
    from oeis_learn.experiments.qualified_models import MODES, seal_record
    from oeis_learn.discovery.formula_checker import check_scoped_evidence
    from oeis_learn.sandbox.rewrite_checker import check_rewrite
    from oeis_learn.data.models import LatentDiscoveryCandidate

    base = load_config(ROOT / "configs/foundation/wat_smoke.yaml").to_dict()
    base_sha = compute_canonical_digest(base)
    refs = {base_sha: base}
    settings = {
        "solver": {"coefficient_bound": 1000, "timeout_ms": 2000, "max_parameters": 8},
        "optimizer": {"waste_threshold_percent": 30},
        "native": {"workers": 1, "cache_bytes": 128 << 20},
        "formula_identity": {"timeout_ms": 2000},
    }
    modes = {name: {"enabled": False, "profile_sha256": None} for name in MODES}
    for name, value in settings.items():
        profile = {"version": "qualified-service/v1", "service": name, "settings": value}
        sha = compute_canonical_digest(profile)
        refs[sha] = profile
        modes[name] = {"enabled": True, "profile_sha256": sha}
    contract = seal_record(
        "qualified_contract",
        base_contract_sha256=base_sha,
        profiles={
            name: base_sha
            for name in (
                "language",
                "runtime",
                "codec",
                "model",
                "objective",
                "sampling",
                "search",
                "admission",
            )
        },
        gates={"G7": report["report_sha256"], "G8": None, "G9": None},
        track="strict_generic",
        initialization="random",
        parent_chain=[],
        objective="sft",
        modes=modes,
    )
    services = activate_qualified_contract(
        contract, g7_report=report, evidence_root=output, resolve=refs.__getitem__
    )
    solution = services.ground("i256.const_?", [7] * 20)
    if solution.outcome != "verified_solution":
        raise ValueError("activated grounding failed")
    rewrite = services.optimize("nop i256.const 7")
    if not rewrite.rewrite_evidence or not check_rewrite(rewrite.rewrite_evidence):
        raise ValueError("activated optimizer failed")
    candidate = LatentDiscoveryCandidate("qualification", "linear", ("x", "y"), 0.0, [1, -1])
    proof = services.prover().prove_relation(candidate, ["n", "n"])
    evidence = load_json(proof.symbolic_proof)
    if proof.status != "FORMULA_IDENTITY" or not check_scoped_evidence(evidence):
        raise ValueError("activated scoped prover failed")
    with services.runner(output / "activation-native") as runner:
        execution = runner.run_single("i256.const 7", [0, 1], request_id="activation")
    if (
        execution.outcome is not None
        or execution.outputs != ["7", "7"]
        or execution.reference_outputs != ["7", "7"]
    ):
        raise ValueError("activated native runner failed")
    # A self-consistent status edit cannot authorize ordinary capabilities.
    partial = deepcopy(report)
    partial["status"] = "PARTIAL"
    partial["report_sha256"] = compute_canonical_digest(partial, "report_sha256")
    try:
        activate_qualified_contract(
            contract, g7_report=partial, evidence_root=output, resolve=refs.__getitem__
        )
    except ValueError:
        pass
    else:
        raise ValueError("partial qualification activated services")
    activation = {
        "version": "g7-activation/v1",
        "contract": contract,
        "references": refs,
        "capabilities": list(services.capabilities),
        "grounding": solution.to_qualified(),
        "rewrite": rewrite.rewrite_evidence,
        "formula": evidence,
        "native": execution.to_dict(),
        "partial_report_rejected": True,
    }
    activation["identity_sha256"] = compute_canonical_digest(activation, "identity_sha256")
    (output / "activation.json").write_bytes(canonical_bytes(activation))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_qualification(args.output)


if __name__ == "__main__":
    main()
