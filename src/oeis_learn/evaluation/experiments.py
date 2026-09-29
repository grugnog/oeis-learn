"""Paired experiment harness enforcing fair candidate reuse, budget equality, and complete outcomes."""

from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from oeis_learn.curriculum.mdl_verifier import MdlVerifier
from oeis_learn.data.benchmark import BenchmarkCohort, BenchmarkTarget, load_benchmark_manifest
from oeis_learn.data.models import (
    ExperimentManifest,
    ExperimentOutcome,
    ExperimentVariant,
)
from oeis_learn.decoder.constant_solver import resolve_program_constants
from oeis_learn.evaluation.checkpoint import CheckpointProvenance, load_checkpoint_v2
from oeis_learn.evaluation.protocol import (
    EvaluationProtocol,
    canonical_json_dumps,
    canonical_json_hash,
)
from oeis_learn.sandbox.runner import WasmRunner


def evaluate_inference_ablation_pair(
    raw_candidates: Sequence[str],
    target_terms: Sequence[int],
    constant_resolution: bool = True,
    max_placeholders: int = 4,
    solver_timeout_ms: int = 250,
    runner: Optional[WasmRunner] = None,
) -> Dict[str, Any]:
    """Evaluates a common pool of candidate programs with or without constant resolution."""
    wasm_runner = runner or WasmRunner(fuel_budget=10000)
    candidates_out = []
    obs = list(target_terms[:20])

    for raw_wat in raw_candidates:
        resolved_wat = raw_wat
        resolved_constants = []
        if constant_resolution and "i64.const_?" in raw_wat:
            r_wat, r_consts, r_st, _, _ = resolve_program_constants(
                wat_code=raw_wat,
                terms=obs,
                timeout_ms=solver_timeout_ms,
                max_placeholders=max_placeholders,
                runner=wasm_runner,
            )
            if r_st == "PASSED":
                resolved_wat = r_wat
                resolved_constants = r_consts

        res = wasm_runner.run_single(resolved_wat, terms_to_generate=len(target_terms))
        extrap_passed = (res.status == "SUCCESS") and (res.output == list(target_terms))

        candidates_out.append({
            "raw_wat": raw_wat,
            "resolved_wat": resolved_wat,
            "constants": resolved_constants,
            "status": res.status,
            "extrap_passed": extrap_passed,
        })

    return {
        "constant_resolution": constant_resolution,
        "candidates": candidates_out,
        "pass_count": sum(1 for c in candidates_out if c["extrap_passed"]),
    }


def verify_experiment_fairness(manifest: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Verifies that all required seed/variant pairs are present and completed."""
    seeds = manifest.get("seeds", [])
    variants = manifest.get("variants", [])
    outcomes = manifest.get("outcomes", [])

    if len(seeds) < 3:
        return False, f"Manifest must declare at least 3 seeds, got {len(seeds)}"
    if len(variants) < 2:
        return False, f"Manifest must declare at least 2 variants, got {len(variants)}"

    completed_pairs = set()
    for o in outcomes:
        if o.get("status") == "COMPLETE":
            completed_pairs.add((o.get("variant_id"), o.get("seed")))

    for v in variants:
        v_id = v.get("variant_id")
        for s in seeds:
            if (v_id, s) not in completed_pairs:
                return False, f"Missing outcome for variant '{v_id}' with seed {s}"

    return True, None


def load_experiment_manifest(manifest_path: str) -> Dict[str, Any]:
    """Loads and validates an experiment manifest JSON."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
