"""Frozen benchmark manifest loader, horizon validation, and leakage detection."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
from oeis_learn.data.models import BenchmarkCohort, BenchmarkTarget
from oeis_learn.evaluation.protocol import canonical_json_dumps, canonical_json_hash


def compute_term_fingerprint(terms: Sequence[Any]) -> str:
    """Computes a deterministic hash over sequence terms to detect exact/shift leakage."""
    norm_terms = [str(int(t)) for t in terms]
    raw = ",".join(norm_terms).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def check_leakage_fingerprints(
    candidate_terms: Sequence[Any],
    candidate_program_hashes: Sequence[str],
    cohort: BenchmarkCohort,
) -> Tuple[bool, Optional[str]]:
    """Checks whether candidate terms or program hashes collide with frozen evaluation targets."""
    cand_fp = compute_term_fingerprint(candidate_terms)
    cand_prog_set = set(candidate_program_hashes)

    for target in cohort.targets:
        if cand_fp == target.term_fingerprint:
            return True, f"Term list collision with evaluation target {target.oeis_id}"
        target_progs = set(target.program_fingerprints)
        overlap = cand_prog_set.intersection(target_progs)
        if overlap:
            return True, f"Program fingerprint match ({next(iter(overlap))}) with target {target.oeis_id}"

    return False, None


def load_benchmark_manifest(
    manifest_path: str,
    expected_sha256: Optional[str] = None,
) -> BenchmarkCohort:
    """Loads a frozen benchmark manifest from JSON with integrity verification."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if expected_sha256 is not None:
        with open(manifest_path, "rb") as f:
            file_sha = f"sha256:{hashlib.sha256(f.read()).hexdigest()}"
        if file_sha != expected_sha256:
            raise ValueError(
                f"Benchmark manifest digest mismatch: expected {expected_sha256}, got {file_sha}"
            )

    schema_ver = data.get("schema_version")
    if schema_ver != "1.0":
        raise ValueError(f"Unsupported benchmark manifest schema_version: {schema_ver}")

    obs_h = int(data.get("observed_horizon", 0))
    uns_h = int(data.get("unseen_horizon", 0))
    if obs_h != 20:
        raise ValueError(f"observed_horizon must be 20, got {obs_h}")
    if uns_h != 100:
        raise ValueError(f"unseen_horizon must be 100, got {uns_h}")

    targets: List[BenchmarkTarget] = []
    for t_data in data.get("targets", []):
        obs = t_data.get("observed_terms", [])
        uns = t_data.get("unseen_terms", [])
        if len(obs) != 20:
            raise ValueError(
                f"Target {t_data.get('oeis_id')} must have exactly 20 observed terms, got {len(obs)}"
            )
        if len(uns) != 100:
            raise ValueError(
                f"Target {t_data.get('oeis_id')} must have exactly 100 unseen terms, got {len(uns)}"
            )

        target = BenchmarkTarget(
            oeis_id=t_data["oeis_id"],
            name=t_data["name"],
            offset=int(t_data["offset"]),
            family=t_data["family"],
            curriculum_stage=int(t_data["curriculum_stage"]),
            observed_terms=[str(x) for x in obs],
            unseen_terms=[str(x) for x in uns],
            result_profile=t_data["result_profile"],
            terms_sha256=t_data["terms_sha256"],
            term_fingerprint=t_data["term_fingerprint"],
            formula_definition_id=t_data.get("formula_definition_id"),
            program_fingerprints=t_data.get("program_fingerprints", []),
            tags=t_data.get("tags", []),
        )
        targets.append(target)

    cohort = BenchmarkCohort(
        schema_version=schema_ver,
        cohort_id=data["cohort_id"],
        manifest_sha256=data.get("manifest_sha256", ""),
        source=data["source"],
        observed_horizon=obs_h,
        unseen_horizon=uns_h,
        targets=targets,
        exclusions=data.get("exclusions", []),
    )
    return cohort
