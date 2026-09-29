#!/usr/bin/env python3
"""Executes 1,000-candidate qualification smoke evaluation."""

from __future__ import annotations

import json
import os
import sys
import time
import torch
from oeis_learn.data.benchmark import load_benchmark_manifest
from oeis_learn.evaluation.checkpoint import load_checkpoint_v2
from oeis_learn.evaluation.protocol import EvaluationProtocol
from oeis_learn.evaluation.synthesis import evaluate_cohort_synthesis


def run_smoke_test(
    checkpoint_path: str = "runs/007_phase4_production_symple/checkpoints/model_epoch_060.v2.pt",
    manifest_path: str = "data/benchmarks/trustworthy_synthesis_v1.json",
    output_path: str = "reports/smoke_1000_candidates.json",
    target_count: int = 1000,
    budget: int = 16,
) -> int:
    print(f"Starting {target_count}-candidate qualification smoke evaluation...")
    device = torch.device("cpu")
    encoder, decoder, prov = load_checkpoint_v2(checkpoint_path, device=device)
    manifest = load_benchmark_manifest(manifest_path)

    total_candidates = 0
    assembly_passes = 0
    runtime_traps = 0
    extrap_successes = 0

    t_start = time.perf_counter()

    # Repeat over targets until target_count reached
    target_idx = 0
    while total_candidates < target_count:
        target = manifest.targets[target_idx % len(manifest.targets)]
        target_idx += 1

        proto = EvaluationProtocol.from_dict({
            "schema_version": "1.0",
            "checkpoint_sha256": prov.checkpoint_sha256,
            "benchmark_manifest_sha256": manifest.manifest_sha256,
            "observed_horizon": 20,
            "unseen_horizon": 100,
            "candidate_budget": budget,
            "base_seed": 42 + total_candidates,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 128,
            "constant_resolution": True,
            "solver_timeout_ms": 250,
            "max_placeholders": 4,
            "fuel_per_invocation": 10000,
            "memory_limit_mib": 16,
            "mdl_ratio_max": 1.20,
            "native_evaluator_required": True,
            "code_revision": "smoke-test",
            "environment_fingerprint": "sha256:" + "0" * 64,
        })

        res = evaluate_cohort_synthesis(encoder, decoder, prov, target, proto, device=device)
        total_candidates += len(res.candidates)

        for c in res.candidates:
            stages = {s.stage: s.status for s in c.stage_records}
            if stages.get("ASSEMBLY") in ("PASSED", "NOT_REQUIRED"):
                assembly_passes += 1
            if stages.get("EXECUTION") == "FAILED":
                runtime_traps += 1
            if c.classification == "EXTRAPOLATING_SUCCESS":
                extrap_successes += 1

        if target_idx % 10 == 0:
            print(f"Progress: {total_candidates}/{target_count} candidates evaluated...")

    elapsed = time.perf_counter() - t_start
    assembly_validity = assembly_passes / total_candidates
    trap_rate = runtime_traps / total_candidates

    print(f"\nSmoke test complete in {elapsed:.2f}s ({elapsed/60.0:.2f}m):")
    print(f"Total candidates evaluated: {total_candidates}")
    print(f"Assembly validity: {assembly_validity * 100:.2f}% ({assembly_passes}/{total_candidates})")
    print(f"Runtime trap rate: {trap_rate * 100:.2f}% ({runtime_traps}/{total_candidates})")
    print(f"Extrapolating successes: {extrap_successes}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    smoke_summary = {
        "total_candidates": total_candidates,
        "assembly_validity_rate": assembly_validity,
        "runtime_trap_rate": trap_rate,
        "extrapolating_successes": extrap_successes,
        "elapsed_seconds": elapsed,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(smoke_summary, f, indent=2)
    print(f"Saved {output_path}")

    # Gate check: 100% assembly validity and <= 15% trap rate
    if assembly_validity < 1.0 or trap_rate > 0.15:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
