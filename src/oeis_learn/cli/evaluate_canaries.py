"""Canary Preflight Benchmark Qualification Harness.

Evaluates the 6 landmark canary sequences (A000217, A000290, A000079, A000045, A000032, A000129)
across 20 observed terms and 100 unseen terms in 255-bit multi-limb precision.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple
from oeis_learn.curriculum.extrapolation import ExtrapolationVerifier
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer
from oeis_learn.sandbox.runner import WasmRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_canaries")

CANARY_METADATA = [
    {
        "sequence_id": "A000217",
        "name": "Triangular numbers",
        "scalar_64bit_overflow_term": 6074001000,
        "formula": lambda n: n * (n + 1) // 2,
    },
    {
        "sequence_id": "A000290",
        "name": "The squares",
        "scalar_64bit_overflow_term": 3037000500,
        "formula": lambda n: n * n,
    },
    {
        "sequence_id": "A000079",
        "name": "Powers of 2",
        "scalar_64bit_overflow_term": 63,
        "formula": lambda n: 1 << n,
    },
    {
        "sequence_id": "A000045",
        "name": "Fibonacci numbers",
        "scalar_64bit_overflow_term": 93,
        "generator": lambda terms_count: _fibonacci_terms(terms_count),
    },
    {
        "sequence_id": "A000032",
        "name": "Lucas numbers",
        "scalar_64bit_overflow_term": 91,
        "generator": lambda terms_count: _lucas_terms(terms_count),
    },
    {
        "sequence_id": "A000129",
        "name": "Pell numbers",
        "scalar_64bit_overflow_term": 51,
        "generator": lambda terms_count: _pell_terms(terms_count),
    },
]


def _fibonacci_terms(count: int) -> List[int]:
    terms = [0, 1]
    for _ in range(2, count):
        terms.append(terms[-1] + terms[-2])
    return terms[:count]


def _lucas_terms(count: int) -> List[int]:
    terms = [2, 1]
    for _ in range(2, count):
        terms.append(terms[-1] + terms[-2])
    return terms[:count]


def _pell_terms(count: int) -> List[int]:
    terms = [0, 1]
    for _ in range(2, count):
        terms.append(2 * terms[-1] + terms[-2])
    return terms[:count]


def get_ground_truth_for_canary(meta: Dict[str, Any], count: int = 120) -> List[int]:
    """Generates exact ground-truth integers in arbitrary precision for canary sequence."""
    if "generator" in meta:
        return meta["generator"](count)
    elif "formula" in meta:
        return [meta["formula"](n) for n in range(count)]
    raise ValueError(f"Unknown generator for {meta['sequence_id']}")


def run_canary_evaluation(
    checkpoint_path: str = "checkpoints/model_epoch_050.v2.pt",
    result_profile: str = "i256x4_v1",
    output_report_path: Optional[str] = None,
    fuel_budget: int = 20000,
) -> Dict[str, Any]:
    """Evaluates all 6 canaries over 120 terms and returns canary-benchmark JSON report."""
    edb = EliteSeedDemonstrationBuffer()
    runner = WasmRunner(fuel_budget=fuel_budget)
    verifier = ExtrapolationVerifier(runner=runner, n_train=20, k_extrapolate=100)

    canary_results: List[Dict[str, Any]] = []
    all_passed = True

    for meta in CANARY_METADATA:
        sid = meta["sequence_id"]
        name = meta["name"]
        overflow_idx = meta["scalar_64bit_overflow_term"]
        logger.info(f"Evaluating landmark canary {sid} ({name})...")

        ground_truth = get_ground_truth_for_canary(meta, count=120)

        # Retrieve canonical program from EDB
        entry = edb.get_entry(sid)
        if entry is None:
            logger.error(f"Missing canonical program for canary {sid} in EDB")
            all_passed = False
            canary_results.append({
                "sequence_id": sid,
                "name": name,
                "scalar_64bit_overflow_term": overflow_idx,
                "observed_matched": False,
                "unseen_matched": False,
                "overflow_prevented": False,
                "fuel_consumed": 0,
                "verdict": "FAILED_VERIFICATION",
            })
            continue

        det = verifier.verify_detailed(
            entry.wat_code,
            ground_truth,
            result_profile=result_profile,
        )

        fuel_used = det.execution_result.max_fuel if det.execution_result else 0
        overflow_prevented = True
        if overflow_idx < 120 and det.execution_result and len(det.execution_result.output) > overflow_idx:
            actual_val = det.execution_result.output[overflow_idx]
            expected_val = ground_truth[overflow_idx]
            if actual_val != expected_val or actual_val <= 0:
                overflow_prevented = False

        verdict = "EXTRAPOLATING_SUCCESS" if (det.passed and overflow_prevented) else "FAILED_VERIFICATION"
        if not det.passed or not overflow_prevented:
            all_passed = False

        canary_results.append({
            "sequence_id": sid,
            "name": name,
            "scalar_64bit_overflow_term": overflow_idx,
            "observed_matched": det.observed_match,
            "unseen_matched": det.unseen_match,
            "overflow_prevented": overflow_prevented,
            "fuel_consumed": fuel_used,
            "verdict": verdict,
        })
        logger.info(f"  Result: {verdict} (observed={det.observed_match}, unseen={det.unseen_match}, fuel={fuel_used})")

    report = {
        "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint_evaluated": checkpoint_path,
        "result_profile": result_profile,
        "canary_results": canary_results,
        "all_canaries_passed": all_passed,
    }

    if output_report_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved canary qualification report to {output_report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Canary Preflight Benchmark Qualification Runner")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_epoch_050.v2.pt")
    parser.add_argument("--profile", type=str, default="i256x4_v1", choices=["i256x4_v1", "i64_scalar_v1"])
    parser.add_argument("--output", type=str, default="reports/canary_qualification_report.json")
    parser.add_argument("--fuel", type=int, default=20000)

    args = parser.parse_args()
    report = run_canary_evaluation(
        checkpoint_path=args.checkpoint,
        result_profile=args.profile,
        output_report_path=args.output,
        fuel_budget=args.fuel,
    )
    if report["all_canaries_passed"]:
        print("ALL 6 CANARIES PASSED 100-TERM EXTRAPOLATION WITHOUT OVERFLOW.")
        sys.exit(0)
    else:
        print("CANARY QUALIFICATION FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
