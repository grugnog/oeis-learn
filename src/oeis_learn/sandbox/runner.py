"""WASM Sandbox execution runner invoking native Rust PyO3 extension or fallback."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union
from oeis_learn.data.models import CanonicalProgramArtifact, ExecutionResult
from oeis_learn.sandbox.optimizer import optimize_wat_program

logger = logging.getLogger(__name__)

try:
    import oeis_wasm_evaluator

    HAS_NATIVE_EVALUATOR = True
except ImportError:
    HAS_NATIVE_EVALUATOR = False
    logger.warning("Native oeis_wasm_evaluator not available, using pure Python wasmtime fallback.")


def decode_i256_limbs(limbs: Sequence[int]) -> int:
    """Reconstructs signed two's complement 256-bit integer from four 64-bit little-endian limbs."""
    if len(limbs) < 4:
        return 0
    l0, l1, l2, l3 = limbs[0], limbs[1], limbs[2], limbs[3]
    u0 = l0 if l0 >= 0 else l0 + (1 << 64)
    u1 = l1 if l1 >= 0 else l1 + (1 << 64)
    u2 = l2 if l2 >= 0 else l2 + (1 << 64)
    return u0 | (u1 << 64) | (u2 << 128) | (l3 << 192)


class WasmRunner:
    """Execution manager for running WAT/WASM programs in resource-bounded sandboxes."""

    def __init__(
        self,
        fuel_budget: int = 10000,
        memory_limit_mib: int = 16,
        terms_to_generate: int = 20,
        use_fallback: bool = False,
    ):
        self.fuel_budget = fuel_budget
        self.memory_limit_mib = memory_limit_mib
        self.terms_to_generate = terms_to_generate
        self.use_fallback = use_fallback or not HAS_NATIVE_EVALUATOR

    def run_single(
        self,
        wat_code: str,
        fuel_budget: Optional[int] = None,
        terms_to_generate: Optional[int] = None,
        result_profile: str = "i64_scalar_v1",
    ) -> ExecutionResult:
        """Evaluates a single WAT program string."""
        fuel = fuel_budget if fuel_budget is not None else self.fuel_budget
        terms = terms_to_generate if terms_to_generate is not None else self.terms_to_generate

        if not self.use_fallback and HAS_NATIVE_EVALUATOR:
            res = oeis_wasm_evaluator.evaluate_wat_single(wat_code, fuel, terms)
            wide_out = getattr(res, "wide_output", [])
            output = res.output
            if result_profile == "i256x4_v1" and wide_out:
                output = [decode_i256_limbs(limbs) for limbs in wide_out]

            return ExecutionResult(
                status=res.status,
                consumed_fuel=getattr(res, "max_fuel", res.consumed_fuel),
                output=output,
                error=res.error,
                max_fuel=getattr(res, "max_fuel", res.consumed_fuel),
                total_fuel=getattr(res, "total_fuel", res.consumed_fuel),
                wide_output=wide_out,
            )
        else:
            from oeis_learn.sandbox.fallback_runner import evaluate_wat_single_fallback
            return evaluate_wat_single_fallback(wat_code, fuel, terms)

    def run_batch(
        self,
        wat_programs: Sequence[str],
        fuel_budget: Optional[int] = None,
        terms_to_generate: Optional[int] = None,
        result_profile: str = "i64_scalar_v1",
    ) -> List[ExecutionResult]:
        """Evaluates a batch of WAT programs concurrently across CPU threads."""
        fuel = fuel_budget if fuel_budget is not None else self.fuel_budget
        terms = terms_to_generate if terms_to_generate is not None else self.terms_to_generate

        if not self.use_fallback and HAS_NATIVE_EVALUATOR:
            results = oeis_wasm_evaluator.evaluate_wat_batch(list(wat_programs), fuel, terms)
            out = []
            for r in results:
                wide_out = getattr(r, "wide_output", [])
                output = r.output
                if result_profile == "i256x4_v1" and wide_out:
                    output = [decode_i256_limbs(limbs) for limbs in wide_out]
                out.append(
                    ExecutionResult(
                        status=r.status,
                        consumed_fuel=getattr(r, "max_fuel", r.consumed_fuel),
                        output=output,
                        error=r.error,
                        max_fuel=getattr(r, "max_fuel", r.consumed_fuel),
                        total_fuel=getattr(r, "total_fuel", r.consumed_fuel),
                        wide_output=wide_out,
                    )
                )
            return out
        else:
            from oeis_learn.sandbox.fallback_runner import evaluate_wat_batch_fallback
            return evaluate_wat_batch_fallback(list(wat_programs), fuel, terms)

    def run_optimized_single(
        self,
        wat_code: str,
        fuel_budget: Optional[int] = None,
        terms_to_generate: Optional[int] = None,
        hard_waste_threshold: float = 0.30,
    ) -> Tuple[ExecutionResult, CanonicalProgramArtifact]:
        """Runs the dead-code elimination & vacuuming pass, then executes the optimized program."""
        artifact = optimize_wat_program(wat_code, hard_waste_threshold=hard_waste_threshold)
        exec_res = self.run_single(
            artifact.opt_wat,
            fuel_budget=fuel_budget,
            terms_to_generate=terms_to_generate,
        )
        return exec_res, artifact

    def run_optimized_batch(
        self,
        wat_programs: Sequence[str],
        fuel_budget: Optional[int] = None,
        terms_to_generate: Optional[int] = None,
        hard_waste_threshold: float = 0.30,
    ) -> List[Tuple[ExecutionResult, CanonicalProgramArtifact]]:
        """Optimizes a batch of WAT programs and executes the optimized modules concurrently."""
        artifacts = [
            optimize_wat_program(wat, hard_waste_threshold=hard_waste_threshold)
            for wat in wat_programs
        ]
        opt_programs = [art.opt_wat for art in artifacts]
        results = self.run_batch(
            opt_programs,
            fuel_budget=fuel_budget,
            terms_to_generate=terms_to_generate,
        )
        return list(zip(results, artifacts))
