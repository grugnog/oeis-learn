"""WASM Sandbox execution runner invoking native Rust PyO3 extension or fallback."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple
from oeis_learn.data.models import CanonicalProgramArtifact, ExecutionResult
from oeis_learn.sandbox.optimizer import optimize_wat_program

logger = logging.getLogger(__name__)

try:
    import oeis_wasm_evaluator

    HAS_NATIVE_EVALUATOR = True
except ImportError:
    HAS_NATIVE_EVALUATOR = False
    logger.debug("Legacy native evaluator unavailable; strict backend selection is explicit.")


def decode_i256_limbs(limbs: Sequence[int]) -> int:
    """Reconstructs signed two's complement 256-bit integer from four 64-bit little-endian limbs."""
    from oeis_learn.sandbox.pipeline import decode_limbs

    return decode_limbs(limbs)


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

        if result_profile == "wat_i256_checked_v1":
            raise ValueError("Strict body execution requires FoundationRunner and explicit backend")

        # Lower macros if present or in multi-limb profile
        from oeis_learn.sandbox.lowering import lower_macro_wat

        if "i256." in wat_code or result_profile == "i256x4_v1":
            wat_code = lower_macro_wat(wat_code)

        if not self.use_fallback and HAS_NATIVE_EVALUATOR:
            res = oeis_wasm_evaluator.evaluate_wat_single(wat_code, fuel, terms)
            wide_out = getattr(res, "wide_output", [])
            output = res.output
            if (result_profile == "i256x4_v1" or wide_out) and wide_out:
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

        if result_profile == "wat_i256_checked_v1":
            raise ValueError("Strict body execution requires FoundationRunner and explicit backend")

        from oeis_learn.sandbox.lowering import lower_macro_wat

        lowered_programs = []
        for p in wat_programs:
            if "i256." in p or result_profile == "i256x4_v1":
                lowered_programs.append(lower_macro_wat(p))
            else:
                lowered_programs.append(p)

        if not self.use_fallback and HAS_NATIVE_EVALUATOR:
            results = oeis_wasm_evaluator.evaluate_wat_batch(list(lowered_programs), fuel, terms)
            out = []
            for r in results:
                wide_out = getattr(r, "wide_output", [])
                output = r.output
                if (result_profile == "i256x4_v1" or wide_out) and wide_out:
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

            return evaluate_wat_batch_fallback(lowered_programs, fuel, terms)

    def run_optimized_single(
        self,
        wat_code: str,
        fuel_budget: Optional[int] = None,
        terms_to_generate: Optional[int] = None,
        hard_waste_threshold: float = 0.30,
        result_profile: str = "i64_scalar_v1",
    ) -> Tuple[ExecutionResult, CanonicalProgramArtifact]:
        """Runs the dead-code elimination & vacuuming pass, then executes the optimized program."""
        artifact = optimize_wat_program(wat_code, hard_waste_threshold=hard_waste_threshold)
        exec_res = self.run_single(
            artifact.opt_wat,
            fuel_budget=fuel_budget,
            terms_to_generate=terms_to_generate,
            result_profile=result_profile,
        )
        return exec_res, artifact

    def run_optimized_batch(
        self,
        wat_programs: Sequence[str],
        fuel_budget: Optional[int] = None,
        terms_to_generate: Optional[int] = None,
        hard_waste_threshold: float = 0.30,
        result_profile: str = "i64_scalar_v1",
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
            result_profile=result_profile,
        )
        return list(zip(results, artifacts))


class FoundationRunner:
    """Explicit strict adapter; single and batch calls share supervised execution.

    Input is a frozen body, never a legacy module. Rust/automatic selection is
    unavailable until US5 conformance. Results are ExecutionEvidence; callers
    use pipeline.verify with evaluator-owned truth for finite CandidateResults.
    """

    def __init__(self, result_dir, *, backend, workers=1, cache_bytes=1 << 30):
        if backend != "python_wasmtime":
            raise ValueError(
                "Only explicit python_wasmtime is qualified; Rust and auto are unavailable"
            )
        from oeis_learn.sandbox.worker_pool import WorkerPool

        self.pool = WorkerPool(result_dir, workers=workers, cache_bytes=cache_bytes)

    def run_single(self, source, indices, *, request_id, **kwargs):
        return self.pool.evaluate(source, indices, request_id=request_id, **kwargs)

    def run_batch(self, sources, indices, *, request_ids, **kwargs):
        sources, request_ids, indices = list(sources), list(request_ids), list(indices)
        if len(sources) != len(request_ids) or len(set(request_ids)) != len(request_ids):
            raise ValueError("One unique attempt identity required per candidate")
        # Bounded chunks preserve input order without exceeding the queue cap.
        results = []
        for start in range(0, len(sources), 32):
            futures = [
                self.pool.submit(source, indices, request_id=identity, **kwargs)
                for source, identity in zip(
                    sources[start : start + 32], request_ids[start : start + 32]
                )
            ]
            results.extend(future.result() for future in futures)
        return results

    def close(self):
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
