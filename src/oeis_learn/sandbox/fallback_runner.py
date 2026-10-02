"""Legacy full-module diagnostics using Wasmtime.

Strict body-only execution uses Runtime inside WorkerPool, via FoundationRunner.
This compatibility API does not constitute independent admission evidence.
"""

from __future__ import annotations

from typing import List, Sequence
import wasmtime
from oeis_learn.data.models import ExecutionResult
from oeis_learn.sandbox.pipeline import decode_limbs


def evaluate_wat_single_fallback(
    wat_code: str, fuel_budget: int = 10000, terms_to_generate: int = 20,
    result_profile: str = "i64_scalar_v1"
) -> ExecutionResult:
    if type(fuel_budget) is not int or fuel_budget <= 0:
        raise ValueError("fuel_budget must be positive")
    if type(terms_to_generate) is not int or terms_to_generate <= 0:
        raise ValueError("terms_to_generate must be positive")
    if fuel_budget > 1_000_000 or terms_to_generate > 120 or len(wat_code.encode()) > 65536:
        raise ValueError("execution cap exceeded")
    if result_profile not in ("i64_scalar_v1", "i256x4_v1"):
        raise ValueError("explicit scalar or wide profile required")
    config = wasmtime.Config()
    config.memory_reservation = 0
    config.memory_guard_size = 0
    config.consume_fuel = True
    config.parallel_compilation = False
    with wasmtime.Engine(config) as engine:
        try:
            wasm = wasmtime.wat2wasm(wat_code)
        except wasmtime.WasmtimeError as exc:
            return ExecutionResult(status="PARSE_ERROR", consumed_fuel=0, output=[], error=str(exc))
        try:
            module = wasmtime.Module(engine, wasm)
        except wasmtime.WasmtimeError as exc:
            return ExecutionResult(
                status="COMPILE_ERROR", consumed_fuel=0, output=[], error=str(exc)
            )
        outputs, wide_outputs, total, maximum = [], [], 0, 0
        with module:
            for n in range(terms_to_generate):
                status, error = "SUCCESS", None
                with wasmtime.Store(engine) as store:
                    allowance = min(fuel_budget, 50_000_000 - total)
                    store.set_limits(memory_size=16*1024*1024, instances=1, memories=1, tables=10)
                    store.set_fuel(allowance)
                    try:
                        instance = wasmtime.Instance(store, module, [])
                        exports = instance.exports(store)
                        func = next(
                            (
                                exports.get(name)
                                for name in ("compute", "generate_term", "a")
                                if isinstance(exports.get(name), wasmtime.Func)
                            ),
                            None,
                        )
                        if func is None:
                            status, error = "MISSING_ENTRYPOINT", "No supported entrypoint found"
                        else:
                            params, results = func.type(store).params, func.type(store).results
                            if (
                                len(params) != 1
                                or str(params[0]) not in ("i32", "i64")
                                or [str(t) for t in results] != ["i64"] * (4 if result_profile == "i256x4_v1" else 1)
                            ):
                                status, error = (
                                    "COMPILE_ERROR",
                                    "Expected one integer input and one or four i64 results",
                                )
                            else:
                                value = func(store, n)
                                if len(results) == 4:
                                    outputs.append(decode_limbs(value))
                                    wide_outputs.append(list(value))
                                else:
                                    if type(value) is not int:
                                        raise ValueError("Expected an exact scalar integer")
                                    outputs.append(value)
                    except wasmtime.Trap as exc:
                        status = (
                            "OUT_OF_FUEL"
                            if exc.trap_code == wasmtime.TrapCode.OUT_OF_FUEL
                            else "EXECUTION_TRAP"
                        )
                        error = str(exc)
                    except (wasmtime.WasmtimeError, ValueError) as exc:
                        status, error = "EXECUTION_TRAP", str(exc)
                    used = allowance - store.get_fuel()
                total += used
                maximum = max(maximum, used)
                if status != "SUCCESS":
                    break
        return ExecutionResult(
            status=status,
            consumed_fuel=maximum,
            output=outputs,
            error=error,
            max_fuel=maximum,
            total_fuel=total,
            wide_output=wide_outputs,
        )


def evaluate_wat_batch_fallback(
    wat_programs: Sequence[str], fuel_budget: int = 10000, terms_to_generate: int = 20,
    result_profile: str = "i64_scalar_v1"
) -> List[ExecutionResult]:
    return [
        evaluate_wat_single_fallback(wat, fuel_budget, terms_to_generate, result_profile) for wat in wat_programs
    ]
