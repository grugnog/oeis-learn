"""Fallback in-memory WASM execution using wasmtime-py with fuel metering."""

from __future__ import annotations

from typing import List, Optional, Sequence
import wasmtime
from oeis_learn.data.models import ExecutionResult


def evaluate_wat_single_fallback(
    wat_code: str, fuel_budget: int = 10000, terms_to_generate: int = 20
) -> ExecutionResult:
    """Evaluates a single WAT program using Python wasmtime bindings."""
    try:
        # Configure engine with fuel
        config = wasmtime.Config()
        config.consume_fuel = True
        engine = wasmtime.Engine(config)

        # In-memory compile WAT text to WASM bytecode
        try:
            wasm_bytes = bytes(wasmtime.wat2wasm(wat_code))
        except Exception as e:
            return ExecutionResult(
                status="PARSE_ERROR",
                consumed_fuel=0,
                output=[],
                error=f"WAT parse error: {e}",
            )

        module = wasmtime.Module(engine, wasm_bytes)
        store = wasmtime.Store(engine)
        store.set_fuel(fuel_budget)

        instance = wasmtime.Instance(store, module, [])
    except Exception as e:
        error_str = str(e)
        status = "OUT_OF_FUEL" if "fuel" in error_str.lower() else "COMPILE_ERROR"
        return ExecutionResult(
            status=status,
            consumed_fuel=fuel_budget,
            output=[],
            error=error_str,
        )

    # Locate function entrypoint
    exports = instance.exports(store)
    func: Optional[wasmtime.Func] = None
    for name in ["compute", "generate_term", "a"]:
        exp = exports.get(name)
        if exp is not None and isinstance(exp, wasmtime.Func):
            func = exp
            break

    if func is None:
        # Check first exported function
        for val in exports.values():
            if isinstance(val, wasmtime.Func):
                func = val
                break

    if func is None:
        fuel_rem = store.get_fuel()
        return ExecutionResult(
            status="MISSING_ENTRYPOINT",
            consumed_fuel=max(0, fuel_budget - fuel_rem),
            output=[],
            error="No entrypoint function found",
        )

    outputs: List[int] = []
    param_types = func.type(store).params
    use_i64 = len(param_types) > 0 and param_types[0] == wasmtime.ValType.i64()

    for n in range(terms_to_generate):
        arg = wasmtime.Val.i64(n) if use_i64 else wasmtime.Val.i32(n)
        try:
            res = func(store, arg)
            term = int(res)  # type: ignore[call-overload]
            outputs.append(term)
        except Exception as e:
            error_str = str(e)
            try:
                fuel_rem = store.get_fuel()
            except Exception:
                fuel_rem = 0
            is_out_of_fuel = fuel_rem == 0 or "fuel" in error_str.lower()
            status = "OUT_OF_FUEL" if is_out_of_fuel else "EXECUTION_TRAP"
            return ExecutionResult(
                status=status,
                consumed_fuel=max(0, fuel_budget - fuel_rem),
                output=outputs,
                error=error_str,
            )

    fuel_rem = store.get_fuel()
    return ExecutionResult(
        status="SUCCESS",
        consumed_fuel=max(0, fuel_budget - fuel_rem),
        output=outputs,
        error=None,
    )


def evaluate_wat_batch_fallback(
    wat_programs: Sequence[str], fuel_budget: int = 10000, terms_to_generate: int = 20
) -> List[ExecutionResult]:
    """Evaluates a batch of WAT programs sequentially using fallback runner."""
    return [
        evaluate_wat_single_fallback(wat, fuel_budget, terms_to_generate)
        for wat in wat_programs
    ]
