"""Integration benchmark test for multi-threaded Rayon WASM execution throughput."""

import time
import pytest
from oeis_learn.sandbox.runner import HAS_NATIVE_EVALUATOR, WasmRunner


@pytest.mark.skipif(not HAS_NATIVE_EVALUATOR, reason="Requires native oeis_wasm_evaluator extension")
def test_batch_execution_throughput():
    runner = WasmRunner(fuel_budget=10000, use_fallback=False)

    # 1,000 diverse programs
    triangular = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (local $n64 i64)
            (local.set $n64 (i64.extend_i32_u (local.get $n)))
            (i64.div_u
                (i64.mul (local.get $n64) (i64.add (local.get $n64) (i64.const 1)))
                (i64.const 2)
            )
        )
    )
    """
    squares = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (local $n64 i64)
            (local.set $n64 (i64.extend_i32_u (local.get $n)))
            (i64.mul (local.get $n64) (local.get $n64))
        )
    )
    """
    fibonacci = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (local $a i64) (local $b i64) (local $temp i64) (local $i i32)
            (local.set $a (i64.const 0))
            (local.set $b (i64.const 1))
            (local.set $i (i32.const 0))
            (block $exit
                (loop $loop
                    (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
                    (local.set $temp (i64.add (local.get $a) (local.get $b)))
                    (local.set $a (local.get $b))
                    (local.set $b (local.get $temp))
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (br $loop)
                )
            )
            (local.get $a)
        )
    )
    """

    programs = [triangular, squares, fibonacci] * 334  # 1,002 programs
    start_time = time.perf_counter()
    results = runner.run_batch(programs, terms_to_generate=20)
    elapsed = time.perf_counter() - start_time

    throughput = len(programs) / elapsed
    print(f"\nEvaluated {len(programs)} WASM modules in {elapsed:.4f}s ({throughput:.1f} evals/sec)")

    assert len(results) == len(programs)
    assert all(r.status == "SUCCESS" for r in results)
    # Target: >= 500 evals/sec
    assert throughput >= 500.0, f"Expected >= 500 evals/sec, got {throughput:.1f}"
