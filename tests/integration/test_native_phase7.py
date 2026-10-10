"""Mandatory native acceptance: import failures are failures, never skips."""

import oeis_wasm_evaluator as native
import pytest
from oeis_learn.sandbox.pipeline import Runtime, Limits
from oeis_learn.sandbox.runner import WasmRunner, FoundationRunner
from oeis_learn.sandbox.native_runtime import native_identity


def test_current_build_four_limbs_single_batch_and_fresh_state(tmp_path):
    assert native_identity()["source_sha256"].startswith("sha256:")
    runner = WasmRunner(backend="rust")
    wide = '(module (func (export "compute") (param i32) (result i64 i64 i64 i64) i64.const -1 i64.const -1 i64.const -1 i64.const -1))'
    assert runner.run_single(wide, result_profile="i256x4_v1").output == [-1] * 20
    assert runner.run_single(wide).status == "COMPILE_ERROR"
    assert runner.run_batch([wide, wide], result_profile="i256x4_v1")[1].output == [-1] * 20
    state = '(module (global $g (mut i64) (i64.const 0)) (func (export "compute") (param i32) (result i64) global.get $g i64.const 1 i64.add global.set $g global.get $g))'
    assert runner.run_single(state).output == [1] * 20
    with FoundationRunner(tmp_path, backend="rust") as strict:
        result = strict.run_batch(["i256.const -1", "i256.const 2"], [0, 1], request_ids=["a", "b"])
    assert result[0].outputs == result[0].reference_outputs == ["-1", "-1"]
    assert result[1].outputs == result[1].reference_outputs == ["2", "2"]


@pytest.mark.parametrize(
    "source,outcome",
    [
        ("i256.const " + str(2**255 - 1) + " i256.const 1 i256.add", "numeric_limit"),
        ("i64.const 1 i64.const 0 i64.div_s drop i256.const 7", "runtime_failure"),
        ("( loop br 0 ) i256.const 7", "execution_limit"),
        ("i64.const 3 local.tee $v0 drop local.get $v0 i64.const 0 i64.const 0 i64.const 0", None),
    ],
)
def test_strict_classification_and_tee(source, outcome):
    for backend in ("python_wasmtime", "rust"):
        value = Runtime(backend=backend).evaluate(source, [0], Limits(fuel_per_term=1000))
        assert value.outcome == outcome, value
        assert value.fuel == sum(value.fuel_terms)


def test_exact_signature_and_no_arbitrary_export():
    source = '(module (func (export "other") (param i32) (result i64) i64.const 1))'
    assert native.evaluate_wat_single(source).status == "MISSING_ENTRYPOINT"
    wrong = '(module (func (export "compute") (param i32 i32) (result i64) i64.const 1))'
    assert native.evaluate_wat_single(wrong).status == "COMPILE_ERROR"
    with pytest.raises(ValueError):
        WasmRunner(backend="rust", use_fallback=True)
