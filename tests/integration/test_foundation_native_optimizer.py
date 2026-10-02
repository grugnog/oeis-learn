import pytest
from oeis_learn.sandbox.optimizer import optimize_wat_program
from oeis_learn.sandbox.pipeline import Runtime
from oeis_learn.sandbox.runner import WasmRunner


def test_legacy_tee_declaration_is_preserved():
    source = '(module (func (export "compute") (param i32) (result i64) (local $x i64) i64.const 7 local.tee $x))'
    assert optimize_wat_program(source).opt_wat == source
    assert WasmRunner().run_single(source).output == [7]*20


def test_typed_drop_rewrite_and_effects():
    source = "i64.const 3 drop nop i256.const 7"
    optimized = optimize_wat_program(source)
    assert optimized.passes_applied == ["typed-nop-and-pure-drop-v1"]
    assert Runtime().evaluate(source, [0, 1]).outputs == Runtime().evaluate(optimized.opt_wat, [0, 1]).reference_outputs
    effect = "i64.const 3 local.tee $v0 drop i256.const 1"
    assert optimize_wat_program(effect).opt_wat == effect


def test_trapping_expression_is_not_eliminated():
    source = '(module (func (export "compute") (param i32) (result i64) i64.const 1 i64.const 0 i64.div_s drop i64.const 7))'
    assert optimize_wat_program(source).opt_wat == source
    assert WasmRunner().run_single(source).status == "EXECUTION_TRAP"


def test_fresh_state_and_explicit_profile():
    source = '(module (global $g (mut i64) (i64.const 0)) (func (export "compute") (param i32) (result i64) global.get $g i64.const 1 i64.add global.set $g global.get $g))'
    assert WasmRunner().run_single(source).output == [1]*20
    wide = '(module (func (export "compute") (param i32) (result i64 i64 i64 i64) i64.const -1 i64.const -1 i64.const -1 i64.const -1))'
    runner = WasmRunner()
    assert runner.run_single(wide).status == "COMPILE_ERROR"
    assert runner.run_single(wide, result_profile="i256x4_v1").output == [-1]*20
    assert runner.run_batch([wide], result_profile="i256x4_v1")[0].output == [-1]*20


def test_native_is_explicitly_pending():
    with pytest.raises(ValueError, match="pending"):
        WasmRunner(backend="rust")
