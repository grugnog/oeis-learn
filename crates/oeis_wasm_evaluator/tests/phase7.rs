use oeis_wasm_evaluator::{
    engine::create_fuel_engine,
    sandbox::{evaluate_profile, ExecutionStatus},
};

#[test]
fn exact_profile_and_fresh_state() {
    let engine = create_fuel_engine().unwrap();
    let source = "(module (global $g (mut i64) (i64.const 0)) (func (export \"compute\") (param i32) (result i64) global.get $g i64.const 1 i64.add global.set $g global.get $g))";
    let result = evaluate_profile(&engine, source, 1000, 20, "i64_scalar_v1");
    assert_eq!(result.status, ExecutionStatus::Success);
    assert_eq!(result.output, vec![1; 20]);
    let wide = "(module (func (export \"compute\") (param i32) (result i64 i64 i64 i64) i64.const -1 i64.const -1 i64.const -1 i64.const -1))";
    assert_eq!(
        evaluate_profile(&engine, wide, 1000, 2, "i64_scalar_v1").status,
        ExecutionStatus::CompileError
    );
    assert_eq!(
        evaluate_profile(&engine, wide, 1000, 2, "i256x4_v1").wide_output,
        vec![[-1; 4]; 2]
    );
}

#[test]
fn tee_trap_and_entrypoint() {
    let engine = create_fuel_engine().unwrap();
    for (body, status) in [
        ("i64.const 7 local.tee $x", ExecutionStatus::Success),
        (
            "i64.const 1 i64.const 0 i64.div_s",
            ExecutionStatus::ExecutionTrap,
        ),
        ("(loop br 0) i64.const 1", ExecutionStatus::OutOfFuel),
    ] {
        let source = format!("(module (func (export \"compute\") (param i32) (result i64) (local $x i64) {body}))");
        assert_eq!(
            evaluate_profile(&engine, &source, 1000, 2, "i64_scalar_v1").status,
            status
        );
    }
}
