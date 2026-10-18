use oeis_synthesis_core::{language::parse,lowering::{lower,validate_candidate},value::{integer,Resources}};
#[test] fn canonical_values_and_all_static_caps() {
    for s in ["-0","01","+1","1e2","1 ","١",""] {assert!(integer(s,4096).is_err());}
    let r=Resources::default();
    for source in [
        "(program (return (to_bigint (int 1))))",
        "(program (local $b0 bigint) (set $index (int 1)) (return $index))",
        "(program (local $c0 control) (loop $c0 (control 2) (set $c0 (control 1))) (return $index))",
        "(program (return (host_status)))",
        "(program (return (mul (control 2) (int 3))))",
    ] {assert!(parse(source,&r,false).is_err(),"{source}");}
    let nested=format!("(program (return {}$index{}))","(add (int 1) ".repeat(33),")".repeat(33));
    assert!(parse(&nested,&r,false).is_err());
    let oversized=" ".repeat(65537);assert!(parse(&oversized,&r,false).is_err());
    let nodes=format!("(program (local $b0 bigint) {} (return $index))","(set $b0 (int 1)) ".repeat(1000));
    assert!(parse(&nodes,&r,false).is_err());
}
#[test] fn closed_lowering_with_complete_types() {
    let r=Resources::default();
    let p=parse("(program (local $b0 bigint) (local $c0 control) (loop $c0 (to_control $index) (set $b0 (add (get $b0) (to_bigint (get $c0))))) (return (select (lt $index (int 0)) (int -1) (get $b0))))",&r,false).unwrap();
    let code=lower(&p,&r).unwrap();
    validate_candidate(&code.wasm).unwrap();
    wasmtime::Module::new(&wasmtime::Engine::default(),&code.wasm).unwrap();
}
#[test] fn forbidden_memory_table_start_and_privileged_imports() {
    for source in [
        "(module (memory 1))", "(module (table 1 funcref))",
        "(module (global i32 (i32.const 0)))", "(module (func $x) (start $x))",
        "(module (import \"oeis.big.v1\" \"host_init\" (func)))",
        "(module (import \"oeis.big.v1\" \"memory\" (memory 1)))",
    ] {assert!(validate_candidate(&wat::parse_str(source).unwrap()).is_err());}
}
