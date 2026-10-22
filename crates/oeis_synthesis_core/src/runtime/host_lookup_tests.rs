use super::*;

const TRANSFER: &str = r#"(module
  (global $calls (mut i32) (i32.const 0))
  (func $bump global.get $calls i32.const 1 i32.add global.set $calls)
  (func (export "host_begin_load") (param i32) (result i32) call $bump i32.const 1)
  (func (export "host_load_byte") (param i32) (result i32) call $bump i32.const 1)
  (func (export "host_finish_load") (param i32) (result i32) call $bump i32.const 1)
  (func (export "host_status") (result i32) call $bump i32.const 0)
  (func (export "count") (result i32) global.get $calls)
)"#;

fn fixture(wat: &str, fuel: u64) -> (BigintRuntime, Store<Limits>, Instance) {
    let runtime = BigintRuntime::new(&wat::parse_str(wat).unwrap()).unwrap();
    let limit = StoreLimitsBuilder::new().instances(2).memories(1).build();
    let mut store = Store::new(&runtime.engine, Limits { inner: limit, memory_denied: false });
    store.limiter(|s| s);
    store.set_fuel(fuel).unwrap();
    store.set_epoch_deadline(1);
    let instance = Instance::new(&mut store, &runtime.runtime, &[]).unwrap();
    (runtime, store, instance)
}

#[test]
fn cache_preserves_every_transfer_call_and_fuel() {
    let observe = |cached| {
        let (_runtime, mut store, instance) = fixture(TRANSFER, 100_000);
        let mut calls = HostCalls::new(cached);
        for n in [IBig::ZERO, IBig::from(-257), IBig::from(512)] {
            assert_eq!(load(&mut calls, &instance, &mut store, &n, true), Ok(1));
        }
        let fuel = store.get_fuel().unwrap();
        let count = instance.get_typed_func::<(), i32>(&mut store, "count").unwrap().call(&mut store, ()).unwrap();
        (count, fuel)
    };
    assert_eq!(observe(true), observe(false));
    assert_eq!(observe(true).0, 14); // begin + bytes + finish + status, for each value
}

#[test]
fn lookup_is_lazy_and_missing_or_mistyped_status_preserves_prior_calls() {
    for guest in [TRANSFER.replace("(export \"host_status\")", "(export \"unused\")"),
                  TRANSFER.replace("(export \"host_status\") (result i32)", "(export \"host_status\") (param i32) (result i32)")] {
        for cached in [false, true] {
            let (_runtime, mut store, instance) = fixture(&guest, 100_000);
            let mut calls = HostCalls::new(cached);
            assert_eq!(load(&mut calls, &instance, &mut store, &IBig::from(1), true), Err("runtime_failure".into()));
            let count = instance.get_typed_func::<(), i32>(&mut store, "count").unwrap().call(&mut store, ()).unwrap();
            assert_eq!(count, 3); // lookup failure must not precede begin/byte/finish
        }
    }
}

#[test]
fn cache_is_fresh_for_each_store_and_fuel_failure() {
    for cached in [false, true] {
        for _ in 0..2 {
            let (_runtime, mut store, instance) = fixture(TRANSFER, 100_000);
            let mut calls = HostCalls::new(cached);
            assert_eq!(load(&mut calls, &instance, &mut store, &IBig::from(1), false), Ok(1));
            let count = instance.get_typed_func::<(), i32>(&mut store, "count").unwrap().call(&mut store, ()).unwrap();
            assert_eq!(count, 4);
            store.set_fuel(0).unwrap();
            assert_eq!(calls.call0(&instance, &mut store, "host_status"), Err("term_fuel".into()));
        }
    }
}

#[test]
fn sticky_failure_and_early_guest_trap_match_the_lookup_baseline() {
    let sticky = TRANSFER.replace("call $bump i32.const 0)", "call $bump i32.const 1)");
    for cached in [false, true] {
        let (_runtime, mut store, instance) = fixture(&sticky, 100_000);
        let mut calls = HostCalls::new(cached);
        for _ in 0..2 {
            assert_eq!(load(&mut calls, &instance, &mut store, &IBig::from(1), false), Err("invalid_handle".into()));
        }
    }
    let bytes = wat::parse_str("(module (func (export \"host_init\") (param i32 i32 i32 i32 i32 i32 i32) (result i32) unreachable))").unwrap();
    let mut runtime = BigintRuntime::new(&bytes).unwrap();
    let indices = vec!["0".into(), "1".into()];
    let (cached, _) = runtime.evaluate_measured("(program (return $index))", &indices, &Resources::default());
    let (baseline, _) = runtime.evaluate_lookup_baseline("(program (return $index))", &indices, &Resources::default());
    assert_eq!(serde_json::to_value(cached).unwrap(), serde_json::to_value(baseline).unwrap());
}
