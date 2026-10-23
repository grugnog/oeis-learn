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
    fixture_with_probe(wat, fuel, false, 4 * 65536)
}

fn fixture_with_probe(wat: &str, fuel: u64, observed: bool, memory_bytes: usize) -> (BigintRuntime, Store<Limits>, Instance) {
    let runtime = BigintRuntime::new(&wat::parse_str(wat).unwrap()).unwrap();
    let limit = StoreLimitsBuilder::new().instances(2).memories(1).memory_size(memory_bytes).build();
    let mut store = Store::new(&runtime.engine, Limits { inner: limit, memory_denied: false, memory_probe: if observed {Some(MemoryProbe::default())}else{None} });
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
            assert_eq!(load(&mut calls, &instance, &mut store, &n, true, &mut InputObserver::new(None)), Ok(1));
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
            assert_eq!(load(&mut calls, &instance, &mut store, &IBig::from(1), true, &mut InputObserver::new(None)), Err("runtime_failure".into()));
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
            assert_eq!(load(&mut calls, &instance, &mut store, &IBig::from(1), false, &mut InputObserver::new(None)), Ok(1));
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
            assert_eq!(load(&mut calls, &instance, &mut store, &IBig::from(1), false, &mut InputObserver::new(None)), Err("invalid_handle".into()));
        }
    }
    let bytes = wat::parse_str("(module (func (export \"host_init\") (param i32 i32 i32 i32 i32 i32 i32) (result i32) unreachable))").unwrap();
    let mut runtime = BigintRuntime::new(&bytes).unwrap();
    let indices = vec!["0".into(), "1".into()];
    let (cached, _) = runtime.evaluate_measured("(program (return $index))", &indices, &Resources::default());
    let (baseline, _) = runtime.evaluate_lookup_baseline("(program (return $index))", &indices, &Resources::default());
    assert_eq!(serde_json::to_value(cached).unwrap(), serde_json::to_value(baseline).unwrap());
}

#[test]
fn input_detail_counts_values_and_bytes_without_changing_fuel() {
    let (_runtime, mut store, instance) = fixture(TRANSFER, 100_000);
    let mut calls = HostCalls::new(true);
    let mut detail = InputLoadMeasurements::default();
    let mut input = InputObserver::new(Some(&mut detail));
    for n in [IBig::ZERO, IBig::from(-257), IBig::from(512)] {
        assert_eq!(load(&mut calls, &instance, &mut store, &n, true, &mut input), Ok(1));
    }
    let observed_fuel = store.get_fuel().unwrap();
    assert_eq!(detail.literal_values_started, 3);
    assert_eq!(detail.index_values_started, 0);
    assert_eq!(detail.values_finished, 3);
    assert_eq!(detail.byte_calls_attempted, 5);
    assert_eq!(detail.byte_calls_returned, 5);
    assert_eq!(detail.first_byte_probe.first_byte.calls, 1);
    for name in ["begin_load", "magnitude_encode", "byte_transfer", "finish_load", "status_check"] {
        assert_eq!(detail.phases[name].calls, 3);
    }
    assert_eq!(detail.phases["index_parse"].calls, 0);
    assert_eq!(detail.phases["seal"].calls, 0);
    let (_runtime, mut store, instance) = fixture(TRANSFER, 100_000);
    let mut calls = HostCalls::new(true);
    for n in [IBig::ZERO, IBig::from(-257), IBig::from(512)] {
        load(&mut calls, &instance, &mut store, &n, true, &mut InputObserver::new(None)).unwrap();
    }
    assert_eq!(observed_fuel, store.get_fuel().unwrap());
}

#[test]
fn input_detail_preserves_partial_transfer_failure() {
    let guest=TRANSFER.replace("(export \"host_load_byte\") (param i32) (result i32) call $bump i32.const 1", "(export \"host_load_byte\") (param i32) (result i32) unreachable");
    let (_runtime, mut store, instance) = fixture(&guest, 100_000);
    let mut calls=HostCalls::new(true);
    let mut detail=InputLoadMeasurements::default();
    assert_eq!(load(&mut calls, &instance, &mut store, &IBig::from(512), false, &mut InputObserver::new(Some(&mut detail))), Err("runtime_failure".into()));
    assert_eq!(detail.index_values_started, 1);
    assert_eq!(detail.values_finished, 0);
    assert_eq!(detail.byte_calls_attempted, 1);
    assert_eq!(detail.byte_calls_returned, 0);
    assert_eq!(detail.phases["byte_transfer"].calls, 1);
    assert_eq!(detail.phases["finish_load"].calls, 0);
    assert_eq!(detail.phases["status_check"].calls, 0);
}

#[test]
fn first_byte_growth_and_trap_keep_fuel_and_partial_evidence() {
    for (trapping, memory_bytes) in [(false, 4 * 65536), (true, 4 * 65536), (false, 65536)] {
        let body = if trapping { "unreachable" } else { "call $bump i32.const 1" };
        let guest = TRANSFER.replace("(global $calls", "(memory 1 4) (global $calls")
            .replace("(export \"host_load_byte\") (param i32) (result i32) call $bump i32.const 1",
                &format!("(export \"host_load_byte\") (param i32) (result i32) global.get $calls i32.const 1 i32.eq if i32.const 1 memory.grow drop end {body}"));
        let mut outcomes = Vec::new();
        for observed in [false, true] {
            let (_runtime, mut store, instance) = fixture_with_probe(&guest, 100_000, observed, memory_bytes);
            let mut detail = InputLoadMeasurements::default();
            let mut input = InputObserver::new(if observed {Some(&mut detail)} else {None});
            let mut calls = HostCalls::new(true);
            let result = load(&mut calls, &instance, &mut store, &IBig::from(512), false, &mut input);
            input.finish_memory(store.data().memory_probe.as_ref());
            outcomes.push((result, store.get_fuel().unwrap(), store.data().memory_denied));
            if observed {
                let p = &detail.first_byte_probe;
                assert_eq!(p.first_byte.calls, 1);
                assert_eq!(p.first_byte_calls_returned, usize::from(!trapping));
                assert_eq!(p.memory_before_first.requests, 1);
                assert_eq!(p.memory_in_first.allowed, usize::from(memory_bytes > 65536));
                assert_eq!(p.memory_in_first.denied, usize::from(memory_bytes == 65536));
                assert_eq!(p.memory_in_first.requested_increase_bytes, 65536);
                assert_eq!(p.memory_after_first.requests, 0);
                assert_eq!(p.memory_events.len(), 2);
                assert_eq!(p.memory_events[1].scope, "first_byte");
                assert_eq!(p.memory_events_omitted, 0);
            }
        }
        assert_eq!(outcomes[0], outcomes[1]);
    }
}

#[test]
fn memory_denial_and_trace_caps_preserve_limiter_decisions() {
    let mut limit = Limits {inner:StoreLimitsBuilder::new().memory_size(65536).build(), memory_denied:false, memory_probe:Some(MemoryProbe::default())};
    assert!(limit.memory_growing(0,65536,None).unwrap());
    for _ in 0..20 {assert!(!limit.memory_growing(65536,131072,None).unwrap());}
    assert!(limit.memory_denied);
    let mut detail = InputLoadMeasurements::default();
    let mut input = InputObserver::new(Some(&mut detail));
    input.finish_memory(limit.memory_probe.as_ref());
    assert_eq!(detail.first_byte_probe.memory_before_first.denied, 20);
    assert_eq!(detail.first_byte_probe.memory_events.len(), 4);
    assert_eq!(detail.first_byte_probe.memory_events_omitted, 17);
    assert_eq!(detail.first_byte_probe.first_byte.calls, 0);
}

#[test]
fn memory_trace_is_bounded_across_fresh_terms_and_failure_callbacks() {
    let mut detail = InputLoadMeasurements::default();
    for term in 0..21 {
        let mut limits = Limits {inner:StoreLimitsBuilder::new().build(),memory_denied:false,memory_probe:Some(MemoryProbe::default())};
        if term<20 {assert!(limits.memory_growing(0,65536,None).unwrap());}
        assert!(limits.memory_grow_failed(wasmtime::Error::msg("fixture allocation failure")).is_err());
        InputObserver::new(Some(&mut detail)).finish_memory(limits.memory_probe.as_ref());
    }
    let p=&detail.first_byte_probe;
    assert_eq!(p.memory_events.len(),16);
    assert_eq!(p.memory_events_omitted,4);
    assert_eq!(p.memory_before_first.requests,20);
    assert_eq!(p.memory_before_first.grow_failed_callbacks,21);
    assert_eq!(p.terms_observed,21);
    assert_eq!(p.memory_probe_terms,21);
}

#[test]
fn growth_before_and_after_first_byte_stays_out_of_first_scope() {
    let guest=TRANSFER.replace("(global $calls", "(memory 1 6) (global $calls")
        .replace("(export \"host_begin_load\") (param i32) (result i32) call $bump", "(export \"host_begin_load\") (param i32) (result i32) i32.const 1 memory.grow drop call $bump")
        .replace("(export \"host_finish_load\") (param i32) (result i32) call $bump", "(export \"host_finish_load\") (param i32) (result i32) i32.const 1 memory.grow drop call $bump");
    let (_runtime, mut store, instance)=fixture_with_probe(&guest,100_000,true,6*65536);
    let mut detail=InputLoadMeasurements::default();
    let mut input=InputObserver::new(Some(&mut detail));
    let mut calls=HostCalls::new(true);
    for n in [IBig::from(512),IBig::ZERO] {
        load(&mut calls,&instance,&mut store,&n,true,&mut input).unwrap();
    }
    input.finish_memory(store.data().memory_probe.as_ref());
    let p=&detail.first_byte_probe;
    assert_eq!(p.first_byte.calls,1);
    assert_eq!(p.first_byte_calls_returned,1);
    assert_eq!(p.memory_before_first.requests,2);
    assert_eq!(p.memory_in_first.requests,0);
    assert_eq!(p.memory_after_first.requests,3);
    assert_eq!(p.memory_events_omitted,1);
    assert_eq!(p.memory_events[2].scope,"after_first");
    assert_eq!(detail.byte_calls_attempted,3);
}

#[test]
fn qualification_reservation_is_explicit_and_does_not_relax_memory_caps() {
    let guest=wat::parse_str("(module (memory (export \"memory\") 1 3) (func (export \"grow\") (result i32) i32.const 1 memory.grow))").unwrap();
    for cap in [65536,131072] {
        let mut observations=Vec::new();
        for reserved in [false,true] {
            let runtime=if reserved {BigintRuntime::new(&guest)}else{BigintRuntime::qualification_zero_growth_control(&guest)}.unwrap();
            let config=runtime.engine_configuration();
            assert_eq!(config["manifest"]["dynamic_memory_reserved_for_growth"],if reserved {65536}else{0});
            assert_eq!(config["manifest"]["purpose"],if reserved {"production"}else{"qualification-zero-growth-control"});
            let limits=StoreLimitsBuilder::new().memory_size(cap).build();
            let mut store=Store::new(&runtime.engine,Limits{inner:limits,memory_denied:false,memory_probe:Some(MemoryProbe::default())});
            store.limiter(|s|s);store.set_fuel(100_000).unwrap();store.set_epoch_deadline(1);
            let instance=Instance::new(&mut store,&runtime.runtime,&[]).unwrap();
            let memory=instance.get_memory(&mut store,"memory").unwrap();
            assert_eq!(memory.data_size(&store),65536); // reservation is not visible memory
            let answer=instance.get_typed_func::<(),i32>(&mut store,"grow").unwrap().call(&mut store,()).unwrap();
            observations.push((answer,store.get_fuel().unwrap(),store.data().memory_denied,memory.data_size(&store)));
            assert_eq!(store.data().memory_probe.as_ref().unwrap().counts.denied,usize::from(cap==65536));
        }
        assert_eq!(observations[0],observations[1]);
        assert_eq!(observations[0].0,if cap==65536 {-1}else{1});
        assert_eq!(observations[0].2,cap==65536);
        assert_eq!(observations[0].3,cap);
    }
}


#[test]
fn production_policy_is_shared_canonical_and_default64k() {
    let bytes=wat::parse_str("(module)").unwrap();
    let runtime=BigintRuntime::new(&bytes).unwrap();
    let policy=production_engine_policy();
    assert_eq!(policy["version"],"bigint-engine-policy/v2");
    assert_eq!(policy["dynamic_memory_reserved_for_growth"],65536);
    let config=runtime.engine_configuration();
    assert_eq!(config["manifest"]["policy_sha256"],production_engine_policy_identity());
    assert_eq!(config["manifest"]["dynamic_memory_reserved_for_growth"],65536);
    assert_eq!(config["manifest"]["purpose"],"production");
    let zero=BigintRuntime::qualification_zero_growth_control(&bytes).unwrap().engine_configuration();
    assert_ne!(config["sha256"],zero["sha256"]);
    assert_eq!(zero["manifest"]["dynamic_memory_reserved_for_growth"],0);
}
