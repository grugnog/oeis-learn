//! Observation must preserve failures, fuel, coverage and cleanup on the same path.
use oeis_synthesis_core::{runtime::{BigintRuntime, PROFILE_PHASES}, value::Resources};

#[test]
fn profiling_early_guest_failure_accounts_for_both_cleanup_stages() {
    let bytes = wat::parse_str("(module (func (export \"host_init\") (param i32 i32 i32 i32 i32 i32 i32) (result i32) unreachable))").unwrap();
    let mut runtime = BigintRuntime::new(&bytes).unwrap();
    let source = "(program (return $index))";
    let indices = vec!["0".into()];
    let limits = Resources::default();
    let (ordinary, aggregate) = runtime.evaluate_measured(source, &indices, &limits);
    let (profiled, timings) = runtime.evaluate_profiled(source, &indices, &limits);
    assert_eq!(serde_json::to_value(ordinary).unwrap(), serde_json::to_value(profiled).unwrap());
    assert!(aggregate.stages.is_none());
    let stages = timings.stages.unwrap();
    assert_eq!(stages.policy, "bigint-runtime-profile/v1");
    assert_eq!(stages.terms_started, 1);
    assert_eq!(stages.terms_finished, 1);
    assert_eq!(stages.phases.len(), PROFILE_PHASES.len());
    for name in ["store_setup", "clock_start", "guest_init", "clock_stop", "store_drop"] {
        assert_eq!(stages.phases[name].calls, 1, "{name}");
    }
    for name in ["input_load", "link_instantiate", "compute", "serialize"] {
        assert_eq!(stages.phases[name].calls, 0, "{name}");
    }
    let attributed: f64 = stages.phases.values().map(|p| p.seconds).sum();
    assert!(attributed <= timings.execution_seconds + 1e-6);
    assert!(stages.unattributed_seconds >= 0.0);
}

#[test]
fn invalid_source_cannot_invent_term_measurements() {
    let bytes = wat::parse_str("(module)").unwrap();
    let mut runtime = BigintRuntime::new(&bytes).unwrap();
    let (result, timings) = runtime.evaluate_profiled("invalid", &["0".into()], &Resources::default());
    assert_eq!(result.outcome, "failure");
    assert!(result.values.is_empty());
    let stages = timings.stages.unwrap();
    assert_eq!(stages.terms_started, 0);
    assert_eq!(stages.terms_finished, 0);
    assert!(stages.phases.values().all(|p| p.calls == 0 && p.seconds == 0.0));
}

#[test]
fn nested_input_profile_is_empty_when_guest_init_fails() {
    let bytes=wat::parse_str("(module (func (export \"host_init\") (param i32 i32 i32 i32 i32 i32 i32) (result i32) unreachable))").unwrap();
    let mut runtime=BigintRuntime::new(&bytes).unwrap();
    let indices=vec!["0".into()];
    let (ordinary, aggregate)=runtime.evaluate_measured("(program (return $index))", &indices, &Resources::default());
    let (profiled, timings)=runtime.evaluate_input_profiled("(program (return $index))", &indices, &Resources::default());
    assert_eq!(serde_json::to_value(ordinary).unwrap(), serde_json::to_value(profiled).unwrap());
    assert!(aggregate.stages.is_none());
    let stages=timings.stages.unwrap();
    assert_eq!(stages.phases["input_load"].calls, 0);
    let detail=stages.input_load.unwrap();
    assert_eq!(detail.policy, "bigint-input-load-profile/v1");
    assert!(detail.phases.values().all(|p|p.calls==0 && p.seconds==0.0));
    assert_eq!(detail.literal_values_started, 0);
    assert_eq!(detail.index_values_started, 0);
    assert_eq!(detail.values_finished, 0);
    assert_eq!(detail.byte_calls_attempted, 0);
    assert_eq!(detail.unattributed_seconds, 0.0);
}
