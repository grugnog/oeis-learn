//! Bounded measurement entry point; delegates all semantics to production APIs.
use oeis_synthesis_core::{runtime::BigintRuntime, value::Resources};
use serde::Deserialize;
use serde_json::json;
use std::io::{self, Read};
use std::time::Instant;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    source: String,
    indices: Vec<String>,
    repetitions: usize,
    #[serde(default)]
    profile: bool,
    #[serde(default)]
    compare_host_lookups: bool,
    #[serde(default)]
    profile_input_load: bool,
    #[serde(default)]
    growth_reservation_control: bool,
}

fn main() {
    let mut raw = Vec::new();
    io::stdin().take(1_048_577).read_to_end(&mut raw).unwrap();
    assert!(raw.len() <= 1_048_576);
    let request: Request = serde_json::from_slice(&raw).unwrap();
    assert!((1..=5).contains(&request.repetitions));
    assert!((1..=100).contains(&request.indices.len()));
    assert!(request.source.len() <= 65536);
    assert!(!request.compare_host_lookups || request.profile);
    assert!(!request.profile_input_load || request.profile);
    assert!(!request.growth_reservation_control || (request.profile && request.profile_input_load && !request.compare_host_lookups));
    let path = std::env::var("OEIS_BIGINT_WASM")
        .unwrap_or("/opt/oeis-build-info/bigint-dependency-smoke.wasm".into());
    let bytes = std::fs::read(path).unwrap();
    let start = Instant::now();
    let mut runtime = if request.growth_reservation_control {BigintRuntime::qualification_growth_control(&bytes)}else{BigintRuntime::new(&bytes)}.unwrap();
    let runtime_compile_seconds = start.elapsed().as_secs_f64();
    let mut samples = Vec::new();
    for index in 0..request.repetitions {
        if request.profile {
            let mut baseline = if request.compare_host_lookups && index % 2 == 0 {
                Some(runtime.evaluate_lookup_baseline(&request.source, &request.indices, &Resources::default()))
            } else { None };
            // Alternate pair order; retain the first sample separately and
            // record baseline order so shared process warmup is explicit.
            let ((result, timings), (control, control_timings)) = if index % 2 == 0 {
                let profiled = if request.profile_input_load {runtime.evaluate_input_profiled(&request.source, &request.indices, &Resources::default())}else{runtime.evaluate_profiled(&request.source, &request.indices, &Resources::default())};
                (profiled, runtime.evaluate_measured(&request.source, &request.indices, &Resources::default()))
            } else {
                let control = runtime.evaluate_measured(&request.source, &request.indices, &Resources::default());
                (if request.profile_input_load {runtime.evaluate_input_profiled(&request.source, &request.indices, &Resources::default())}else{runtime.evaluate_profiled(&request.source, &request.indices, &Resources::default())}, control)
            };
            let parity = serde_json::to_value(&result).unwrap() == serde_json::to_value(&control).unwrap();
            if request.compare_host_lookups && baseline.is_none() {
                baseline=Some(runtime.evaluate_lookup_baseline(&request.source, &request.indices, &Resources::default()));
            }
            let mut sample=json!({"result":result,"timings":timings,"control":{"result":control,"timings":control_timings},"parity":parity,"order":if index%2==0 {"profiled-first"}else{"control-first"}});
            if let Some((baseline_result,baseline_timings))=baseline {
                let parity=serde_json::to_value(&baseline_result).unwrap()==sample["control"]["result"];
                sample["lookup_baseline"]=json!({"result":baseline_result,"timings":baseline_timings,"parity":parity,"order":if index%2==0 {"baseline-first"}else{"baseline-last"}});
            }
            samples.push(sample);
            continue;
        }
        let (result, timings) = runtime.evaluate_measured(
            &request.source, &request.indices, &Resources::default(),
        );
        samples.push(json!({"result": result, "timings": timings}));
    }
    // Linux qualification role: process high-water RSS includes JIT and both
    // ordinary/profiled execution, measured before report serialization.
    let peak_process_rss_bytes=std::fs::read_to_string("/proc/self/status").ok().and_then(|s| {
        s.lines().find_map(|line|line.strip_prefix("VmHWM:").and_then(|value|value.split_whitespace().next()?.parse::<u64>().ok()).and_then(|kb|kb.checked_mul(1024)))
    });
    println!("{}", json!({
        "version": if request.profile {"bigint-measurements/v2"} else {"bigint-measurements/v1"},
        "profile_input_load": request.profile_input_load,
        "host_lookup_comparison": if request.compare_host_lookups {json!({"version":"per-term-host-lookups/v1","cached":"lazy-per-term","baseline":"per-call"})}else{serde_json::Value::Null},
        "runtime_compile_seconds": runtime_compile_seconds,
        "engine_configuration":runtime.engine_configuration(),
        "peak_process_rss_bytes":peak_process_rss_bytes,
        "samples": samples,
    }));
}
