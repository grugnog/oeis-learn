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
}

fn main() {
    let mut raw = Vec::new();
    io::stdin().take(1_048_577).read_to_end(&mut raw).unwrap();
    assert!(raw.len() <= 1_048_576);
    let request: Request = serde_json::from_slice(&raw).unwrap();
    assert!((1..=5).contains(&request.repetitions));
    assert!((1..=100).contains(&request.indices.len()));
    assert!(request.source.len() <= 65536);
    let path = std::env::var("OEIS_BIGINT_WASM")
        .unwrap_or("/opt/oeis-build-info/bigint-dependency-smoke.wasm".into());
    let bytes = std::fs::read(path).unwrap();
    let start = Instant::now();
    let mut runtime = BigintRuntime::new(&bytes).unwrap();
    let runtime_compile_seconds = start.elapsed().as_secs_f64();
    let mut samples = Vec::new();
    for index in 0..request.repetitions {
        if request.profile {
            // Alternate pair order; the first profiled sample retains cold setup.
            let ((result, timings), (control, control_timings)) = if index % 2 == 0 {
                let profiled = runtime.evaluate_profiled(&request.source, &request.indices, &Resources::default());
                (profiled, runtime.evaluate_measured(&request.source, &request.indices, &Resources::default()))
            } else {
                let control = runtime.evaluate_measured(&request.source, &request.indices, &Resources::default());
                (runtime.evaluate_profiled(&request.source, &request.indices, &Resources::default()), control)
            };
            let parity = serde_json::to_value(&result).unwrap() == serde_json::to_value(&control).unwrap();
            samples.push(json!({"result":result,"timings":timings,"control":{"result":control,"timings":control_timings},"parity":parity,"order":if index%2==0 {"profiled-first"}else{"control-first"}}));
            continue;
        }
        let (result, timings) = runtime.evaluate_measured(
            &request.source, &request.indices, &Resources::default(),
        );
        samples.push(json!({"result": result, "timings": timings}));
    }
    println!("{}", json!({
        "version": if request.profile {"bigint-measurements/v2"} else {"bigint-measurements/v1"},
        "runtime_compile_seconds": runtime_compile_seconds,
        "samples": samples,
    }));
}
