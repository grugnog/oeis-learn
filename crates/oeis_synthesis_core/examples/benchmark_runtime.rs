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
    for _ in 0..request.repetitions {
        let (result, timings) = runtime.evaluate_measured(
            &request.source, &request.indices, &Resources::default(),
        );
        samples.push(json!({"result": result, "timings": timings}));
    }
    println!("{}", json!({
        "version": "bigint-measurements/v1",
        "runtime_compile_seconds": runtime_compile_seconds,
        "samples": samples,
    }));
}
