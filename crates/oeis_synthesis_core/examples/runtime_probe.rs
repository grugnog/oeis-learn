//! Bounded model-free diagnostic used by differential tests; not worker protocol.
use std::io::{self,Read};
use oeis_synthesis_core::{runtime::BigintRuntime,value::Resources};
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request { source:String,indices:Vec<String>,#[serde(default)] limits:Resources }
fn main() {
    let mut raw=Vec::new();io::stdin().take(4*1024*1024+1).read_to_end(&mut raw).unwrap();
    if raw.len()>4*1024*1024 {panic!("frame limit");}
    let req:Request=serde_json::from_slice(&raw).expect("invalid request");
    let path=std::env::var("OEIS_BIGINT_WASM").unwrap_or("/opt/oeis-build-info/bigint-dependency-smoke.wasm".into());
    let bytes=std::fs::read(path).expect("qualified runtime artifact required");
    let runtime=BigintRuntime::new(&bytes).expect("runtime module");
    println!("{}",serde_json::to_string(&runtime.evaluate(&req.source,&req.indices,&req.limits)).unwrap());
}
