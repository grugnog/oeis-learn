//! One bounded test request, never a production worker or public admission path.
use crate::{evidence::{self,Evidence},fit,runtime::BigintRuntime,value::{digest,Resources},ExecutionBackend};
use serde::Deserialize;
use serde_json::json;
use std::{io::{self,Read},time::{Instant,Duration}};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    operation:String, source:String, indices:Vec<String>, #[serde(default)] values:Vec<String>,
    #[serde(default)] limits:Resources, #[serde(default)] evidence:Option<Evidence>,
    #[serde(default)] reference_sha256:String, #[serde(default)] seed:u32,
    #[serde(default="none")] abstraction_policy:String,
    #[serde(default="bigint")] language_profile:String,
}
fn none()->String{"none".into()}
fn bigint()->String{"bigint-body/v1".into()}
pub fn run(retained:&mut dyn ExecutionBackend)->Result<(),String>{
    let mut raw=Vec::new();io::stdin().take(4*1024*1024+1).read_to_end(&mut raw).map_err(|e|e.to_string())?;
    if raw.len()>4*1024*1024{return Err("frame_limit".into());}
    let request:Request=serde_json::from_slice(&raw).map_err(|e|e.to_string())?;
    request.limits.validate()?;
    if request.indices.is_empty()||request.indices.len()>100||request.source.len()>65536{return Err("invalid_request".into());}
    let deadline=Instant::now()+Duration::from_secs(2);
    let path=std::env::var("OEIS_BIGINT_WASM").unwrap_or("/opt/oeis-build-info/bigint-dependency-smoke.wasm".into());
    let bytes=std::fs::read(path).map_err(|e|e.to_string())?;let runtime_id=digest(&bytes);
    let output=if request.language_profile==retained.profile(){
        if request.operation!="execute"{json!({"outcome":"unsupported","reason":"retained fitting requires an explicit supported translation"})}else{
            match retained.execute(&request.source,&request.indices){Ok(values)=>json!({"outcome":"ok","values":values}),Err(reason)=>json!({"outcome":"failure","reason":reason,"values":[]})}
        }
    }else if request.language_profile!="bigint-body/v1"{json!({"outcome":"unsupported","reason":"unknown language profile"})}else{match request.operation.as_str(){
        "execute"=>{let mut runtime=BigintRuntime::new(&bytes)?;serde_json::to_value(runtime.evaluate(&request.source,&request.indices,&request.limits)).map_err(|e|e.to_string())?},
        "fit"=>{
            let mut runtime=BigintRuntime::new(&bytes)?;
            fit::fit(&request.source,&request.indices,&request.values,&request.limits,&request.abstraction_policy,request.seed,deadline,&runtime_id,&request.reference_sha256,|source,indices,limits|{
                let result=runtime.evaluate(source,indices,limits);if result.outcome=="ok"{Ok(result.values)}else{Err(result.reason.unwrap_or("runtime_failure".into()))}
            })
        },
        "replay_unsat"=>{match evidence::replay(&request.source,&request.indices,&request.values,&request.limits,request.evidence.as_ref().ok_or("missing_evidence")?,deadline){Ok(outcome)=>json!({"outcome":outcome}),Err(reason)=>json!({"outcome":if reason=="verification_failure"{"verification_failure"}else{"resource_exhausted"},"reason":reason})}},
        _=>return Err("unknown_operation".into()),
    }};
    let encoded=serde_json::to_string(&output).map_err(|e|e.to_string())?;if encoded.len()>4*1024*1024{return Err("response_limit".into());}println!("{encoded}");Ok(())
}
