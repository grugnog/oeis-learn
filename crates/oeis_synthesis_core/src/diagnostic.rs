//! One-request core routing shared by qualification and framed transport; no public admission.
use crate::{evidence::{self,Evidence},fit,runtime::BigintRuntime,value::{digest,Resources},ExecutionBackend};
use serde::Deserialize;
use serde_json::json;
use std::{io::{self,Read},time::{Instant,Duration}};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    request:crate::protocol::Request,
    #[serde(default)] limits:Resources,
    #[serde(default)] evidence:Option<Evidence>,
    reference_sha256:String,
}
pub fn run(retained:&mut dyn ExecutionBackend)->Result<(),String>{
    let mut raw=Vec::new();io::stdin().take(4*1024*1024+1).read_to_end(&mut raw).map_err(|e|e.to_string())?;
    let output=dispatch(&raw,retained)?;
    println!("{}",serde_json::to_string(&output).map_err(|e|e.to_string())?);
    Ok(())
}
/// Shared core routing; callers own framing, aggregate deadlines and admission.
pub fn dispatch(raw:&[u8],retained:&mut dyn ExecutionBackend)->Result<serde_json::Value,String>{
    if raw.len()>4*1024*1024{return Err("frame_limit".into());}
    let envelope:Envelope=serde_json::from_slice(&raw).map_err(|e|e.to_string())?;
    envelope.request.validate_qualification()?;
    if envelope.request.candidates.len()!=1{return Err("diagnostic_requires_one_candidate".into());}
    let candidate=&envelope.request.candidates[0];
    let operation=match envelope.request.operation{crate::protocol::Operation::Fit=>"fit",crate::protocol::Operation::Execute=>"execute",crate::protocol::Operation::ReplayUnsat=>"replay_unsat"};
    let source=candidate.original_source.as_str();
    let indices=envelope.request.visible_indices.as_slice();
    let values=envelope.request.visible_values.as_slice();
    let language_profile=envelope.request.language_profile.as_str();
    if operation!="execute"{envelope.limits.validate()?;}
    if envelope.evidence.as_ref().is_some_and(|e|e.language_profile.as_str()!=language_profile){return Err("evidence_profile_mismatch".into());}
    if indices.is_empty()||indices.len()>100||source.len()>65536{return Err("invalid_request".into());}
    let deadline=Instant::now()+Duration::from_millis(u64::from(envelope.request.remaining_budget.wall_time_ms.min(2000)));
    let path=std::env::var("OEIS_BIGINT_WASM").unwrap_or("/opt/oeis-build-info/bigint-dependency-smoke.wasm".into());
    let bytes=std::fs::read(path).map_err(|e|e.to_string())?;
    let binary=std::fs::read(std::env::current_exe().map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let runtime_id=digest(serde_json::to_string(&json!({"guest_sha256":digest(&bytes),"binary_sha256":digest(&binary),"profile":language_profile})).unwrap().as_bytes());
    let output=if language_profile==retained.profile(){
        if operation=="fit"{
            fit::fit(source,indices,values,&envelope.limits,&candidate.abstraction_policy,envelope.request.seed,deadline,&runtime_id,&envelope.reference_sha256,language_profile,|source,indices,_|retained.execute(source,indices))
        }else if operation=="replay_unsat"{
            match evidence::replay(source,indices,values,&envelope.limits,envelope.evidence.as_ref().ok_or("missing_evidence")?,deadline){Ok(outcome)=>json!({"outcome":outcome}),Err(reason)=>json!({"outcome":"verification_failure","reason":reason})}
        }else if operation=="execute"{
            match retained.execute(source,indices){Ok(values)=>json!({"outcome":"ok","values":values}),Err(reason)=>json!({"outcome":"failure","reason":reason,"values":[]})}
        }else{return Err("unknown_operation".into());}
    }else if language_profile!="bigint-body/v1"{json!({"outcome":"unsupported","reason":"unknown language profile"})}else{match operation{
        "execute"=>{let mut runtime=BigintRuntime::new(&bytes)?;serde_json::to_value(runtime.evaluate(source,indices,&envelope.limits)).map_err(|e|e.to_string())?},
        "fit"=>{
            let mut runtime=BigintRuntime::new(&bytes)?;
            fit::fit(source,indices,values,&envelope.limits,&candidate.abstraction_policy,envelope.request.seed,deadline,&runtime_id,&envelope.reference_sha256,language_profile,|source,indices,limits|{
                let result=runtime.evaluate(source,indices,limits);if result.outcome=="ok"{Ok(result.values)}else{Err(result.reason.unwrap_or("runtime_failure".into()))}
            })
        },
        "replay_unsat"=>{match evidence::replay(source,indices,values,&envelope.limits,envelope.evidence.as_ref().ok_or("missing_evidence")?,deadline){Ok(outcome)=>json!({"outcome":outcome}),Err(reason)=>json!({"outcome":if reason=="verification_failure"{"verification_failure"}else{"resource_exhausted"},"reason":reason})}},
        _=>return Err("unknown_operation".into()),
    }};
    let encoded=serde_json::to_string(&output).map_err(|e|e.to_string())?;if encoded.len()>4*1024*1024{return Err("response_limit".into());}Ok(output)
}
