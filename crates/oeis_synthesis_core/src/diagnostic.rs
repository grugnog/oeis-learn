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
/// One installed, immutable artifact snapshot per supervised worker generation.
/// No candidates, stores, guest instances, results or resource profiles are cached.
#[derive(Default)]
pub(crate) struct DispatchContext { installed:Option<Installed> }
struct Installed { bytes:Vec<u8>, binary_hash:String, runtime:Option<BigintRuntime> }
impl Installed {
    fn load()->Result<Self,String> {
        let path=std::env::var("OEIS_BIGINT_WASM").unwrap_or("/opt/oeis-build-info/bigint-dependency-smoke.wasm".into());
        let mut bytes=Vec::new();
        std::fs::File::open(path).map_err(|e|e.to_string())?.take(16*1024*1024+1)
            .read_to_end(&mut bytes).map_err(|e|e.to_string())?;
        if bytes.len()>16*1024*1024{return Err("runtime_failure".into());}
        let binary=std::fs::read(std::env::current_exe().map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        Ok(Self{bytes,binary_hash:digest(&binary),runtime:None})
    }
    fn execute(&mut self,source:&str,indices:&[String],limits:&Resources)->Result<serde_json::Value,String> {
        if self.runtime.is_none(){self.runtime=Some(BigintRuntime::new(&self.bytes)?);}
        // Preserve the existing setup charge even on a warm engine. Each evaluate
        // still compiles the candidate and creates fresh stores/instances/clocks.
        serde_json::to_value(self.runtime.as_mut().unwrap().evaluate(source,indices,limits)).map_err(|e|e.to_string())
    }
}
pub fn run(retained:&mut dyn ExecutionBackend)->Result<(),String>{
    let mut raw=Vec::new();io::stdin().take(4*1024*1024+1).read_to_end(&mut raw).map_err(|e|e.to_string())?;
    let output=dispatch(&raw,retained)?;
    println!("{}",serde_json::to_string(&output).map_err(|e|e.to_string())?);
    Ok(())
}
/// Shared core routing; callers own framing, aggregate deadlines and admission.
pub fn dispatch(raw:&[u8],retained:&mut dyn ExecutionBackend)->Result<serde_json::Value,String>{
    dispatch_with_context(raw,retained,&mut DispatchContext::default())
}
pub(crate) fn dispatch_with_context(raw:&[u8],retained:&mut dyn ExecutionBackend,context:&mut DispatchContext)->Result<serde_json::Value,String>{
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
    if context.installed.is_none(){context.installed=Some(Installed::load()?);}
    let installed=context.installed.as_mut().unwrap();
    let runtime_id=crate::runtime::execution_identity(&digest(&installed.bytes),&installed.binary_hash,language_profile);
    let output=if language_profile==retained.profile(){
        if operation=="fit"{
            fit::fit(source,indices,values,&envelope.limits,&candidate.abstraction_policy,envelope.request.seed,deadline,&runtime_id,&envelope.reference_sha256,language_profile,|source,indices,_|retained.execute(source,indices))
        }else if operation=="replay_unsat"{
            match evidence::replay(source,indices,values,&envelope.limits,envelope.evidence.as_ref().ok_or("missing_evidence")?,deadline){Ok(outcome)=>json!({"outcome":outcome}),Err(reason)=>json!({"outcome":"verification_failure","reason":reason})}
        }else if operation=="execute"{
            match retained.execute(source,indices){Ok(values)=>json!({"outcome":"ok","values":values}),Err(reason)=>json!({"outcome":"failure","reason":reason,"values":[]})}
        }else{return Err("unknown_operation".into());}
    }else if language_profile!="bigint-body/v1"{json!({"outcome":"unsupported","reason":"unknown language profile"})}else{match operation{
        "execute"=>installed.execute(source,indices,&envelope.limits)?,
        "fit"=>{
            let mut runtime=BigintRuntime::new(&installed.bytes)?;
            fit::fit(source,indices,values,&envelope.limits,&candidate.abstraction_policy,envelope.request.seed,deadline,&runtime_id,&envelope.reference_sha256,language_profile,|source,indices,limits|{
                let result=runtime.evaluate_after_setup(source,indices,limits);if result.outcome=="ok"{Ok(result.values)}else{Err(result.reason.unwrap_or("runtime_failure".into()))}
            })
        },
        "replay_unsat"=>{match evidence::replay(source,indices,values,&envelope.limits,envelope.evidence.as_ref().ok_or("missing_evidence")?,deadline){Ok(outcome)=>json!({"outcome":outcome}),Err(reason)=>json!({"outcome":if reason=="verification_failure"{"verification_failure"}else{"resource_exhausted"},"reason":reason})}},
        _=>return Err("unknown_operation".into()),
    }};
    let encoded=serde_json::to_string(&output).map_err(|e|e.to_string())?;if encoded.len()>4*1024*1024{return Err("response_limit".into());}Ok(output)
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    fn snapshot(bytes:Vec<u8>)->Installed {
        Installed{bytes,binary_hash:digest(b"test-binary"),runtime:None}
    }
    #[test]
    fn failed_execution_keeps_engine_but_never_reuses_results_or_limits() {
        let guest=wat::parse_str("(module (func (export \"host_init\") (param i32 i32 i32 i32 i32 i32 i32) (result i32) unreachable))").unwrap();
        let mut installed=snapshot(guest);
        let source="(program (return $index))";
        let indices=vec!["0".into()];let limits=Resources::default();
        let a=installed.execute(source,&indices,&limits).unwrap();
        assert_eq!(a["reason"],"runtime_failure");
        // Test sentinel: rebuilding from bytes would now fail. Production never
        // mutates this installed snapshot during a worker generation.
        installed.bytes=b"not wasm".to_vec();
        let b=installed.execute(source,&indices,&limits).unwrap();
        assert_eq!(a,b);
        let invalid=installed.execute("invalid",&indices,&limits).unwrap();
        assert_ne!(invalid["reason"],a["reason"]);
        let mut small=limits.clone();small.aggregate_fuel=1;
        let limited=installed.execute(source,&indices,&small).unwrap();
        assert_eq!(limited["resource_identity"],small.identity());
        assert!(limited["values"].as_array().unwrap().is_empty());
        assert_eq!(installed.execute(source,&indices,&limits).unwrap(),a);
        assert!(snapshot(b"not wasm".to_vec()).execute(source,&indices,&limits).is_err());
    }
    #[test]
    fn installed_guest_reuse_matches_fresh_execution_and_independent_terms() {
        let path=std::env::var("OEIS_BIGINT_WASM").unwrap_or("/opt/oeis-build-info/bigint-dependency-smoke.wasm".into());
        if !std::path::Path::new(&path).is_file(){eprintln!("installed guest unavailable; pinned-container test required");return;}
        let bytes=std::fs::read(path).unwrap();let mut warm=snapshot(bytes.clone());
        let indices:Vec<String>=(0..100).map(|n|n.to_string()).collect();
        let limits=Resources::default();
        for k in [3i64,-7,1024,3] {
            let source=format!("(program (return (add (mul $index (int {k})) (int 5))))");
            let actual=warm.execute(&source,&indices,&limits).unwrap();
            let fresh=snapshot(bytes.clone()).execute(&source,&indices,&limits).unwrap();
            assert_eq!(actual,fresh);
            assert_eq!(actual["outcome"],"ok");
            assert_eq!(actual["values"],json!((0..100).map(|n|(n*k+5).to_string()).collect::<Vec<_>>()));
            let failed=warm.execute("(program (return (div $index (int 0))))",&indices,&limits).unwrap();
            assert_ne!(failed["outcome"],"ok");assert!(failed["values"].as_array().unwrap().is_empty());
        }
    }
}
