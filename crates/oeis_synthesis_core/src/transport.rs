//! Framing and batch routing only. Mathematical algorithms remain in the core.
use crate::{diagnostic,protocol::{Request,MAX_FRAME_BYTES,VERSION},value::Resources,ExecutionBackend};
use serde_json::{json,Value};
use std::{io::{self,Read,Write},time::Instant};

fn send(out:&mut impl Write,value:&Value)->Result<(),String>{
    let bytes=serde_json::to_vec(value).map_err(|e|e.to_string())?;
    if bytes.len()>MAX_FRAME_BYTES{return Err("response frame limit".into());}
    out.write_all(&(bytes.len() as u32).to_le_bytes()).and_then(|_|out.write_all(&bytes)).and_then(|_|out.flush()).map_err(|e|e.to_string())
}
pub fn serve(backend:&mut dyn ExecutionBackend)->Result<(),String>{
    let contract=std::env::var("OEIS_SYNTHESIS_CONTRACT").map_err(|_|"missing contract")?;
    let reference=std::env::var("OEIS_REFERENCE_SHA256").map_err(|_|"missing reference identity")?;
    let resources:Resources=serde_json::from_str(&std::env::var("OEIS_SYNTHESIS_RESOURCES").map_err(|_|"missing resource profile")?).map_err(|e|e.to_string())?;
    resources.validate()?;
    let input=io::stdin();let mut input=input.lock();let output=io::stdout();let mut output=output.lock();
    loop {
        let mut header=[0u8;4];
        match input.read(&mut header[..1]) {Ok(0)=>return Ok(()),Ok(_)=>(),Err(e)=>return Err(e.to_string())}
        input.read_exact(&mut header[1..]).map_err(|e|e.to_string())?;
        let size=u32::from_le_bytes(header) as usize;
        if size==0||size>MAX_FRAME_BYTES{return Err("frame limit before allocation".into());}
        let mut bytes=vec![0;size];input.read_exact(&mut bytes).map_err(|e|e.to_string())?;
        let request:Request=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
        request.validate_qualification()?;
        if request.contract_sha256!=contract||request.candidates.len()>request.remaining_budget.attempts as usize{return Err("contract or batch allowance mismatch".into());}
        let start=Instant::now();let budget=u64::from(request.remaining_budget.wall_time_ms);
        let mut completed=0;
        for candidate in &request.candidates {
            let elapsed=start.elapsed().as_millis() as u64;
            if elapsed>=budget {break;}
            let mut one=request.clone();one.candidates=vec![candidate.clone()];one.remaining_budget.wall_time_ms=(budget-elapsed).min(60000) as u32;
            // Keep the numerical profile immutable; the supervisor owns the aggregate deadline.
            let limits=resources.clone();
            let raw=serde_json::to_vec(&json!({"request":one,"limits":limits,"evidence":candidate.evidence,"reference_sha256":reference})).map_err(|e|e.to_string())?;
            let mut result=match diagnostic::dispatch(&raw,backend){Ok(v)=>v,Err(reason)=>json!({"outcome":"invalid_source","reason":reason,"assignments":[],"admitted":false})};
            result["candidate_id"]=json!(candidate.candidate_id);completed+=1;
            let unstarted:Vec<_>=request.candidates[completed..].iter().map(|c|&c.candidate_id).collect();
            send(&mut output,&json!({"protocol":VERSION,"request_id":request.request_id,"worker_generation":request.worker_generation,"contract_sha256":contract,"results":[result],"unstarted":unstarted,"complete":false}))?;
        }
        let unstarted:Vec<_>=request.candidates[completed..].iter().map(|c|&c.candidate_id).collect();
        send(&mut output,&json!({"protocol":VERSION,"request_id":request.request_id,"worker_generation":request.worker_generation,"contract_sha256":contract,"results":[],"unstarted":unstarted,"complete":true}))?;
    }
}
