//! Native orchestration. Results are provisional and have no admission authority.
use crate::{affine::{self,Solution},evidence,language,parameters,smt,substitute,symbolic::Problem,value::{bits,digest,Resources}};
use dashu_int::IBig;
use serde_json::{json,Value};
use std::time::{Instant,Duration};
pub fn fit<F>(source:&str,indices:&[String],values:&[String],r:&Resources,policy:&str,seed:u32,deadline:Instant,runtime_identity:&str,reference_identity:&str,mut execute:F)->Value
where F:FnMut(&str,&[String],&Resources)->Result<Vec<String>,String> {
    let start=Instant::now();let deadline=deadline.min(start+Duration::from_secs(2));
    let mut phase="validation";let mut completed=Vec::new();let mut assignments=Vec::new();let mut proof=None;let mut uniqueness="unchecked".to_string();let mut scope=0;
    let run=(||->Result<String,String>{
        let original=language::parse(source,r,true)?;
        let (program,map)=match policy{"none"=>(original,vec![]),"eligible-literals/v1"=>parameters::abstract_literals(&original,r)?,_=>return Err("unsupported:abstraction_policy".into())};
        phase="symbolic";let problem=Problem::build(&program,indices,values,r,deadline)?;
        phase="affine";
        let mut discovered=Vec::new();
        let solution=if let Some(rows)=&problem.affine_rows{affine::solve(rows,problem.parameters,deadline)?}else{Solution::RankDeficient};
        match solution {
            Solution::Unique(xs)=>{if xs.iter().any(|n|bits(n)>4096){return Err("coefficient_limit".into());}discovered.push(xs);uniqueness="unique_in_scope".into();scope=4096;},
            Solution::Inconsistent|Solution::NonIntegral=>{scope=4096;proof=Some(evidence::record(source,indices,values,r,&problem,4096,policy));completed.push(4096);return Ok("unsat_in_scope".into());},
            Solution::RankDeficient=>{
                phase="smt";
                for bound in [16,64,256,1024,4096] {
                    let found=smt::search(&problem,r,bound,deadline,seed,8)?;scope=bound;uniqueness=found.uniqueness;
                    discovered.extend(found.assignments);
                    if found.outcome=="unsat_in_scope" {completed.push(bound);proof=Some(evidence::record(source,indices,values,r,&problem,bound,policy));continue;}
                    if discovered.is_empty(){return Err(format!("{}:{}",found.outcome,found.reason.unwrap_or_default()));}
                    break;
                }
                if discovered.is_empty(){return Ok("unsat_in_scope".into());}
            }
        }
        discovered.sort_by_key(|xs|{let nonzero=xs.iter().filter(|n|**n!=IBig::ZERO).count();let magnitude:usize=xs.iter().map(bits).sum();(nonzero,magnitude,xs.clone())});discovered.dedup();
        phase="production";let mut rejected=Vec::new();
        for xs in discovered.into_iter().take(8) {
            if Instant::now()>=deadline{break;}
            let final_source=match substitute::substitute(&program,&xs,r){Ok(s)=>s,Err(e)=>{rejected.push(e);continue;}};
            let mut limits=r.clone();limits.evaluation_seconds=limits.evaluation_seconds.min(deadline.saturating_duration_since(Instant::now()).as_secs_f64());limits.term_seconds=limits.term_seconds.min(limits.evaluation_seconds);
            let production=match execute(&final_source,indices,&limits){Ok(v)=>v,Err(e)=>{rejected.push(e);continue;}};
            if Instant::now()>=deadline{break;}
            if production!=values {rejected.push("production_disagreement".into());continue;}
            assignments.push(json!({"outcome":"assignment_found","source":final_source,"source_sha256":digest(final_source.as_bytes()),"original_source_sha256":digest(source.as_bytes()),"transformed_source":program.source,"occurrences":map,"assignment":xs.iter().enumerate().map(|(i,n)|(format!("p{i}"),n.to_string())).collect::<std::collections::BTreeMap<_,_>>(),"indices":indices,"values":production,"prompt_sha256":evidence::prompt_identity(indices,values),"resource_sha256":r.identity(),"runtime_sha256":runtime_identity,"reference_sha256":reference_identity,"language_profile":"bigint-body/v1","coverage":indices.len(),"coefficient_bits":scope,"uniqueness":uniqueness}));
        }
        if assignments.is_empty(){return Err(if rejected.iter().any(|e|e=="production_disagreement"){"verification_failure".into()}else{"production_limit_or_deadline".into()});}
        Ok("assignment_found".into())
    })();
    let (outcome,reason)=match run{Ok(outcome)=>(outcome,None),Err(e)=>{let status=if e.starts_with("unsupported"){"unsupported"}else if e.starts_with("unknown"){"unknown"}else if e=="verification_failure"{ "verification_failure" }else if ["invalid_source","invalid_prompt"].contains(&e.as_str()){ "invalid_source" }else{"resource_exhausted"};(status.to_string(),Some(e))}};
    json!({"outcome":outcome,"reason":reason,"phase":phase,"assignments":assignments,"completed_bounds":completed,"coefficient_bits":scope,"uniqueness":uniqueness,"evidence":proof,"elapsed_seconds":start.elapsed().as_secs_f64(),"admitted":false,"metrics":{"memory_bytes":null,"unavailable_reason":"native RSS measured by owning process supervisor"}})
}
