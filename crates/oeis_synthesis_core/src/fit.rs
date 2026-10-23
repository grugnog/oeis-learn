//! Native orchestration. Results are provisional and have no admission authority.
use crate::{affine::{self,Solution},evidence,language,parameters,smt,substitute,symbolic::Problem,value::{bits,digest,Resources}};
use dashu_int::IBig;
use serde_json::{json,Value};
use std::time::Instant;
#[cfg(test)]
use std::time::Duration;
pub fn fit<F>(source:&str,indices:&[String],values:&[String],r:&Resources,policy:&str,seed:u32,deadline:Instant,runtime_identity:&str,reference_identity:&str,language_profile:&str,execute:F)->Value
where F:FnMut(&str,&[String],&Resources)->Result<Vec<String>,String> {
    fit_with(source,indices,values,r,policy,seed,deadline,runtime_identity,reference_identity,language_profile,execute,Instant::now,smt::search)
}
// Clock/search injection is private and used only by deterministic phase-boundary tests.
fn fit_with<F,N,S>(source:&str,indices:&[String],values:&[String],r:&Resources,policy:&str,seed:u32,deadline:Instant,runtime_identity:&str,reference_identity:&str,language_profile:&str,mut execute:F,mut now:N,mut search:S)->Value
where F:FnMut(&str,&[String],&Resources)->Result<Vec<String>,String>,
      N:FnMut()->Instant,
      S:FnMut(&Problem,&Resources,usize,Instant,u32,usize)->Result<smt::Search,String> {
    let start=now();let budget=crate::fit_budget::FitBudget::new(start,deadline);let mut searches=Vec::new();
    let mut phase="validation";let mut completed=Vec::new();let mut assignments=Vec::new();let mut proof=None;let mut uniqueness="unchecked".to_string();let mut scope=0;
    let run=(||->Result<String,String>{
        if now()>=budget.search{return Err("deadline".into());}
        let original=match language_profile{"bigint-body/v1"=>language::parse(source,r,true)?,"wat_i256_checked_v1"=>crate::retained::logical(source,r)?,_=>return Err("unsupported:language_profile".into())};
        let (program,map)=match policy{"none"=>(original,vec![]),"eligible-literals/v1"=>parameters::abstract_literals(&original,r)?,_=>return Err("unsupported:abstraction_policy".into())};
        phase="symbolic";let mut problem=Problem::build(&program,indices,values,r,budget.search)?;problem.checked_signed256=language_profile=="wat_i256_checked_v1";
        phase="affine";
        let mut discovered=Vec::new();
        let solution=if let Some(rows)=&problem.affine_rows{affine::solve(rows,problem.parameters,budget.search)?}else{Solution::RankDeficient};
        match solution {
            Solution::Unique(xs)=>{if xs.iter().any(|n|bits(n)>4096){return Err("coefficient_limit".into());}discovered.push(xs);uniqueness="unique_in_scope".into();scope=4096;},
            Solution::Inconsistent|Solution::NonIntegral=>{scope=4096;proof=Some(evidence::record(source,indices,values,r,&problem,4096,policy));completed.push(4096);return Ok("unsat_in_scope".into());},
            Solution::RankDeficient=>{
                phase="smt";
                for bound in [16,64,256,1024,4096] {
                    let found=search(&problem,r,bound,budget.search,seed,8)?;
                    searches.push(json!({"coefficient_bits":bound,"outcome":found.outcome,"reason":found.reason,"uniqueness":found.uniqueness,"assignments":found.assignments.len(),"checks":found.checks}));
                    scope=bound;uniqueness=found.uniqueness;
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
            if now()>=budget.production{break;}
            let mut final_source=match substitute::substitute(&program,&xs,r){Ok(s)=>s,Err(e)=>{rejected.push(e);continue;}};
            if problem.checked_signed256 {final_source=match crate::retained::final_source(&language::parse(&final_source,r,false)?){Ok(s)=>s,Err(e)=>{rejected.push(e);continue;}};}
            let remaining=budget.production.saturating_duration_since(now());if remaining.is_zero(){break;}
            let mut limits=r.clone();limits.evaluation_seconds=limits.evaluation_seconds.min(remaining.as_secs_f64());limits.term_seconds=limits.term_seconds.min(limits.evaluation_seconds);
            let production=match execute(&final_source,indices,&limits){Ok(v)=>v,Err(e)=>{rejected.push(e);continue;}};
            if now()>=budget.production{break;}
            if production!=values {rejected.push("production_disagreement".into());continue;}
            assignments.push(json!({"outcome":"assignment_found","source":final_source,"source_sha256":digest(final_source.as_bytes()),"original_source_sha256":digest(source.as_bytes()),"transformed_source":program.source,"occurrences":map,"assignment":xs.iter().enumerate().map(|(i,n)|(format!("p{i}"),n.to_string())).collect::<std::collections::BTreeMap<_,_>>(),"indices":indices,"values":production,"prompt_sha256":evidence::prompt_identity(indices,values),"resource_sha256":r.identity(),"runtime_sha256":runtime_identity,"reference_sha256":reference_identity,"language_profile":language_profile,"coverage":indices.len(),"coefficient_bits":scope,"uniqueness":uniqueness}));
        }
        if assignments.is_empty(){return Err(if rejected.iter().any(|e|e=="production_disagreement"){"verification_failure".into()}else{"production_limit_or_deadline".into()});}
        Ok("assignment_found".into())
    })();
    let (outcome,reason)=match run{Ok(outcome)=>(outcome,None),Err(e)=>{let status=if e.starts_with("unsupported"){"unsupported"}else if e.starts_with("unknown"){"unknown"}else if e=="verification_failure"{ "verification_failure" }else if ["invalid_source","invalid_prompt"].contains(&e.as_str()){ "invalid_source" }else{"resource_exhausted"};(status.to_string(),Some(e))}};
    json!({"outcome":outcome,"reason":reason,"phase":phase,"assignments":assignments,"completed_bounds":completed,"coefficient_bits":scope,"uniqueness":uniqueness,"evidence":proof,"elapsed_seconds":now().saturating_duration_since(start).as_secs_f64(),"timing":{"policy":crate::fit_budget::POLICY,"native_allowance_seconds":budget.response.saturating_duration_since(start).as_secs_f64(),"search_allowance_seconds":budget.search.saturating_duration_since(start).as_secs_f64(),"production_reserve_seconds":budget.production.duration_since(budget.search).as_secs_f64(),"response_reserve_seconds":budget.response.duration_since(budget.production).as_secs_f64()},"searches":searches,"admitted":false,"metrics":{"memory_bytes":null,"unavailable_reason":"native RSS measured by owning process supervisor"}})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit_budget::FitBudget;
    use std::cell::Cell;

    // Advance a logical clock at phase boundaries: no sleeps or scheduler-dependent assertions.
    fn late_search(overrun: bool, has_model: bool) -> (Value, usize) {
        let start = Instant::now();
        let deadline = start + Duration::from_secs(2);
        let budget = FitBudget::new(start, deadline);
        let clock = Cell::new(start);
        let calls = Cell::new(0);
        let result = fit_with(
            "(program (return (add (param.get p0) (param.get p1))))",
            &["0".into()], &["1".into()], &Resources::default(), "none", 75271729,
            deadline, "runtime", "reference", "bigint-body/v1",
            |_, _, limits| {
                calls.set(calls.get() + 1);
                assert!(limits.evaluation_seconds > 0.0);
                assert!(limits.evaluation_seconds <= 0.5);
                if overrun { clock.set(budget.production); }
                Ok(vec!["1".into()])
            },
            || clock.get(),
            |_, _, _, received, _, _| {
                assert_eq!(received, budget.search);
                clock.set(received);
                Ok(smt::Search {
                    assignments: if has_model { vec![vec![IBig::ZERO, IBig::ONE]] } else { vec![] },
                    outcome: "resource_exhausted".into(), uniqueness: "incomplete".into(),
                    reason: Some("deadline".into()), checks: 2,
                })
            },
        );
        (result, calls.get())
    }

    #[test]
    fn early_sat_survives_search_deadline_and_stays_provisional() {
        let (result, calls) = late_search(false, true);
        assert_eq!(calls, 1);
        assert_eq!(result["outcome"], "assignment_found");
        assert_eq!(result["assignments"].as_array().unwrap().len(), 1);
        assert_eq!(result["assignments"][0]["uniqueness"], "incomplete");
        assert_eq!(result["searches"][0]["outcome"], "resource_exhausted");
        assert_eq!(result["searches"][0]["reason"], "deadline");
        assert_eq!(result["admitted"], false);
        assert!(result["completed_bounds"].as_array().unwrap().is_empty());
    }

    #[test]
    fn late_production_cannot_supply_an_assignment() {
        let (result, calls) = late_search(true, true);
        assert_eq!(calls, 1);
        assert_eq!(result["outcome"], "resource_exhausted");
        assert!(result["assignments"].as_array().unwrap().is_empty());
    }

    #[test]
    fn unknown_without_an_earlier_model_never_runs_production() {
        let (result, calls) = late_search(false, false);
        assert_eq!(calls, 0);
        assert_eq!(result["outcome"], "resource_exhausted");
        assert!(result["assignments"].as_array().unwrap().is_empty());
        assert!(result["evidence"].is_null());
    }

    #[test]
    fn expired_request_does_no_search_or_production() {
        let start = Instant::now();
        let result = fit_with(
            "(program (return (param.get p0)))", &["0".into()], &["1".into()],
            &Resources::default(), "none", 0, start - Duration::from_millis(1),
            "runtime", "reference", "bigint-body/v1",
            |_, _, _| panic!("expired production"), || start,
            |_, _, _, _, _, _| panic!("expired search"),
        );
        assert_eq!(result["outcome"], "resource_exhausted");
        assert!(result["assignments"].as_array().unwrap().is_empty());
    }
}
