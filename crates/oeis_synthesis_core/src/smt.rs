//! Worker-local exact QF_LIA/QF_NIA; UNKNOWN never yields a model.
use crate::{symbolic::{Node,Problem},value::{integer,Resources}};
use dashu_int::IBig;
use std::time::Instant;
use z3::{ast::{Bool,Int},Config,Params,SatResult,Solver};
#[derive(Debug)]
pub struct Search {pub assignments:Vec<Vec<IBig>>,pub outcome:String,pub uniqueness:String,pub reason:Option<String>}
fn numeral(n:&IBig)->Int {n.to_string().parse().expect("canonical integer")}
fn exact_model(n:&Int)->Result<IBig,String> {
    let text=n.to_string();
    let text=if let Some(s)=text.strip_prefix("(- ").and_then(|s|s.strip_suffix(')')) {format!("-{s}")}else{text};
    integer(&text,4096)
}
pub fn search(p:&Problem,r:&Resources,bound:usize,deadline:Instant,seed:u32,cap:usize)->Result<Search,String> {
    if ![16,64,256,1024,4096].contains(&bound)||cap==0||cap>8{return Err("invalid_scope".into());}
    let mut config=Config::new();config.set_model_generation(true);
    z3::with_z3_config(&config,||{
        let logic=if p.affine_rows.is_some(){"QF_LIA"}else{"QF_NIA"};
        let solver=Solver::new_for_logic(logic).ok_or("unsupported_logic")?;
        let vars:Vec<Int>=(0..p.parameters).map(|i|Int::new_const(format!("p{i}"))).collect();
        let max=(IBig::ONE<<bound)-IBig::ONE;
        for x in &vars {solver.assert(x.le(numeral(&max)));solver.assert(x.ge(numeral(&(-&max))));}
        let max=(IBig::ONE<<r.intermediate_bits)-IBig::ONE;
        let hi=numeral(&max);let lo=numeral(&(-&max));let mut exprs:Vec<Int>=Vec::new();
        for node in &p.nodes {
            if Instant::now()>=deadline{return Err("deadline".into());}
            let value=match node {
                Node::Constant(n)=>numeral(n),Node::Parameter(i)=>vars[*i as usize].clone(),
                Node::Add(a,b)=>Int::add(&[&exprs[*a],&exprs[*b]]),
                Node::Sub(a,b)=>Int::sub(&[&exprs[*a],&exprs[*b]]),
                Node::Mul(a,b)=>Int::mul(&[&exprs[*a],&exprs[*b]]),
            };
            // Guard all intermediate expressions, including cancelled/eager values.
            solver.assert(value.le(&hi));solver.assert(value.ge(&lo));exprs.push(value);
        }
        for (out,expected) in p.outputs.iter().zip(&p.expected){solver.assert(exprs[*out].eq(numeral(expected)));}
        let mut result=Search{assignments:vec![],outcome:"unknown".into(),uniqueness:"unchecked".into(),reason:None};
        loop {
            let remaining=deadline.saturating_duration_since(Instant::now()).as_millis();
            if remaining==0 {result.outcome="resource_exhausted".into();result.uniqueness=if result.assignments.len()>1{"multiple"}else{"incomplete"}.into();result.reason=Some("deadline".into());break;}
            let mut params=Params::new();params.set_u32("timeout",remaining.min(2000) as u32);params.set_u32("random_seed",seed);solver.set_params(&params);
            match solver.check() {
                SatResult::Unsat=>{result.outcome=if result.assignments.is_empty(){"unsat_in_scope"}else{"assignment_found"}.into();result.uniqueness=if result.assignments.len()==1{"unique_in_scope"}else if result.assignments.len()>1{"multiple"}else{"unchecked"}.into();break;},
                SatResult::Unknown=>{result.reason=solver.get_reason_unknown();result.outcome=if Instant::now()>=deadline{"resource_exhausted"}else{"unknown"}.into();result.uniqueness=if result.assignments.len()>1{"multiple"}else{"incomplete"}.into();break;},
                SatResult::Sat=>{
                    let model=solver.get_model().ok_or("missing_model")?;
                    let values=vars.iter().map(|x|exact_model(&model.eval(x,true).ok_or("missing_numeral")?)).collect::<Result<Vec<_>,String>>()?;
                    if result.assignments.len()==cap {result.uniqueness="multiple".into();result.outcome="assignment_found".into();break;}
                    let blocking:Vec<Bool>=vars.iter().zip(&values).map(|(x,n)|x.eq(numeral(n)).not()).collect();
                    result.assignments.push(values);solver.assert(Bool::or(&blocking));
                }
            }
        }
        Ok(result)
    })
}
