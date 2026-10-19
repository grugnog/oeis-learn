//! Replay derives the problem anew; attached SMT text is never trusted or executed.
use crate::{language,parameters,smt,symbolic::{Node,Problem},value::{digest,Resources}};
use serde::{Serialize,Deserialize};
use std::time::Instant;
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Evidence {pub version:String,pub source_sha256:String,pub prompt_sha256:String,pub resource_sha256:String,pub constraints_sha256:String,pub coefficient_bits:usize,pub abstraction_policy:String,pub solver:String,pub language_profile:String}
pub fn prompt_identity(indices:&[String],values:&[String])->String{digest(serde_json::to_string(&serde_json::json!({"indices":indices,"values":values})).unwrap().as_bytes())}
pub fn constraints(p:&Problem)->String {
    let nodes:Vec<_>=p.nodes.iter().map(|n|match n{Node::Constant(v)=>serde_json::json!(["int",v.to_string()]),Node::Parameter(i)=>serde_json::json!(["param",i]),Node::Add(a,b)=>serde_json::json!(["add",a,b]),Node::Sub(a,b)=>serde_json::json!(["sub",a,b]),Node::Mul(a,b)=>serde_json::json!(["mul",a,b])}).collect();
    digest(serde_json::to_string(&serde_json::json!({"nodes":nodes,"outputs":p.outputs,"expected":p.expected.iter().map(ToString::to_string).collect::<Vec<_>>(),"parameters":p.parameters,"guards":if p.checked_signed256{"all-intermediate-checked256/v1"}else{"all-intermediate-magnitude/v1"}})).unwrap().as_bytes())
}
pub fn record(source:&str,indices:&[String],values:&[String],r:&Resources,p:&Problem,bound:usize,policy:&str)->Evidence {
    Evidence{version:"scoped-unsat/v1".into(),source_sha256:digest(source.as_bytes()),prompt_sha256:prompt_identity(indices,values),resource_sha256:r.identity(),constraints_sha256:constraints(p),coefficient_bits:bound,abstraction_policy:policy.into(),solver:"z3-0.21.1/native-fitting-v1".into(),language_profile:if p.checked_signed256{"wat_i256_checked_v1"}else{"bigint-body/v1"}.into()}
}
pub fn replay(source:&str,indices:&[String],values:&[String],r:&Resources,e:&Evidence,deadline:Instant)->Result<String,String> {
    let mut program=match e.language_profile.as_str(){"bigint-body/v1"=>language::parse(source,r,true)?,"wat_i256_checked_v1"=>crate::retained::logical(source,r)?,_=>return Err("verification_failure".into())};
    match e.abstraction_policy.as_str(){"none"=>(),"eligible-literals/v1"=>program=parameters::abstract_literals(&program,r)?.0,_=>return Err("verification_failure".into())}
    let mut p=Problem::build(&program,indices,values,r,deadline)?;p.checked_signed256=e.language_profile=="wat_i256_checked_v1";
    if *e!=record(source,indices,values,r,&p,e.coefficient_bits,&e.abstraction_policy){return Err("verification_failure".into());}
    let result=smt::search(&p,r,e.coefficient_bits,deadline,0,1)?;
    Ok(if result.outcome=="unsat_in_scope"{"replayed_unsat".into()}else{result.outcome})
}
