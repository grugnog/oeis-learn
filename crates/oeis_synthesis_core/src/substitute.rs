//! Typed substitution followed by complete source validation, never token repair.
use crate::{language::{self, Expr, Program, Statement}, value::{integer, Resources}};
use dashu_int::IBig;
pub fn substitute(p:&Program,values:&[IBig],r:&Resources)->Result<String,String> {
    if values.len()!=p.parameters.len(){return Err("invalid_assignment".into());}
    for n in values {integer(&n.to_string(),r.literal_bits.min(4096))?;}
    fn expr(e:&Expr,values:&[IBig])->Expr{match e{Expr::Param(p)=>Expr::Int(values[*p as usize].to_string()),Expr::Call(op,args,t)=>Expr::Call(op.clone(),args.iter().map(|a|expr(a,values)).collect(),*t),_=>e.clone()}}
    fn body(xs:&[Statement],values:&[IBig])->Vec<Statement>{xs.iter().map(|s|match s{Statement::Set(n,e)=>Statement::Set(n.clone(),expr(e,values)),Statement::If(e,a,b)=>Statement::If(expr(e,values),body(a,values),body(b,values)),Statement::Loop(n,e,xs)=>Statement::Loop(n.clone(),expr(e,values),body(xs,values))}).collect()}
    let mut out=p.clone();out.body=body(&p.body,values);out.result=expr(&p.result,values);
    Ok(language::parse(&language::canonical_source(&out),r,false)?.source)
}
