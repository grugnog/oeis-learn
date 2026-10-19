//! Explicit checked-256 add/sub stack translation. No wrapping/scalar promotion.
use crate::{language::{self,Expr,Program},value::{checked256_limbs,Resources}};
pub fn logical(source:&str,r:&Resources)->Result<Program,String>{
    if source.len()>r.source_bytes{return Err("invalid_source".into());}
    let words:Vec<_>=source.split_whitespace().collect();if words.len()>1024{return Err("invalid_source".into());}
    let mut stack:Vec<String>=Vec::new();let mut at=0;
    while at<words.len(){let op=words[at];at+=1;match op{
        "i256.const"|"i256.zero"=>{let n=if op=="i256.zero"{"0"}else{let n=*words.get(at).ok_or("invalid_source")?;at+=1;n};checked256_limbs(n)?;stack.push(format!("(int {n})"));},
        "i256.add"|"i256.sub"=>{let b=stack.pop().ok_or("invalid_source")?;let a=stack.pop().ok_or("invalid_source")?;stack.push(format!("({} {a} {b})",if op=="i256.add"{"add"}else{"sub"}));},
        _=>return Err("unsupported:retained_scalar_wrapping_bitwise_control_streaming".into()),
    }}
    if stack.len()!=1{return Err("invalid_source".into());}
    language::parse(&format!("(program (return {}))",stack[0]),r,true)
}
pub fn final_source(p:&Program)->Result<String,String>{
    fn emit(e:&Expr)->Result<String,String>{match e{
        Expr::Int(n)=>{checked256_limbs(n)?;Ok(format!("i256.const {n}"))},
        Expr::Call(op,args,_) if ["add","sub"].contains(&op.as_str())&&args.len()==2=>Ok(format!("{} {} i256.{op}",emit(&args[0])?,emit(&args[1])?)),
        _=>Err("unsupported:retained_translation".into()),
    }}
    if !p.body.is_empty()||!p.locals.is_empty(){return Err("unsupported:retained_translation".into());}emit(&p.result)
}
