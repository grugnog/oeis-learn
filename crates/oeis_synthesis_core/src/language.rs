//! Bounded typed logical source. This parser never accepts arbitrary WAT.
use crate::value::{integer, Resources};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Type { Big, Control, Bool }
#[derive(Clone,Debug,PartialEq,Eq)]
pub enum Expr {
    Index, Int(String), Control(u32), Param(u8), Get(String), Bool(bool),
    Call(String,Vec<Expr>,Type),
}
impl Expr {
    pub fn kind(&self,locals:&BTreeMap<String,Type>)->Type {
        match self {
            Self::Index|Self::Int(_)|Self::Param(_)=>Type::Big,
            Self::Control(_)=>Type::Control, Self::Bool(_)=>Type::Bool,
            Self::Get(s)=>locals[s], Self::Call(_,_,t)=>*t,
        }
    }
}
#[derive(Clone,Debug,PartialEq,Eq)]
pub enum Statement { Set(String,Expr), If(Expr,Vec<Statement>,Vec<Statement>), Loop(String,Expr,Vec<Statement>) }
#[derive(Clone,Debug)]
pub struct Program {
    pub locals:BTreeMap<String,Type>, pub body:Vec<Statement>, pub result:Expr,
    pub parameters:Vec<u8>, pub source:String,
}
#[derive(Clone,Debug)]
enum S { Atom(String), List(Vec<S>) }
impl S {
    fn atom(&self)->Result<&str,String> { match self { Self::Atom(s)=>Ok(s),_=>Err("invalid_source".into()) } }
    fn list(&self)->Result<&[S],String> { match self { Self::List(s)=>Ok(s),_=>Err("invalid_source".into()) } }
}
fn read(tokens:&[&str],pos:&mut usize,depth:usize,nodes:&mut usize,r:&Resources)->Result<S,String> {
    *nodes+=1;
    if *nodes>r.ast_nodes || depth>r.nesting { return Err("invalid_source".into()); }
    let t=*tokens.get(*pos).ok_or("invalid_source")?; *pos+=1;
    if t==")" { return Err("invalid_source".into()); }
    if t!="(" { return Ok(S::Atom(t.into())); }
    let mut items=Vec::new();
    while tokens.get(*pos)!=Some(&")") {
        items.push(read(tokens,pos,depth+1,nodes,r)?);
    }
    *pos+=1;
    if items.is_empty() { return Err("invalid_source".into()); }
    Ok(S::List(items))
}
struct Checker<'a> { r:&'a Resources, locals:BTreeMap<String,Type>, params:Vec<u8>, allow_params:bool }
impl Checker<'_> {
    fn expression(&mut self,node:&S)->Result<Expr,String> {
        if matches!(node,S::Atom(s) if s=="$index") { return Ok(Expr::Index); }
        let xs=node.list()?; let op=xs[0].atom()?; let args=&xs[1..];
        if op=="int" && args.len()==1 {
            let s=args[0].atom()?; integer(s,self.r.literal_bits)?;
            return Ok(Expr::Int(s.into()));
        }
        if op=="control" && args.len()==1 {
            let s=args[0].atom()?; integer(s,31)?;
            let n=s.parse::<u32>().map_err(|_| "invalid_source")?;
            if n>self.r.control_max { return Err("invalid_source".into()); }
            return Ok(Expr::Control(n));
        }
        if op=="param.get" && args.len()==1 && self.allow_params {
            let s=args[0].atom()?;
            if s.len()!=2 || !s.starts_with('p') || !(b'0'..=b'7').contains(&s.as_bytes()[1]) { return Err("invalid_source".into()); }
            let p=s.as_bytes()[1]-b'0';
            if !self.params.contains(&p) { self.params.push(p); }
            return Ok(Expr::Param(p));
        }
        if op=="get" && args.len()==1 {
            let name=args[0].atom()?;
            if !self.locals.contains_key(name) { return Err("invalid_source".into()); }
            return Ok(Expr::Get(name.into()));
        }
        if (op=="true" || op=="false") && args.is_empty() { return Ok(Expr::Bool(op=="true")); }
        let expressions=args.iter().map(|a|self.expression(a)).collect::<Result<Vec<_>,_>>()?;
        let ts=expressions.iter().map(|e|e.kind(&self.locals)).collect::<Vec<_>>();
        use Type::*;
        let out=match (op,ts.as_slice()) {
            ("add"|"sub"|"mul",[Big,Big])=>Big,
            ("eq"|"lt"|"le",[Big,Big])=>Bool,
            ("to_control",[Big])=>Control,("to_bigint",[Control])=>Big,
            ("increment",[Control])=>Control,("control.lt",[Control,Control])=>Bool,
            ("select",[Bool,a,b]) if a==b=>*a,
            _=>return Err("invalid_source".into()),
        };
        Ok(Expr::Call(op.into(),expressions,out))
    }
    fn statements(&mut self,nodes:&[S],counters:&BTreeSet<String>)->Result<Vec<Statement>,String> {
        let mut out=Vec::new();
        for node in nodes {
            let xs=node.list()?; let op=xs[0].atom()?; let args=&xs[1..];
            out.push(match op {
                "set" if args.len()==2=>{
                    let name=args[0].atom()?; let e=self.expression(&args[1])?;
                    if counters.contains(name) || self.locals.get(name)!=Some(&e.kind(&self.locals)) { return Err("invalid_source".into()); }
                    Statement::Set(name.into(),e)
                },
                "if" if args.len()==3=>{
                    let cond=self.expression(&args[0])?;
                    let a=args[1].list()?; let b=args[2].list()?;
                    if cond.kind(&self.locals)!=Type::Bool || a[0].atom()?!="then" || b[0].atom()?!="else" { return Err("invalid_source".into()); }
                    Statement::If(cond,self.statements(&a[1..],counters)?,self.statements(&b[1..],counters)?)
                },
                "loop" if args.len()>=3=>{
                    let name=args[0].atom()?; let bound=self.expression(&args[1])?;
                    if self.locals.get(name)!=Some(&Type::Control) || counters.contains(name) || bound.kind(&self.locals)!=Type::Control { return Err("invalid_source".into()); }
                    let mut nested=counters.clone(); nested.insert(name.into());
                    Statement::Loop(name.into(),bound,self.statements(&args[2..],&nested)?)
                },
                _=>return Err("invalid_source".into()),
            });
        }
        Ok(out)
    }
}
pub fn parse(source:&str,r:&Resources,allow_parameters:bool)->Result<Program,String> {
    r.validate()?;
    if source.len()>r.source_bytes || !source.is_ascii() { return Err("invalid_source".into()); }
    let expanded=source.replace('('," ( ").replace(')'," ) ");
    let tokens=expanded.split_whitespace().collect::<Vec<_>>();
    let (mut pos,mut nodes)=(0,0);
    let tree=read(&tokens,&mut pos,0,&mut nodes,r)?;
    let xs=tree.list()?;
    if pos!=tokens.len() || xs[0].atom()?!="program" { return Err("invalid_source".into()); }
    let mut c=Checker{r,locals:BTreeMap::new(),params:Vec::new(),allow_params:allow_parameters};
    let mut at=1;
    while at<xs.len() {
        let item=xs[at].list()?;
        if item[0].atom()?!="local" { break; }
        if item.len()!=3 { return Err("invalid_source".into()); }
        let name=item[1].atom()?;
        let (typ,prefix,cap)=match item[2].atom()? {
            "bigint"=>(Type::Big,"$b",r.bigint_locals),"control"=>(Type::Control,"$c",r.control_locals),
            "bool"=>(Type::Bool,"$q",r.bool_locals),_=>return Err("invalid_source".into()),
        };
        if !(0..cap).any(|i|name==format!("{prefix}{i}")) || c.locals.insert(name.into(),typ).is_some() { return Err("invalid_source".into()); }
        at+=1;
    }
    if at>=xs.len() { return Err("invalid_source".into()); }
    let last=xs.last().unwrap().list()?;
    if last.len()!=2 || last[0].atom()?!="return" { return Err("invalid_source".into()); }
    let body=c.statements(&xs[at..xs.len()-1],&BTreeSet::new())?;
    let result=c.expression(&last[1])?;
    if result.kind(&c.locals)!=Type::Big { return Err("invalid_source".into()); }
    Ok(Program{locals:c.locals,body,result,parameters:c.params,source:source.into()})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn typed_and_complete() {
        let r=Resources::default();
        for source in ["(program (return (control 1)))","(program (return (get $b0)))","(program (return (int -0)))","(program (return $index)) junk","(program (return (param.get p0)))"] { assert!(parse(source,&r,false).is_err(),"{source}"); }
        assert!(parse("(program (return (add $index (int -7))))",&r,false).is_ok());
        let p=parse("(program (return (mul (param.get p7) (param.get p7))))",&r,true).unwrap();
        assert_eq!(p.parameters,vec![7]);
    }
}
