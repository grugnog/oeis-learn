//! Bounded symbolic DAG. Every executed bigint expression remains a numeric guard.
use crate::{dependence, language::{Expr, Program, Statement, Type}, value::{bits, integer, Resources}};
use dashu_int::IBig;
use std::{collections::BTreeMap, time::Instant};
#[derive(Clone, Debug)]
pub enum Node { Constant(IBig), Parameter(u8), Add(usize,usize), Sub(usize,usize), Mul(usize,usize) }
#[derive(Clone, Debug)]
struct Big { id: usize, affine: Option<Vec<IBig>>, dependent: bool }
#[derive(Clone, Debug)]
enum Value { Big(Big), Control(u32), Bool(bool) }
#[derive(Debug)]
pub struct Problem { pub nodes: Vec<Node>, pub outputs: Vec<usize>, pub expected: Vec<IBig>, pub affine_rows: Option<Vec<Vec<IBig>>>, pub parameters: usize, pub checked_signed256: bool }
struct Builder<'a> { problem: Problem, r: &'a Resources, deadline: Instant, env: BTreeMap<String,Value>, index: IBig, loops: u32 }
impl Builder<'_> {
    fn tick(&self) -> Result<(),String> { if Instant::now()>=self.deadline {Err("deadline".into())} else {Ok(())} }
    fn node(&mut self, node: Node, affine: Option<Vec<IBig>>, dependent: bool)->Result<Big,String> {
        self.tick()?;
        if self.problem.nodes.len()>=100_000 {return Err("symbolic_node_limit".into());}
        if affine.as_ref().is_some_and(|xs|xs.iter().any(|x|bits(x)>131072)) {return Err("symbolic_integer_limit".into());}
        if matches!(node, Node::Mul(_, _)) && affine.is_none() { self.problem.affine_rows = None; }
        let id=self.problem.nodes.len(); self.problem.nodes.push(node); Ok(Big{id,affine,dependent})
    }
    fn constant(&mut self,n: IBig)->Result<Big,String> { if bits(&n)>self.r.intermediate_bits {return Err("integer_limit".into());} let mut a=vec![IBig::ZERO;self.problem.parameters+1]; a[0]=n.clone(); self.node(Node::Constant(n),Some(a),false) }
    fn concrete(&self,b: &Big)->Result<IBig,String> { if b.dependent {return Err("unsupported:parameter_control".into());} b.affine.as_ref().map(|a|a[0].clone()).ok_or("unsupported:parameter_control".into()) }
    fn expression(&mut self,e:&Expr)->Result<Value,String> {
        self.tick()?;
        Ok(match e {
            Expr::Index=>Value::Big(self.constant(self.index.clone())?),
            Expr::Int(n)=>Value::Big(self.constant(integer(n,self.r.literal_bits)?)?),
            Expr::Param(p)=>{let mut a=vec![IBig::ZERO;self.problem.parameters+1];a[*p as usize+1]=IBig::ONE;Value::Big(self.node(Node::Parameter(*p),Some(a),true)?)},
            Expr::Control(n)=>Value::Control(*n),Expr::Bool(b)=>Value::Bool(*b),Expr::Get(n)=>self.env[n].clone(),
            Expr::Call(op,args,_)=>{
                let xs=args.iter().map(|a|self.expression(a)).collect::<Result<Vec<_>,_>>()?;
                match (op.as_str(),xs.as_slice()) {
                    ("add"|"sub"|"mul",[Value::Big(a),Value::Big(b)])=>{
                        let affine: Option<Vec<IBig>>=match (&a.affine,&b.affine) {
                            (Some(x),Some(y))=>match op.as_str() {
                                "add"=>Some(x.iter().zip(y).map(|(a,b)|a+b).collect()),
                                "sub"=>Some(x.iter().zip(y).map(|(a,b)|a-b).collect()),
                                "mul" if !a.dependent=>Some(y.iter().map(|v|v*&x[0]).collect()),
                                "mul" if !b.dependent=>Some(x.iter().map(|v|v*&y[0]).collect()),
                                _=>None,
                            },_=>None,
                        };
                        if !a.dependent && !b.dependent && affine.as_ref().is_some_and(|v|bits(&v[0])>self.r.intermediate_bits) {return Err("integer_limit".into());}
                        let node=match op.as_str(){"add"=>Node::Add(a.id,b.id),"sub"=>Node::Sub(a.id,b.id),_=>Node::Mul(a.id,b.id)};
                        Value::Big(self.node(node,affine,a.dependent||b.dependent)?)
                    },
                    ("eq"|"lt"|"le",[Value::Big(a),Value::Big(b)])=>{let a=self.concrete(a)?;let b=self.concrete(b)?;Value::Bool(match op.as_str(){"eq"=>a==b,"lt"=>a<b,_=>a<=b})},
                    ("to_control",[Value::Big(a)])=>{let n=self.concrete(a)?.to_string().parse::<u32>().map_err(|_|"control_range")?;if n>self.r.control_max{return Err("control_range".into());}Value::Control(n)},
                    ("to_bigint",[Value::Control(n)])=>Value::Big(self.constant(IBig::from(*n))?),
                    ("increment",[Value::Control(n)]) if *n<self.r.control_max=>Value::Control(n+1),
                    ("control.lt",[Value::Control(a),Value::Control(b)])=>Value::Bool(a<b),
                    ("select",[Value::Bool(cond),a,b])=>if *cond{a.clone()}else{b.clone()},
                    _=>return Err("control_range".into()),
                }
            }
        })
    }
    fn statements(&mut self,body:&[Statement])->Result<(),String> {
        for s in body {self.tick()?;match s {
            Statement::Set(n,e)=>{let v=self.expression(e)?;self.env.insert(n.clone(),v);},
            Statement::If(e,a,b)=>{let Value::Bool(cond)=self.expression(e)? else {return Err("invalid_source".into());};self.statements(if cond{a}else{b})?;},
            Statement::Loop(n,e,body)=>{
                let Value::Control(count)=self.expression(e)? else {return Err("invalid_source".into());};
                if count>self.r.loop_iterations{return Err("loop_limit".into());}
                for i in 0..count {self.loops+=1;if self.loops>self.r.aggregate_loop_iterations{return Err("loop_limit".into());}self.env.insert(n.clone(),Value::Control(i));self.statements(body)?;}
                self.env.insert(n.clone(),Value::Control(count));
            }
        }} Ok(())
    }
}
impl Problem {
    pub fn build(p:&Program,indices:&[String],values:&[String],r:&Resources,deadline:Instant)->Result<Self,String> {
        r.validate()?;dependence::validate(p)?;
        if indices.is_empty()||indices.len()>20||indices.len()!=values.len(){return Err("invalid_prompt".into());}
        let exact=indices.iter().map(|n|integer(n,r.input_bits)).collect::<Result<Vec<_>,_>>()?;
        if !exact.windows(2).all(|w|w[0]<w[1]){return Err("invalid_prompt".into());}
        let expected=values.iter().map(|n|integer(n,r.input_bits)).collect::<Result<Vec<_>,_>>()?;
        let mut b=Builder{problem:Self{nodes:vec![],outputs:vec![],expected,affine_rows:Some(vec![]),parameters:p.parameters.len(),checked_signed256:false},r,deadline,env:BTreeMap::new(),index:IBig::ZERO,loops:0};
        for (i,index) in exact.into_iter().enumerate() {
            b.env.clear();b.index=index;b.loops=0;
            for (n,t) in &p.locals {let v=match t{Type::Big=>Value::Big(b.constant(IBig::ZERO)?),Type::Control=>Value::Control(0),Type::Bool=>Value::Bool(false)};b.env.insert(n.clone(),v);}
            b.statements(&p.body)?;
            let Value::Big(out)=b.expression(&p.result)? else{return Err("invalid_source".into());};
            b.problem.outputs.push(out.id);
            match (b.problem.affine_rows.as_mut(),out.affine) {
                (Some(rows),Some(a))=>{let mut row=a[1..].to_vec();row.push(&b.problem.expected[i]-&a[0]);rows.push(row);},
                _=>b.problem.affine_rows=None,
            }
        }
        Ok(b.problem)
    }
}
