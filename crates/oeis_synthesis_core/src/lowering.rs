//! Typed AST -> closed candidate module; only the seven approved runtime imports.
use crate::language::{Expr,Program,Statement,Type};
use crate::value::Resources;

pub const ABI: &[&str]=&["from_i64","constant","add","sub","mul","cmp","to_control"];
pub struct Lowered { pub wasm:Vec<u8>, pub literals:Vec<String> }
struct Builder<'a> { r:&'a Resources, literals:Vec<String>, temps:usize }
impl Builder<'_> {
    fn temp(&mut self)->String { let n=self.temps;self.temps+=1;format!("$__t{n}") }
    fn expr(&mut self,e:&Expr)->Result<String,String> {
        Ok(match e {
            Expr::Index=>"local.get $index\n".into(),
            Expr::Int(s)=>{let n=self.literals.len();self.literals.push(s.clone());format!("i32.const {n}\ncall $constant\n")},
            Expr::Control(n)=>format!("i32.const {n}\n"),Expr::Bool(b)=>format!("i32.const {}\n",u8::from(*b)),
            Expr::Get(s)=>format!("local.get {s}\n"),Expr::Param(_)=>return Err("invalid_source".into()),
            Expr::Call(op,args,_)=>{
                let mut code=String::new();
                if op=="select" {
                    let c=self.temp();code+=&self.expr(&args[0])?;code+=&format!("local.set {c}\n");
                    code+=&self.expr(&args[1])?;code+=&self.expr(&args[2])?;
                    code+=&format!("local.get {c}\nselect\n");
                } else {
                    for a in args {code+=&self.expr(a)?;}
                    code+=&match op.as_str() {
                        "add"|"sub"|"mul"=>format!("call ${op}\n"),
                        "eq"=>"call $cmp\ni32.eqz\n".into(),
                        "lt"=>"call $cmp\ni32.const 0\ni32.lt_s\n".into(),
                        "le"=>"call $cmp\ni32.const 0\ni32.le_s\n".into(),
                        "to_control"=>"call $to_control\n".into(),
                        "to_bigint"=>"i64.extend_i32_u\ncall $from_i64\n".into(),
                        "control.lt"=>"i32.lt_u\n".into(),
                        "increment"=>{
                            let t=self.temp();format!("local.tee {t}\ni32.const {}\ni32.ge_u\nif\ni32.const -6\nreturn\nend\nlocal.get {t}\ni32.const 1\ni32.add\n",self.r.control_max)
                        },
                        _=>return Err("invalid_source".into()),
                    };
                }
                code
            }
        })
    }
    fn body(&mut self,statements:&[Statement])->Result<String,String> {
        let mut s=String::new();
        for stmt in statements {
            match stmt {
                Statement::Set(name,e)=>{s+=&self.expr(e)?;s+=&format!("local.set {name}\n");},
                Statement::If(e,a,b)=>{s+=&self.expr(e)?;s+="if\n";s+=&self.body(a)?;s+="else\n";s+=&self.body(b)?;s+="end\n";},
                Statement::Loop(name,e,body)=>{
                    let bound=self.temp();let label=bound.replace("$__t","__loop");
                    s+=&self.expr(e)?;
                    s+=&format!("local.tee {bound}\ni32.const {}\ni32.gt_u\nif\ni32.const -7\nreturn\nend\ni32.const 0\nlocal.set {name}\nblock ${label}_end\nloop ${label}\nlocal.get {name}\nlocal.get {bound}\ni32.ge_u\nbr_if ${label}_end\nlocal.get $__iterations\ni32.const 1\ni32.add\nlocal.tee $__iterations\ni32.const {}\ni32.gt_u\nif\ni32.const -7\nreturn\nend\n",self.r.loop_iterations,self.r.aggregate_loop_iterations);
                    s+=&self.body(body)?;
                    s+=&format!("local.get {name}\ni32.const 1\ni32.add\nlocal.set {name}\nbr ${label}\nend\nend\n");
                }
            }
        }
        Ok(s)
    }
}
pub fn lower(p:&Program,r:&Resources)->Result<Lowered,String> {
    if !p.parameters.is_empty() {return Err("invalid_source".into());}
    let mut b=Builder{r,literals:Vec::new(),temps:0};
    let mut body=String::new();
    for (name,t) in &p.locals { if *t==Type::Big { body+=&format!("i64.const 0\ncall $from_i64\nlocal.set {name}\n"); } }
    body+=&b.body(&p.body)?;body+=&b.expr(&p.result)?;
    let mut wat=String::from("(module\n");
    for op in ABI {
        let params=match *op {"from_i64"=>"i64","constant"|"to_control"=>"i32",_=>"i32 i32"};
        wat+=&format!("(import \"oeis.big.v1\" \"{op}\" (func ${op} (param {params}) (result i32)))\n");
    }
    wat+="(func (export \"compute\") (param $index i32) (result i32)\n(local $__iterations i32)\n";
    for name in p.locals.keys() {wat+=&format!("(local {name} i32)\n");}
    for i in 0..b.temps {wat+=&format!("(local $__t{i} i32)\n");}
    wat+=&body;wat+="))";
    let wasm=wat::parse_str(&wat).map_err(|_|"runtime_failure".to_string())?;
    validate_candidate(&wasm)?;
    Ok(Lowered{wasm,literals:b.literals})
}
/// Defense in depth: no candidate-owned memory/table/global/start/data/element.
/// Only internally lowered modules reach the adapter, never arbitrary bytes.
pub fn validate_candidate(bytes:&[u8])->Result<(),String> {
    use wasmparser::{Parser,Payload,TypeRef,ExternalKind};
    let mut imported=Vec::new();let mut exports=0;let mut functions=0;
    for payload in Parser::new(0).parse_all(bytes) {
        match payload.map_err(|_|"invalid_source")? {
            Payload::ImportSection(s)=>for i in s {let i=i.map_err(|_|"invalid_source")?;
                if i.module!="oeis.big.v1" || !ABI.contains(&i.name) || !matches!(i.ty,TypeRef::Func(_)) || imported.contains(&i.name) {return Err("invalid_source".into());}
                imported.push(i.name);
            },
            Payload::ExportSection(s)=>for e in s {let e=e.map_err(|_|"invalid_source")?;
                if e.name!="compute" || e.kind!=ExternalKind::Func || e.index!=7 {return Err("invalid_source".into());} exports+=1;
            },
            Payload::FunctionSection(s)=>functions=s.count(),
            Payload::MemorySection(_)|Payload::TableSection(_)|Payload::GlobalSection(_)|Payload::StartSection{..}
            |Payload::DataSection(_)|Payload::ElementSection(_)|Payload::TagSection(_)=>return Err("invalid_source".into()),
            _=>{},
        }
    }
    if imported.len()!=ABI.len() || exports!=1 || functions!=1 {return Err("invalid_source".into());}
    Ok(())
}
