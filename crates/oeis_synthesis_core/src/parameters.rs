//! First-occurrence literal abstraction. Equal literals never share an ID.
use crate::{dependence, language::{self, Expr, Program, Statement}, value::Resources};
use serde::Serialize;
use std::collections::BTreeSet;
#[derive(Debug, Clone, Serialize)]
pub struct Occurrence { pub ordinal: usize, pub parameter: String, pub original: String }
pub fn abstract_literals(p: &Program, r: &Resources) -> Result<(Program, Vec<Occurrence>), String> {
    // Explicit parameters and automatic abstraction are separate policies.
    if !p.parameters.is_empty() { return Err("unsupported:mixed_abstraction".into()); }
    let protected = dependence::protected_literals(p);
    let mut map = Vec::new(); let mut ordinal = 0;
    fn expr(e: &Expr, protected: &BTreeSet<usize>, map: &mut Vec<Occurrence>, ordinal: &mut usize) -> Expr {
        match e {
            Expr::Int(n) => {
                let at = *ordinal; *ordinal += 1;
                if !protected.contains(&(e as *const Expr as usize)) && map.len() < 8 {
                    let id = map.len() as u8;
                    map.push(Occurrence { ordinal: at, parameter: format!("p{id}"), original: n.clone() });
                    Expr::Param(id)
                } else { e.clone() }
            }
            Expr::Call(op, args, t) => Expr::Call(op.clone(), args.iter().map(|a| expr(a, protected, map, ordinal)).collect(), *t),
            _ => e.clone(),
        }
    }
    fn body(xs: &[Statement], protected: &BTreeSet<usize>, map: &mut Vec<Occurrence>, ordinal: &mut usize) -> Vec<Statement> {
        xs.iter().map(|s| match s {
            Statement::Set(n, e) => Statement::Set(n.clone(), expr(e, protected, map, ordinal)),
            Statement::If(e, a, b) => { let e = expr(e, protected, map, ordinal); let a = body(a, protected, map, ordinal); let b = body(b, protected, map, ordinal); Statement::If(e,a,b) }
            Statement::Loop(n, e, xs) => { let e = expr(e, protected, map, ordinal); Statement::Loop(n.clone(), e, body(xs, protected, map, ordinal)) }
        }).collect()
    }
    let mut out = p.clone(); out.body = body(&p.body, &protected, &mut map, &mut ordinal);
    out.result = expr(&p.result, &protected, &mut map, &mut ordinal);
    let source = language::canonical_source(&out);
    Ok((language::parse(&source, r, true)?, map))
}
