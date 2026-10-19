//! Conservative whole-program control dependence, including loop-carried locals.
use crate::language::{Expr, Program, Statement};
use std::collections::{BTreeMap, BTreeSet};
pub const PARAMETER: usize = usize::MAX;
type Dependencies = BTreeSet<usize>;
fn dependencies(e: &Expr, locals: &BTreeMap<String, Dependencies>) -> Dependencies {
    match e {
        Expr::Int(_) => BTreeSet::from([e as *const Expr as usize]),
        Expr::Param(_) => BTreeSet::from([PARAMETER]),
        Expr::Get(n) => locals.get(n).cloned().unwrap_or_default(),
        Expr::Call(_, args, _) => args.iter().flat_map(|a| dependencies(a, locals)).collect(),
        _ => BTreeSet::new(),
    }
}
fn assignments(body: &[Statement], locals: &mut BTreeMap<String, Dependencies>) {
    for s in body {
        match s {
            Statement::Set(n, e) => { let d = dependencies(e, locals); locals.entry(n.clone()).or_default().extend(d); }
            Statement::If(_, a, b) => { assignments(a, locals); assignments(b, locals); }
            Statement::Loop(_, _, body) => assignments(body, locals),
        }
    }
}
fn controls(e: &Expr, locals: &BTreeMap<String, Dependencies>, protected: &mut Dependencies) {
    if let Expr::Call(op, args, _) = e {
        if ["eq", "lt", "le", "to_control", "control.lt"].contains(&op.as_str()) { protected.extend(dependencies(e, locals)); }
        if op == "select" { protected.extend(dependencies(&args[0], locals)); }
        for a in args { controls(a, locals, protected); }
    }
}
fn statements(body: &[Statement], locals: &BTreeMap<String, Dependencies>, protected: &mut Dependencies) {
    for s in body {
        match s {
            Statement::Set(_, e) => controls(e, locals, protected),
            Statement::If(e, a, b) => { protected.extend(dependencies(e, locals)); controls(e, locals, protected); statements(a, locals, protected); statements(b, locals, protected); }
            Statement::Loop(_, e, body) => { protected.extend(dependencies(e, locals)); controls(e, locals, protected); statements(body, locals, protected); }
        }
    }
}
pub fn protected_literals(p: &Program) -> Dependencies {
    let mut locals = BTreeMap::new();
    loop { let old = locals.clone(); assignments(&p.body, &mut locals); if locals == old { break; } }
    let mut protected = BTreeSet::new();
    statements(&p.body, &locals, &mut protected);
    controls(&p.result, &locals, &mut protected);
    protected
}
pub fn validate(p: &Program) -> Result<(), String> {
    if protected_literals(p).contains(&PARAMETER) { Err("unsupported:parameter_control".into()) } else { Ok(()) }
}
