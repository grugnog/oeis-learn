//! Named, deliberately narrow checked signed-256 adapter for native synthesis.
//!
//! Input is the retained bare macro-body subset: i256.const/zero/add/sub.
//! Scalar/wrapping/bitwise/streaming/control translations are explicitly outside
//! this adapter. Existing legacy validators/evaluators remain unchanged. The
//! original checked preamble bytes implement every executed arithmetic operation.
use crate::{engine::create_fuel_engine, sandbox::Compiled};
use oeis_synthesis_core::{
    value::{checked256_limbs, integer, signed256_decimal},
    ExecutionBackend,
};

pub struct CheckedSigned256 {
    pub term_fuel: u64,
    pub aggregate_fuel: u64,
}
impl Default for CheckedSigned256 {
    fn default() -> Self {
        Self {
            term_fuel: 1_000_000,
            aggregate_fuel: 100_000_000,
        }
    }
}
impl ExecutionBackend for CheckedSigned256 {
    fn profile(&self) -> &str {
        "wat_i256_checked_v1"
    }
    fn execute(&mut self, source: &str, indices: &[String]) -> Result<Vec<String>, String> {
        if source.len() > 65536
            || indices.is_empty()
            || indices.len() > 100
            || self.term_fuel == 0
            || self.aggregate_fuel == 0
        {
            return Err("invalid_source".into());
        }
        let words = source.split_whitespace().collect::<Vec<_>>();
        if words.len() > 1024 {
            return Err("invalid_source".into());
        }
        let mut at = 0;
        let mut depth = 0usize;
        let mut body = String::new();
        while at < words.len() {
            let op = words[at];
            at += 1;
            match op {
                "i256.const"|"i256.zero"=>{
                    let text=if op=="i256.zero" {"0"} else {let text=*words.get(at).ok_or("invalid_source")?;at+=1;text};
                    for limb in checked256_limbs(text)? {body+=&format!("i64.const {limb}\n");}
                    depth+=1;
                },
                "i256.add"|"i256.sub"=>{
                    if depth<2 {return Err("invalid_source".into());}depth-=1;
                    body+=if op=="i256.add" {"call $i256_add\n"} else {"call $i256_sub\n"};
                },
                _=>return Err("unsupported:signed256_adapter/v1 excludes scalar/wrapping/bitwise/streaming/control translation".into()),
            }
        }
        if depth != 1 {
            return Err("invalid_source".into());
        }
        let preamble = include_str!("../../../src/oeis_learn/sandbox/preamble.wat");
        let start = preamble.find("(module").ok_or("runtime_failure")? + 7;
        let end = preamble.rfind(')').ok_or("runtime_failure")?;
        let module = format!(
            "(module {} (func (export \"compute\") (param i32) (result i64 i64 i64 i64) {body}))",
            &preamble[start..end]
        );
        let engine = create_fuel_engine().map_err(|_| "runtime_failure")?;
        let compiled = Compiled::new(&engine, &module, self.profile())?;
        let mut outputs = Vec::new();
        let mut remaining = self.aggregate_fuel;
        for text in indices {
            integer(text, 32)?;
            let index = text
                .parse::<i32>()
                .map_err(|_| "unsupported:retained index must fit i32")?;
            if remaining == 0 {
                return Err("aggregate_fuel".into());
            }
            let term = compiled.term(index, self.term_fuel.min(remaining));
            remaining = remaining.saturating_sub(term.fuel);
            if let Some(outcome) = term.outcome {
                return Err(outcome);
            }
            outputs.push(signed256_decimal(&term.values.ok_or("runtime_failure")?)?);
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_boundary_fixtures_and_explicit_unsupported() {
        let mut backend = CheckedSigned256::default();
        let hi = "57896044618658097711785492504343953926634992332820282019728792003956564819967";
        let lo = "-57896044618658097711785492504343953926634992332820282019728792003956564819968";
        for n in [hi, lo] {
            assert_eq!(
                backend
                    .execute(&format!("i256.const {n}"), &["-7".into(), "99".into()])
                    .unwrap(),
                vec![n, n]
            );
        }
        assert_eq!(
            backend
                .execute(
                    &format!("i256.const {hi} i256.const 1 i256.add"),
                    &["0".into()]
                )
                .unwrap_err(),
            "numeric_limit"
        );
        for op in ["i256.mul_scalar", "i64.add", "i64.xor", "stream", "call"] {
            assert!(backend
                .execute(op, &["0".into()])
                .unwrap_err()
                .starts_with("unsupported:"));
        }
        assert_eq!(
            backend.execute("i256.const -1", &["0".into()]).unwrap(),
            vec!["-1"]
        );
    }
}
