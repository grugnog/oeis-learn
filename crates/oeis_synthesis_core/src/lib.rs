//! Native synthesis algorithms with disabled production routing; no Python callbacks.
pub mod protocol;
pub mod value;
pub mod language;
pub mod lowering;
pub mod runtime;

/// Binding supplies retained-profile execution; bigint backend belongs to core.
/// Exact values cross this boundary as strings, never machine integers.
pub trait ExecutionBackend {
    fn profile(&self) -> &str;
    fn execute(&mut self, source: &str, indices: &[String]) -> Result<Vec<String>, String>;
}

pub const PRODUCTION_ENABLED: bool = false;

/// Dependency/FFI diagnostic only, not a candidate fitter or admission path.
pub fn dependency_smoke() -> String {
    use z3::{ast::Int, SatResult, Solver};
    let exact = "340282366920938463463374607431768211457";
    let x = Int::new_const("dependency_smoke");
    let numeral: Int = exact.parse().expect("fixed exact numeral");
    let solver = Solver::new();
    solver.assert(x.eq(&numeral));
    assert_eq!(solver.check(), SatResult::Sat);
    let answer = solver
        .get_model()
        .unwrap()
        .eval(&x, true)
        .unwrap()
        .to_string();
    assert_eq!(answer, exact);
    answer
}

pub mod dependence;
pub mod parameters;
pub mod symbolic;
pub mod affine;
pub mod smt;
pub mod fit;
pub mod substitute;
pub mod evidence;
pub mod cache;
pub mod diagnostic;
pub mod retained;

pub mod transport;
