//! Fixed-budget SMT checker for independently derived loop-certificate obligations.
//! No constant fitting, arbitrary command execution, model, or proof promotion API.
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use z3::{Params, SatResult, Solver};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    declarations: Vec<String>,
    assertion: String,
}
#[derive(Serialize)]
struct Response {
    result: &'static str,
    solver: String,
}
fn main() {
    let mut data = Vec::new();
    io::stdin().take(1_048_577).read_to_end(&mut data).unwrap();
    assert!(data.len() <= 1_048_576);
    let request: Request = serde_json::from_slice(&data).unwrap();
    assert!(request.declarations.len() <= 33 && request.assertion.len() <= 1_000_000);
    let mut text = String::new();
    for name in &request.declarations {
        assert!(name == "n" || (0..32).any(|n| name == &format!("s{n}")));
        text.push_str(&format!("(declare-const {name} Int)\n"));
    }
    text.push_str(&format!("(assert {})", request.assertion));
    let solver = Solver::new();
    let mut params = Params::new();
    params.set_u32("timeout", 1500);
    solver.set_params(&params);
    solver.from_string(text);
    let result = match solver.check() {
        SatResult::Sat => "sat",
        SatResult::Unsat => "unsat",
        SatResult::Unknown => "unknown",
    };
    let response = Response {
        result,
        solver: "z3-rust/0.21.1;z3-src/501.0.1".into(),
    };
    println!("{}", serde_json::to_string(&response).unwrap());
}
