//! Independent exact-fitting counterexamples; no neural model or Python search.
use dashu_int::IBig;
use oeis_synthesis_core::{affine::{solve, Solution}, language::parse, parameters::abstract_literals,
    symbolic::Problem, value::Resources};
use std::time::{Duration, Instant};
fn rows(a: &[&[i64]]) -> Vec<Vec<IBig>> { a.iter().map(|r| r.iter().map(|x| IBig::from(*x)).collect()).collect() }
#[test]
fn exact_elimination_and_integer_feasibility() {
    let deadline = Instant::now() + Duration::from_secs(2);
    assert_eq!(solve(&rows(&[&[2, 1], &[4, 2]]), 1, deadline).unwrap(), Solution::NonIntegral);
    assert_eq!(solve(&rows(&[&[1, 2], &[1, 3]]), 1, deadline).unwrap(), Solution::Inconsistent);
    assert_eq!(solve(&rows(&[&[2, 3, 1]]), 2, deadline).unwrap(), Solution::RankDeficient);
    let big = -(IBig::ONE << 512usize);
    assert_eq!(solve(&vec![vec![IBig::ONE, big.clone()]], 1, deadline).unwrap(), Solution::Unique(vec![big]));
}
#[test]
fn abstraction_never_merges_equal_occurrences_or_control_literals() {
    let r = Resources::default();
    let p = parse("(program (local $b0 bigint) (set $b0 (int 7)) (if (lt $index (get $b0)) (then) (else)) (return (add (int 7) (int 7))))", &r, true).unwrap();
    let (p, map) = abstract_literals(&p, &r).unwrap();
    assert_eq!(map.len(), 2);
    assert!(p.source.contains("(set $b0 (int 7))"));
    assert!(p.source.contains("(add (param.get p0) (param.get p1))"));
}
#[test]
fn tainted_control_is_semantic_and_intermediate_guards_survive() {
    let r = Resources::default();
    let p = parse("(program (return (select (lt (param.get p0) (int 0)) (int 1) (int 1))))", &r, true).unwrap();
    assert!(Problem::build(&p, &["0".into()], &["1".into()], &r, Instant::now()+Duration::from_secs(2)).unwrap_err().contains("parameter_control"));
    let p = parse("(program (return (sub (mul (param.get p0) (param.get p0)) (mul (param.get p0) (param.get p0)))))", &r, true).unwrap();
    let problem = Problem::build(&p, &["0".into()], &["0".into()], &r, Instant::now()+Duration::from_secs(2)).unwrap();
    assert!(problem.nodes.len() >= 7);
    assert!(problem.affine_rows.is_none());
}
