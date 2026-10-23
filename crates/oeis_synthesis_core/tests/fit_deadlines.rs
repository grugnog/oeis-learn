//! Deadline containment and exact-model enumeration, independent of neural runs.
use oeis_synthesis_core::{fit_budget::FitBudget, language, smt, symbolic::Problem, value::Resources};
use std::time::{Duration, Instant};

#[test]
fn nested_deadlines_never_extend_the_caller_or_native_ceiling() {
    let start = Instant::now();
    for allowance in [Duration::from_nanos(1), Duration::from_millis(40), Duration::from_secs(9)] {
        let caller = start + allowance;
        let budget = FitBudget::new(start, caller);
        assert!(start <= budget.search);
        assert!(budget.search <= budget.production);
        assert!(budget.production <= budget.response);
        assert!(budget.response <= caller);
        assert!(budget.response <= start + Duration::from_secs(2));
    }
    let budget = FitBudget::new(start, start + Duration::from_millis(100));
    assert_eq!(budget.search - start, Duration::from_millis(70));
    assert_eq!(budget.production - budget.search, Duration::from_millis(25));
    assert_eq!(budget.response - budget.production, Duration::from_millis(5));
    let expired = start - Duration::from_millis(1);
    let budget = FitBudget::new(start, expired);
    assert_eq!(budget.search, expired);
    assert_eq!(budget.production, expired);
    assert_eq!(budget.response, expired);
}

fn search(source: &str, value: &str, cap: usize) -> smt::Search {
    let r = Resources::default();
    let deadline = Instant::now() + Duration::from_secs(2);
    let program = language::parse(source, &r, true).unwrap();
    let problem = Problem::build(&program, &["0".into()], &[value.into()], &r, deadline).unwrap();
    smt::search(&problem, &r, 16, deadline, 75271729, cap).unwrap()
}

#[test]
fn eight_assignments_prove_ambiguity_without_a_ninth_check() {
    let found = search("(program (return (add (param.get p0) (param.get p1))))", "1", 8);
    assert_eq!(found.assignments.len(), 8);
    assert_eq!(found.checks, 8);
    assert_eq!(found.outcome, "assignment_found");
    assert_eq!(found.uniqueness, "multiple");
    for (i, xs) in found.assignments.iter().enumerate() {
        assert_eq!((&xs[0] + &xs[1]).to_string(), "1");
        assert!(!found.assignments[..i].contains(xs));
    }
}

#[test]
fn one_model_still_requires_a_blocking_check() {
    let ambiguous = search("(program (return (add (param.get p0) (param.get p1))))", "1", 1);
    assert_eq!(ambiguous.assignments.len(), 1);
    assert_eq!(ambiguous.checks, 2);
    assert_eq!(ambiguous.uniqueness, "multiple");
    let unique = search("(program (return (param.get p0)))", "4", 1);
    assert_eq!(unique.assignments.len(), 1);
    assert_eq!(unique.checks, 2);
    assert_eq!(unique.uniqueness, "unique_in_scope");
}
