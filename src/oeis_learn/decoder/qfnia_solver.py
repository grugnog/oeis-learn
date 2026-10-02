"""Bounded mathematical-integer solving with truthful outcomes."""
from dataclasses import dataclass
import time
import z3


@dataclass(frozen=True)
class IntegerSolution:
    outcome: str
    constants: tuple[int, ...] = ()
    elapsed_ms: float = 0
    reason: str | None = None
    method: str = "z3_qfnia"


def solve_integer_constraints(unknown_count, constraints_builder, bound=1000, timeout_ms=240):
    if type(unknown_count) is not int or not 0 <= unknown_count <= 8:
        raise ValueError("parameter count must be in 0..8")
    if type(bound) is not int or not 0 <= bound <= 1000:
        raise ValueError("coefficient bound must be in 0..1000")
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 2000:
        raise ValueError("solver timeout must be in 1..2000 ms")
    started = time.monotonic()
    variables = [z3.Int(f"c_{i}") for i in range(unknown_count)]
    solver = z3.SolverFor("QF_NIA")
    solver.set(timeout=timeout_ms)
    solver.add(*[condition for c in variables for condition in (c >= -bound, c <= bound)])
    solver.add(*constraints_builder(variables))
    status = solver.check()
    elapsed = (time.monotonic() - started) * 1000
    if status == z3.sat:
        return IntegerSolution("verified_solution", tuple(solver.model().eval(c, model_completion=True).as_long() for c in variables), elapsed)
    if status == z3.unsat:
        return IntegerSolution("proved_unsat_in_scope", elapsed_ms=elapsed)
    reason = solver.reason_unknown()
    return IntegerSolution("timeout" if "timeout" in reason.lower() else "unknown", elapsed_ms=elapsed, reason=reason)


def solve_qfnia_tactical(unknown_count, constraints_builder, bound=1000, timeout_ms=240):
    result = solve_integer_constraints(unknown_count, constraints_builder, bound, timeout_ms)
    return result.outcome == "verified_solution", list(result.constants), result.elapsed_ms
