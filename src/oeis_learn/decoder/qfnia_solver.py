"""Tactical Z3 QF_NIA Non-Linear Integer Constraint Solver.

Models constants over mathematical integers Z and solves non-linear recurrence systems
using Z3 CAD/Interval propagation tactics with a bounded 240 ms timeout.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple
import z3
from oeis_learn.data.models import ModularFilterCertificate


def solve_qfnia_tactical(
    unknown_count: int,
    constraints_builder: Callable[[List[z3.ArithRef]], List[z3.BoolRef]],
    bound: int = 1000,
    timeout_ms: int = 240,
) -> Tuple[bool, List[int], float]:
    """Solves non-linear integer constraints over unknown coefficients in [-bound, bound].

    Uses the tactical pipeline:
      (then simplify solve-eqs purify-arith (try-for qfnia timeout_ms))
    Returns (is_sat, constants, elapsed_ms).
    """
    start_t = time.perf_counter()
    k = unknown_count

    # 1. Allocate Z3 integer variables
    c_vars = [z3.Int(f"c_{i}") for i in range(k)]

    # 2. Bound constraints
    bounds = []
    for c in c_vars:
        bounds.append(c >= -bound)
        bounds.append(c <= bound)

    # 3. Domain constraints
    domain_constraints = constraints_builder(c_vars)

    # 4. Tactical solver pipeline
    tactic = z3.Then(
        "simplify",
        "solve-eqs",
        "purify-arith",
        z3.TryFor(z3.Tactic("qfnia"), timeout_ms),
    )
    solver = tactic.solver()
    for b in bounds:
        solver.add(b)
    for c in domain_constraints:
        solver.add(c)

    # Set parameters
    solver.set("timeout", timeout_ms)

    check_res = solver.check()
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    if check_res == z3.sat:
        model = solver.model()
        constants = []
        for c in c_vars:
            val = model.eval(c, model_completion=True)
            constants.append(val.as_long())
        return True, constants, elapsed_ms
    else:
        return False, [], elapsed_ms
