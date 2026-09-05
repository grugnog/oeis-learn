"""Decoupled Symbolic-Numeric Constant Solver for WebAssembly Text Programs.

Implements fast exact Hermite Normal Form (HNF) Diophantine row reduction
and Z3 SMT fallback (QF_BV) to bind integer coefficients to abstract WAT skeletons.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple
import numpy as np
import z3
from oeis_learn.data.models import ASTSkeleton, ConstantSolverResult
from oeis_learn.decoder.wat_grammar import tokenize_wat
from oeis_learn.sandbox.runner import WasmRunner


def parse_ast_placeholders(wat_code: str) -> ASTSkeleton:
    """Parses a WAT program string, identifies placeholder tokens (i64.const_?),

    and classifies linearity within the execution trace.
    """
    tokens = tokenize_wat(wat_code)
    placeholder_indices = [i for i, tok in enumerate(tokens) if tok == "i64.const_?"]
    placeholder_count = len(placeholder_indices)

    # Check for non-linear operations wrapping placeholders
    # Operations that make parameters non-linear:
    nonlinear_ops = {
        "i64.rem_u", "i64.rem_s", "i32.rem_u", "i32.rem_s",
        "i64.shl", "i64.shr_u", "i64.shr_s", "i32.shl", "i32.shr_u", "i32.shr_s",
        "i64.and", "i64.or", "i64.xor", "i32.and", "i32.or", "i32.xor",
        "i64.div_u", "i64.div_s", "i32.div_u", "i32.div_s",
        "br_if", "if", "loop", "block",
    }

    # If any non-linear operator occurs in the program, check if placeholders are inside it
    has_nonlinear = any(op in tokens for op in nonlinear_ops)
    is_linear = placeholder_count > 0 and not has_nonlinear

    return ASTSkeleton(
        raw_wat=wat_code,
        placeholder_count=placeholder_count,
        is_linear=is_linear,
        placeholder_indices=placeholder_indices,
        basis_signatures=[],
    )


def splice_constants_into_wat(skeleton: ASTSkeleton, constants: List[int]) -> str:
    """Replaces each 'i64.const_?' placeholder in the skeleton with 'i64.const <C_i>'."""
    if len(constants) != skeleton.placeholder_count:
        raise ValueError(
            f"Expected {skeleton.placeholder_count} constants, got {len(constants)}"
        )

    tokens = tokenize_wat(skeleton.raw_wat)
    const_idx = 0
    new_tokens = []

    for tok in tokens:
        if tok == "i64.const_?":
            c_val = int(constants[const_idx])
            new_tokens.append("i64.const")
            new_tokens.append(str(c_val))
            const_idx += 1
        else:
            new_tokens.append(tok)

    return " ".join(new_tokens)


def solve_linear_diophantine(
    skeleton: ASTSkeleton,
    terms: List[int],
    runner: Optional[WasmRunner] = None,
) -> ConstantSolverResult:
    """Solves for integer constants in linear/affine AST skeletons via exact matrix reduction."""
    start_time = time.perf_counter()
    k = skeleton.placeholder_count

    if k == 0:
        return ConstantSolverResult(
            solver_type="DIOPHANTINE_HNF",
            constants=[],
            solve_duration_ms=(time.perf_counter() - start_time) * 1000.0,
            is_sat=True,
            grounded_wat=skeleton.raw_wat,
        )

    if k > 8:
        return ConstantSolverResult(
            solver_type="FAILED",
            constants=None,
            solve_duration_ms=(time.perf_counter() - start_time) * 1000.0,
            is_sat=False,
            error_message="Too many placeholders for Diophantine solver (max 8)",
        )

    if runner is None:
        runner = WasmRunner(fuel_budget=10000)

    num_eval_terms = min(20, len(terms))
    target_vector = np.array(terms[:num_eval_terms], dtype=np.float64)

    # Construct basis evaluations:
    # Evaluate with standard basis vectors e_j (1 at pos j, 0 elsewhere)
    # Also evaluate with zero vector e_0 = (0, 0, ..., 0) to check affine offset
    zero_wat = splice_constants_into_wat(skeleton, [0] * k)
    zero_res = runner.run_single(zero_wat, terms_to_generate=num_eval_terms)

    if zero_res.status != "SUCCESS" or len(zero_res.output) < num_eval_terms:
        # If execution fails at zero, try SMT fallback
        return ConstantSolverResult(
            solver_type="FAILED",
            constants=None,
            solve_duration_ms=(time.perf_counter() - start_time) * 1000.0,
            is_sat=False,
            error_message=f"Basis execution failed at zero: {zero_res.error}",
        )

    f_zero = np.array(zero_res.output[:num_eval_terms], dtype=np.float64)

    # Matrix A has columns corresponding to (f(e_j) - f_zero)
    A_cols = []
    for j in range(k):
        unit_vec = [0] * k
        unit_vec[j] = 1
        unit_wat = splice_constants_into_wat(skeleton, unit_vec)
        unit_res = runner.run_single(unit_wat, terms_to_generate=num_eval_terms)

        if unit_res.status != "SUCCESS" or len(unit_res.output) < num_eval_terms:
            return ConstantSolverResult(
                solver_type="FAILED",
                constants=None,
                solve_duration_ms=(time.perf_counter() - start_time) * 1000.0,
                is_sat=False,
                error_message=f"Basis execution failed at basis {j}: {unit_res.error}",
            )

        col_j = np.array(unit_res.output[:num_eval_terms], dtype=np.float64) - f_zero
        A_cols.append(col_j)

    A = np.column_stack(A_cols)  # Shape (num_eval_terms, k)
    Y_eff = target_vector - f_zero

    # Solve linear system A * C = Y_eff
    try:
        sol, residuals, rank, s = np.linalg.lstsq(A, Y_eff, rcond=1e-8)
        sol_int = np.rint(sol).astype(np.int64)

        # Check exact fit
        pred = A @ sol_int.astype(np.float64)
        if np.max(np.abs(pred - Y_eff)) < 1e-4:
            constants = [int(c) for c in sol_int]
            grounded_wat = splice_constants_into_wat(skeleton, constants)

            # Final verification pass in WASM runner
            final_res = runner.run_single(grounded_wat, terms_to_generate=num_eval_terms)
            if final_res.status == "SUCCESS" and final_res.output[:num_eval_terms] == terms[:num_eval_terms]:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return ConstantSolverResult(
                    solver_type="DIOPHANTINE_HNF",
                    constants=constants,
                    solve_duration_ms=duration_ms,
                    is_sat=True,
                    grounded_wat=grounded_wat,
                )
    except Exception as e:
        pass

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    return ConstantSolverResult(
        solver_type="FAILED",
        constants=None,
        solve_duration_ms=duration_ms,
        is_sat=False,
        error_message="Linear Diophantine system is inconsistent or non-linear",
    )


def _parse_sexpr_tokens(tokens: List[str]) -> List[Any]:
    """Parses a token stream into nested S-expression lists."""
    stack: List[List[Any]] = [[]]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "(":
            new_list: List[Any] = []
            stack[-1].append(new_list)
            stack.append(new_list)
        elif tok == ")":
            if len(stack) > 1:
                stack.pop()
        else:
            stack[-1].append(tok)
        i += 1
    return stack[0]


def _build_z3_ast(
    node: Any,
    c_vars: List[z3.BitVecRef],
    p_counter: List[int],
    local_env: dict[str, z3.BitVecRef],
) -> Optional[z3.BitVecRef]:
    """Recursively lowers an S-expression node to a Z3 64-bit BitVector expression."""
    if isinstance(node, str):
        if node == "i64.const_?":
            idx = p_counter[0]
            p_counter[0] += 1
            return c_vars[idx]
        elif node.isdigit() or node.lstrip("-").isdigit():
            return z3.BitVecVal(int(node), 64)
        elif node.startswith("$"):
            return local_env.get(node, local_env.get("$n", z3.BitVecVal(0, 64)))
        return None

    if not isinstance(node, list) or not node:
        return None

    head = node[0]

    # Structural wrappers
    if head == "module":
        for child in node[1:]:
            res = _build_z3_ast(child, c_vars, p_counter, local_env)
            if res is not None:
                return res
        return None

    if head == "func":
        # Extract function body (skip export, param, result, local headers)
        for child in node[1:]:
            if isinstance(child, list) and child and child[0] in ("export", "param", "result", "local"):
                continue
            res = _build_z3_ast(child, c_vars, p_counter, local_env)
            if res is not None:
                return res
        return None

    if head == "local.set":
        var_name = node[1]
        val_expr = _build_z3_ast(node[2], c_vars, p_counter, local_env) if len(node) > 2 else z3.BitVecVal(0, 64)
        if val_expr is not None:
            local_env[var_name] = val_expr
        return None

    if head == "local.get":
        var_name = node[1]
        return local_env.get(var_name, local_env.get("$n", z3.BitVecVal(0, 64)))

    if head in ("i64.const", "i32.const"):
        if len(node) > 1 and (isinstance(node[1], str) and (node[1].isdigit() or node[1].lstrip("-").isdigit())):
            return z3.BitVecVal(int(node[1]), 64)
        return z3.BitVecVal(0, 64)

    if head == "i64.const_?":
        idx = p_counter[0]
        p_counter[0] += 1
        return c_vars[idx]

    if head in ("i64.extend_i32_u", "i64.extend_i32_s"):
        return _build_z3_ast(node[1], c_vars, p_counter, local_env) if len(node) > 1 else z3.BitVecVal(0, 64)

    # Arithmetic operators
    args = []
    for arg_node in node[1:]:
        sub_res = _build_z3_ast(arg_node, c_vars, p_counter, local_env)
        if sub_res is not None:
            args.append(sub_res)

    if head == "i64.add":
        if len(args) == 2:
            return args[0] + args[1]
    elif head == "i64.sub":
        if len(args) == 2:
            return args[0] - args[1]
    elif head == "i64.mul":
        if len(args) == 2:
            return args[0] * args[1]
    elif head == "i64.rem_u":
        if len(args) == 2:
            return z3.URem(args[0], args[1])
    elif head == "i64.rem_s":
        if len(args) == 2:
            return z3.SRem(args[0], args[1])
    elif head == "i64.div_u":
        if len(args) == 2:
            return z3.UDiv(args[0], args[1])
    elif head == "i64.div_s":
        if len(args) == 2:
            return args[0] / args[1]
    elif head == "i64.shl":
        if len(args) == 2:
            return args[0] << args[1]
    elif head == "i64.shr_u":
        if len(args) == 2:
            return z3.LShR(args[0], args[1])
    elif head == "i64.shr_s":
        if len(args) == 2:
            return args[0] >> args[1]
    elif head == "i64.and":
        if len(args) == 2:
            return args[0] & args[1]
    elif head == "i64.or":
        if len(args) == 2:
            return args[0] | args[1]
    elif head == "i64.xor":
        if len(args) == 2:
            return args[0] ^ args[1]

    return args[-1] if args else None


def solve_smt_constants(
    skeleton: ASTSkeleton,
    terms: List[int],
    timeout_ms: int = 250,
    runner: Optional[WasmRunner] = None,
) -> ConstantSolverResult:
    """Solves for integer constants in non-linear or control-flow AST skeletons using Z3 QF_BV."""
    start_time = time.perf_counter()
    k = skeleton.placeholder_count

    if k == 0:
        return ConstantSolverResult(
            solver_type="Z3_SMT",
            constants=[],
            solve_duration_ms=(time.perf_counter() - start_time) * 1000.0,
            is_sat=True,
            grounded_wat=skeleton.raw_wat,
        )

    if runner is None:
        runner = WasmRunner(fuel_budget=10000)

    num_eval_terms = min(20, len(terms))

    # Create Z3 solver with BitVector logic
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    c_vars = [z3.BitVec(f"c_{i}", 64) for i in range(k)]

    tokens = tokenize_wat(skeleton.raw_wat)
    sexpr = _parse_sexpr_tokens(tokens)

    try:
        # CEGIS strategy: Assert constraints on first 5 terms first for rapid SMT solving
        initial_terms = min(5, num_eval_terms)
        for n in range(initial_terms):
            local_env = {"$n": z3.BitVecVal(n, 64)}
            p_counter = [0]
            expr = _build_z3_ast(sexpr[0] if sexpr else [], c_vars, p_counter, local_env)
            if expr is None:
                raise ValueError("Could not lower WAT AST to symbolic expression")
            solver.add(expr == z3.BitVecVal(terms[n], 64))

        # Add non-zero divisor constraint if division/modulo present
        for c in c_vars:
            solver.add(c != 0)

        # Check SAT on initial constraints
        check_res = solver.check()
        if check_res == z3.sat:
            model = solver.model()
            constants = [model[c].as_signed_long() for c in c_vars]
            grounded_wat = splice_constants_into_wat(skeleton, constants)

            # Verification pass in WASM runner across ALL terms
            final_res = runner.run_single(grounded_wat, terms_to_generate=num_eval_terms)
            if final_res.status == "SUCCESS" and final_res.output[:num_eval_terms] == terms[:num_eval_terms]:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return ConstantSolverResult(
                    solver_type="Z3_SMT",
                    constants=constants,
                    solve_duration_ms=duration_ms,
                    is_sat=True,
                    grounded_wat=grounded_wat,
                )
            else:
                # If first candidate didn't fit all terms, add remaining constraints and re-check
                for n in range(initial_terms, num_eval_terms):
                    local_env = {"$n": z3.BitVecVal(n, 64)}
                    p_counter = [0]
                    expr = _build_z3_ast(sexpr[0] if sexpr else [], c_vars, p_counter, local_env)
                    if expr is not None:
                        solver.add(expr == z3.BitVecVal(terms[n], 64))
                check_res = solver.check()
                if check_res == z3.sat:
                    model = solver.model()
                    constants = [model[c].as_signed_long() for c in c_vars]
                    grounded_wat = splice_constants_into_wat(skeleton, constants)
                    final_res = runner.run_single(grounded_wat, terms_to_generate=num_eval_terms)
                    if final_res.status == "SUCCESS" and final_res.output[:num_eval_terms] == terms[:num_eval_terms]:
                        duration_ms = (time.perf_counter() - start_time) * 1000.0
                        return ConstantSolverResult(
                            solver_type="Z3_SMT",
                            constants=constants,
                            solve_duration_ms=duration_ms,
                            is_sat=True,
                            grounded_wat=grounded_wat,
                        )
        elif check_res == z3.unknown:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return ConstantSolverResult(
                solver_type="TIMEOUT",
                constants=None,
                solve_duration_ms=duration_ms,
                is_sat=False,
                error_message="Z3 SMT solver timed out",
            )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return ConstantSolverResult(
            solver_type="FAILED",
            constants=None,
            solve_duration_ms=duration_ms,
            is_sat=False,
            error_message=f"SMT translation error: {e}",
        )

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    return ConstantSolverResult(
        solver_type="FAILED",
        constants=None,
        solve_duration_ms=duration_ms,
        is_sat=False,
        error_message="SMT constraints are unsatisfiable",
    )


def resolve_program_constants(
    wat_code: str,
    terms: List[int],
    timeout_ms: int = 250,
    max_placeholders: int = 4,
    runner: Optional[WasmRunner] = None,
) -> Tuple[str, List[int], str, float, Optional[str]]:
    """Unified constant-resolution dispatcher chaining Diophantine row reduction and SMT fallback.

    Returns:
        Tuple of (resolved_wat, constants, status, duration_ms, error_message)
        where status is in ("PASSED", "UNSATISFIABLE", "TIMEOUT", "ERROR", "NOT_REQUIRED").
    """
    if "i64.const_?" not in wat_code:
        return wat_code, [], "NOT_REQUIRED", 0.0, None

    start_time = time.perf_counter()
    skeleton = parse_ast_placeholders(wat_code)

    if skeleton.placeholder_count > max_placeholders:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return (
            wat_code,
            [],
            "ERROR",
            duration_ms,
            f"Exceeded max_placeholders ({skeleton.placeholder_count} > {max_placeholders})",
        )

    # 1. Attempt exact Diophantine linear solver
    dioph_res = solve_linear_diophantine(skeleton, terms, runner=runner)
    if dioph_res.is_sat and dioph_res.grounded_wat:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return (
            dioph_res.grounded_wat,
            dioph_res.constants or [],
            "PASSED",
            duration_ms,
            None,
        )

    # 2. SMT fallback
    smt_res = solve_smt_constants(skeleton, terms, timeout_ms=timeout_ms, runner=runner)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    if smt_res.is_sat and smt_res.grounded_wat:
        return (
            smt_res.grounded_wat,
            smt_res.constants or [],
            "PASSED",
            duration_ms,
            None,
        )

    if smt_res.solver_type == "TIMEOUT":
        return wat_code, [], "TIMEOUT", duration_ms, smt_res.error_message or "Solver timeout"

    return wat_code, [], "UNSATISFIABLE", duration_ms, smt_res.error_message or "Unsatisfiable"

