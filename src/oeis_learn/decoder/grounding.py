"""Shared bounded grounding service. Finite fits are not sequence proofs.

The supported straight-line subset has explicit scalar bit-vector and checked
whole-i256 semantics. Unsupported control flow never becomes an UNSAT claim.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import hashlib
import time
from oeis_learn.decoder.wat_grammar import tokenize_wat
from oeis_learn.sandbox.contained import run_contained


class Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class GroundingResult:
    outcome: str
    source: str
    constants: tuple[int, ...]
    fit_indices: tuple[int, ...]
    visible_prompt_sha256: str
    method: str
    semantics: str
    bound: int
    elapsed_ms: float
    reason: str | None = None
    execution: dict | None = None
    solver_version: str = "not_run"

    def to_dict(self):
        return asdict(self)


def source_hash(text):
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def splice(source, constants):
    values = iter(constants)
    words = []
    for word in tokenize_wat(source):
        if word in ("i64.const_?", "i256.const_?"):
            words.extend((word[:-2], str(next(values))))
        else:
            words.append(word)
    return " ".join(words)


def instructions(source):
    if not isinstance(source, str) or len(source.encode()) > 65536 or ";" in source:
        raise Unsupported("bounded uncommented source required")
    tokens = tokenize_wat(source)
    if len(tokens) > 2048:
        raise Unsupported("token limit")
    stack = [[]]
    for token in tokens:
        if token == "(":
            child = []
            stack[-1].append(child)
            stack.append(child)
            if len(stack) > 64:
                raise Unsupported("nesting limit")
        elif token == ")":
            if len(stack) == 1:
                raise Unsupported("unbalanced source")
            stack.pop()
        else:
            stack[-1].append(token)
    if len(stack) != 1:
        raise Unsupported("unbalanced source")
    nodes = stack[0]
    legacy = bool(nodes and isinstance(nodes[0], list) and nodes[0][:1] == ["module"])
    locals_ = {"$n": 32}
    if legacy:
        if len(nodes) != 1 or len(nodes[0]) != 2 or not isinstance(nodes[0][1], list) or nodes[0][1][:1] != ["func"]:
            raise Unsupported("one scalar function required")
        nodes = nodes[0][1][1:]
        declarations = [node for node in nodes if isinstance(node, list) and node and node[0] in ("param", "result", "export", "local")]
        if [d for d in declarations if d[0] == "param"] != [["param", "$n", "i32"]] or [d for d in declarations if d[0] == "result"] != [["result", "i64"]] or [d for d in declarations if d[0] == "export"] != [["export", '"compute"']]:
            raise Unsupported("unsupported function signature")
        for decl in declarations:
            if decl[0] == "local":
                if len(decl) != 3 or decl[1] in locals_ or not decl[1].startswith("$") or decl[2] not in ("i32", "i64"):
                    raise Unsupported("invalid local declaration")
                locals_[decl[1]] = int(decl[2][1:])
        nodes = [node for node in nodes if node not in declarations]
    else:
        locals_.update({f"$v{i}": 64 for i in range(32)})
        locals_.update({f"$c{i}": 32 for i in range(8)})
    words = []
    def flatten(node):
        if isinstance(node, str):
            words.append(node)
            return
        if not node or not isinstance(node[0], str):
            raise Unsupported("invalid folded instruction")
        head, *args = node
        immediate = head.startswith("local.") or head in ("i32.const", "i64.const", "i256.const")
        if immediate:
            if not args or not isinstance(args[0], str):
                raise Unsupported("missing immediate")
            immediate_value, *args = args
        for arg in args:
            flatten(arg)
        words.append(head)
        if immediate:
            words.append(immediate_value)
    for node in nodes:
        flatten(node)
    return words, locals_, legacy


def translate(source, index, constants):
    import z3
    words, locals_, legacy = instructions(source)
    env = {name: (width, z3.BitVecVal(index if name == "$n" else 0, width), 0) for name, width in locals_.items()}
    stack, guards = [], []
    position = parameter = 0
    def pop(width=None):
        if not stack:
            raise Unsupported("stack underflow")
        item = stack.pop()
        if width is not None and item[0] != width:
            raise Unsupported("stack type mismatch")
        return item
    while position < len(words):
        op = words[position]
        position += 1
        if op in ("i64.const_?", "i256.const_?"):
            if parameter >= len(constants):
                raise Unsupported("parameter mismatch")
            c = constants[parameter]
            parameter += 1
            width = 64 if op.startswith("i64") else 256
            if width == 256 and legacy:
                raise Unsupported("wide value in scalar function")
            value = z3.Int2BV(c, 64) if width == 64 and z3.is_int(c) else c
            stack.append((width, value, 1))
        elif op in ("i32.const", "i64.const", "i256.const"):
            width = int(op[1:op.index(".")])
            value = int(words[position]); position += 1
            if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
                raise Unsupported("noncanonical signed literal")
            stack.append((width, z3.IntVal(value) if width == 256 else z3.BitVecVal(value, width), 0))
        elif op in ("local.get", "local.set", "local.tee"):
            name = words[position]; position += 1
            if name not in env:
                raise Unsupported("undeclared local")
            if op == "local.get":
                stack.append(env[name])
            else:
                env[name] = pop(env[name][0])
                if op == "local.tee":
                    stack.append(env[name])
        elif op in ("i64.extend_i32_s", "i64.extend_i32_u"):
            _, value, degree = pop(32)
            stack.append((64, z3.SignExt(32, value) if op.endswith("_s") else z3.ZeroExt(32, value), degree))
        elif op == "nop":
            pass
        elif op == "drop":
            pop()
        elif op in ("i32.add", "i32.sub", "i32.mul", "i64.add", "i64.sub", "i64.mul", "i256.add", "i256.sub", "i256.mul_scalar"):
            width = int(op[1:op.index(".")])
            _, right, rd = pop(64 if op.endswith("mul_scalar") else width)
            _, left, ld = pop(width)
            if op.endswith("mul_scalar"):
                right = right.arg(0) if right.decl().kind() == z3.Z3_OP_INT2BV else z3.BV2Int(right, is_signed=True)
            if op.endswith("add"):
                value, degree = left + right, max(ld, rd)
            elif op.endswith("sub"):
                value, degree = left - right, max(ld, rd)
            else:
                value, degree = left * right, ld + rd
            if width == 256:
                guards.extend((value >= -(1 << 255), value < (1 << 255)))
            stack.append((width, value, degree))
        else:
            raise Unsupported(f"unsupported operation: {op}")
    if len(stack) != 1 or stack[0][0] != (64 if legacy else 256) or parameter != len(constants):
        raise Unsupported("invalid final stack or parameter count")
    return stack[0][1], stack[0][2], guards, legacy


def affine(source):
    import z3
    try:
        k = sum(w in ("i64.const_?", "i256.const_?") for w in tokenize_wat(source))
        if k > 8:
            return False
        return translate(source, 0, [z3.BitVec(f"c{i}", 64) if "(module" in source.replace(" ", "") else z3.Int(f"c{i}") for i in range(k)])[1] <= 1
    except (ValueError, IndexError, z3.Z3Exception):
        return False


def _ground(source, terms, bound, timeout_ms, recurrence_data=None):
    import z3
    from oeis_learn.sandbox.pipeline import Runtime
    from oeis_learn.decoder.dixon_solver import solve_exact_bounded
    from oeis_learn.decoder.recurrence import recognize, reference
    from oeis_learn.sandbox.runner import WasmRunner
    k = sum(w in ("i64.const_?", "i256.const_?") for w in tokenize_wat(source))
    method, semantics = "not_run", "not_run"
    def result(outcome, constants=(), reason=None, execution=None):
        return GroundingResult(outcome, splice(source, constants) if outcome == "verified_solution" else source, tuple(constants), tuple(range(len(terms))), source_hash(repr(terms)), method, semantics, bound, 0, reason, execution, z3.get_version_string())
    if k > 8:
        return result("unsupported", reason="at most eight parameters")
    rec = recognize(source)
    if rec is not None:
        method, semantics = "exact_recurrence_order2", "checked_i256"
        if len(terms) < 3 or tuple(terms[:2]) != rec:
            return result("unsupported", reason="recurrence initial conditions or data insufficient")
        matrix, rhs = recurrence_data
        solved = solve_exact_bounded(matrix, rhs, 2, bound, timeout_ms)
        if solved.outcome != "verified_solution":
            return result(solved.outcome, reason=solved.reason)
        values = solved.constants
        try:
            expected = reference(rec, values, len(terms))
        except OverflowError:
            return result("unknown", reason="recurrence intermediate exceeds signed i256")
        run = WasmRunner(fuel_budget=1_000_000).run_single(splice(source, values), terms_to_generate=len(terms), result_profile="i256x4_v1")
        if run.status != "SUCCESS" or run.output != list(terms) or expected != list(terms):
            return result("unknown", reason="final recurrence verification failed")
        return result("verified_solution", values, execution={"production":run.output,"reference":expected,"scope":"visible_prefix"})
    legacy = source.lstrip().startswith("(module") or source.lstrip().startswith("( module")
    if k == 0 and not legacy:
        method, semantics = "execute_empty_parameter_domain", "checked_i256"
        evidence = Runtime().evaluate(source, list(range(len(terms))))
        if evidence.outcome:
            return result("unknown", reason=evidence.reason)
        fits = evidence.outputs == evidence.reference_outputs == [str(t) for t in terms]
        return result("verified_solution" if fits else "proved_unsat_in_scope", execution=evidence.to_dict())
    try:
        _, _, legacy = instructions(source)
        zero = splice(source, [0] * k)
        if legacy:
            import wasmtime
            config = wasmtime.Config(); config.parallel_compilation = False
            config.memory_reservation = 0; config.memory_guard_size = 0
            with wasmtime.Engine(config) as engine:
                wasmtime.Module.validate(engine, wasmtime.wat2wasm(zero))
        else:
            from oeis_learn.sandbox.wat_ast import parse
            parse(zero)
        cs = [z3.BitVec(f"c{i}", 64) if legacy else z3.Int(f"c{i}") for i in range(k)]
        expressions, guards, degrees = [], [], []
        for i in range(len(terms)):
            expr, degree, gs, _ = translate(source, i, cs)
            expressions.append(z3.simplify(expr)); guards.extend(z3.simplify(g) for g in gs); degrees.append(degree)
    except (ValueError, IndexError, z3.Z3Exception) as exc:
        return result("unsupported", reason=str(exc))
    def has_bv(expr):
        return z3.is_bv(expr) or any(has_bv(c) for c in expr.children())
    mixed = any(has_bv(e) for e in expressions + guards)
    method = "z3_qfbv" if legacy else "z3_int_bv" if mixed else "z3_qfnia"
    semantics = "modular_i64" if legacy else "checked_i256"
    if legacy and any(not -(1 << 63) <= t < (1 << 63) for t in terms):
        return result("proved_unsat_in_scope", reason="target outside scalar result range")
    solver = z3.Solver() if method == "z3_int_bv" else z3.SolverFor("QF_BV" if legacy else "QF_NIA")
    solver.set(timeout=timeout_ms)
    solver.add(*[g for c in cs for g in (c >= -bound, c <= bound)], *guards)
    solver.add(*[e == t for e, t in zip(expressions, terms)])
    values = None
    if not legacy and not mixed and all(d <= 1 for d in degrees):
        zero = [(c, z3.IntVal(0)) for c in cs]
        offsets = [z3.simplify(z3.substitute(e, *zero)).as_long() if cs else e.as_long() for e in expressions]
        matrix = [[z3.simplify(z3.substitute(e, *[(c, z3.IntVal(int(j == column))) for j, c in enumerate(cs)])).as_long() - offset for column in range(k)] for e, offset in zip(expressions, offsets)]
        solved = solve_exact_bounded(matrix, [t - o for t, o in zip(terms, offsets)], k, bound, timeout_ms)
        if solved.outcome == "proved_unsat_in_scope":
            method = solved.method
            return result(solved.outcome)
        if solved.outcome == "verified_solution":
            values = solved.constants
            # Affine solution must still satisfy EVERY intermediate overflow guard.
            if not all(z3.is_true(z3.simplify(z3.substitute(g, *[(c, z3.IntVal(v)) for c, v in zip(cs, values)]))) for g in guards):
                values = None
            else:
                method = solved.method
    if values is None and method == "z3_qfnia":
        from oeis_learn.decoder.qfnia_solver import solve_integer_constraints
        constraints = guards + [e == t for e, t in zip(expressions, terms)]
        def build(fresh):
            substitutions = list(zip(cs, fresh))
            return [z3.substitute(c, *substitutions) if substitutions else c for c in constraints]
        solved = solve_integer_constraints(k, build, bound, timeout_ms)
        if solved.outcome != "verified_solution":
            return result(solved.outcome, reason=solved.reason)
        values = solved.constants
    if values is None:
        status = solver.check()
        if status == z3.unsat:
            return result("proved_unsat_in_scope")
        if status == z3.unknown:
            reason = solver.reason_unknown()
            return result("timeout" if "timeout" in reason.lower() else "unknown", reason=reason)
        values = tuple(solver.model().eval(c, model_completion=True).as_signed_long() if legacy else solver.model().eval(c, model_completion=True).as_long() for c in cs)
    grounded = splice(source, values)
    if legacy:
        from oeis_learn.decoder.grounding_reference import scalar_value
        production = WasmRunner().run_single(grounded, terms_to_generate=len(terms))
        independent = [scalar_value(grounded, n) for n in range(len(terms))]
        if production.status != "SUCCESS" or production.output != independent or independent != list(terms):
            return result("unknown", reason="final production/reference verification failed")
        evidence = {"production":production.output,"reference":independent,"scope":"visible_prefix"}
    else:
        production = Runtime().evaluate(grounded, list(range(len(terms))))
        if production.outcome or production.outputs != production.reference_outputs or production.outputs != [str(t) for t in terms]:
            return result("unknown", reason="final production/reference verification failed")
        evidence = production.to_dict()
    return result("verified_solution", values, execution=evidence)


def ground(source, terms, *, bound=1000, timeout_ms=240):
    started = time.monotonic()
    if type(bound) is not int or not 0 <= bound <= 1000 or type(timeout_ms) is not int or not 1 <= timeout_ms <= 2000:
        raise ValueError("bounds: coefficients0..1000, solver1..2000 ms")
    visible = tuple(terms[:20])
    if not visible or any(type(t) is not int or not -(1 << 255) <= t < (1 << 255) for t in visible):
        raise ValueError("one to twenty exact signed i256 terms required")
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise ValueError("source exceeds64KiB")
    from oeis_learn.decoder.recurrence import recognize, lag_system
    recurrence_data = lag_system(visible) if recognize(source) is not None and len(visible) >= 3 else None
    status, value = run_contained(_ground, (source, visible, bound, timeout_ms, recurrence_data))
    elapsed = (time.monotonic() - started) * 1000
    if status == "ok":
        return replace(value, elapsed_ms=elapsed)
    return GroundingResult("timeout" if status == "timeout" else "unknown", source, (), tuple(range(len(visible))), source_hash(repr(visible)), "not_run", "not_run", bound, elapsed, str(value))
